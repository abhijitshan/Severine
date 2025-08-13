#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <random>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <limits>
#include "../../include/hyperparameter.hpp"
#include "../../include/modelAdapters.hpp"
#include "../../include/randomSearch.hpp"
#include "../../include/bayesianOptimization.hpp"
#include "../../include/tuner.hpp"
const int ITERATIONS = 100;
const int THREADS = 1;
const std::string OUTPUT_DIR = "hypertune_benchmark_results";
#ifdef __APPLE__
#include <sys/resource.h>
struct MemoryStats {
    size_t current_usage_kb = 0;
    size_t peak_usage_kb = 0;
    void update() {
        struct rusage rusage;
        if (getrusage(RUSAGE_SELF, &rusage) == 0) {
            current_usage_kb = static_cast<size_t>(rusage.ru_maxrss / 1024);
            peak_usage_kb = std::max(peak_usage_kb, current_usage_kb);
        } else {
             current_usage_kb = 0;
        }
    }
     size_t getCurrentMemoryUsageKB() const {
         struct rusage rusage;
         if (getrusage(RUSAGE_SELF, &rusage) == 0) {
             return static_cast<size_t>(rusage.ru_maxrss / 1024);
         }
         return 0;
     }
};
#else
struct MemoryStats {
    size_t current_usage_kb = 0;
    size_t peak_usage_kb = 0;
    void update() {  current_usage_kb=0; peak_usage_kb=0;}
    size_t getCurrentMemoryUsageKB() const { return 0;  }
};
#endif
struct BenchmarkResult {
    std::vector<double> scores;
    std::vector<double> best_score_history;
    std::vector<size_t> memory_usage_kb;
    double total_time = 0;
    size_t peak_memory_kb = 0;
    severine::Config best_config;
    double best_score = -std::numeric_limits<double>::infinity();
};
void generateSyntheticData(
    std::vector<std::vector<double>>& X_train,
    std::vector<double>& y_train,
    std::vector<std::vector<double>>& X_test,
    std::vector<double>& y_test,
    int n_samples = 1000,
    int n_features = 20,
    double noise = 0.1)
{
    std::mt19937 rng(42); // Fixed seed 42
    std::normal_distribution<double> noise_dist(0.0, noise);
    std::vector<double> coefficients(n_features);
    std::uniform_real_distribution<double> coef_dist(-1.0, 1.0);
    for (int i = 0; i < n_features; ++i) {
        coefficients[i] = coef_dist(rng);
    }
    X_train.resize(n_samples);
    y_train.resize(n_samples);
    std::uniform_real_distribution<double> feature_dist(-5.0, 5.0);
    for (int i = 0; i < n_samples; ++i) {
        X_train[i].resize(n_features);
        y_train[i] = 0.0; // Initialize target
        for (int j = 0; j < n_features; ++j) {
            X_train[i][j] = feature_dist(rng); // Use feature_dist
            y_train[i] += X_train[i][j] * coefficients[j];
        }
        y_train[i] += noise_dist(rng);
    }
    int n_test_samples = n_samples / 5;
    X_test.resize(n_test_samples);
    y_test.resize(n_test_samples);
    for (int i = 0; i < n_test_samples; ++i) {
        X_test[i].resize(n_features);
        y_test[i] = 0.0; // Initialize target
        for (int j = 0; j < n_features; ++j) {
            X_test[i][j] = feature_dist(rng); // Use feature_dist
            y_test[i] += X_test[i][j] * coefficients[j];
        }
        y_test[i] += noise_dist(rng);
    }
}
severine::SearchSpace createLinearRegressionSearchSpace() {
    severine::SearchSpace searchSpace;
    auto alpha_param = std::make_shared<severine::FloatHyperparameter>(
        "alpha", 0.0001f, 1.0f, severine::FloatHyperparameter::Distribution::LOG_UNIFORM);
    searchSpace.addHyperparameter(alpha_param);
    auto fit_intercept_param = std::make_shared<severine::BooleanHyperparameter>("fit_intercept");
    searchSpace.addHyperparameter(fit_intercept_param);
    auto max_iter_param = std::make_shared<severine::IntegerHyperparameter>("max_iter", 100, 5000); // Increased range
    searchSpace.addHyperparameter(max_iter_param);
    auto tol_param = std::make_shared<severine::FloatHyperparameter>(
        "tol", 1e-6f, 1e-2f, severine::FloatHyperparameter::Distribution::LOG_UNIFORM);
    searchSpace.addHyperparameter(tol_param);
    return searchSpace;
}
BenchmarkResult runBenchmark(
    const std::string& strategy_name,
    std::shared_ptr<severine::ModelInterface> model, // Pass original model
    const severine::SearchSpace& searchSpace)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shared_ptr<severine::SearchStrategy> strategy;
    if (strategy_name == "random") {
        strategy = std::make_shared<severine::RandomSearch>(searchSpace, rng);
    } else if (strategy_name == "bayesian") {
        strategy = std::make_shared<severine::BayesianOptimization>(
            searchSpace, rng,
            10,                                     // initialSamples
            2.0,                                    // explorationFactor
            severine::KernelType::MATERN,
            severine::AcquisitionFunctionType::UCB,
            true);
    } else {
        throw std::invalid_argument("Unknown strategy: " + strategy_name);
    }
    severine::TunerConfig tunerConfig;
    tunerConfig.maxIterations = ITERATIONS;
    tunerConfig.numThreads = THREADS;
    tunerConfig.verbose = false;
    if (THREADS > 1) {
        tunerConfig.parallelStrategy = severine::ParallelizationStrategy::THREAD_POOL;
    } else {
        tunerConfig.parallelStrategy = severine::ParallelizationStrategy::NONE;
    }
    severine::Tuner tuner(model, strategy, tunerConfig);
    MemoryStats memStats;
    memStats.update(); // Initial memory
    BenchmarkResult result;
    result.scores.reserve(ITERATIONS);
    result.best_score_history.reserve(ITERATIONS);
    result.memory_usage_kb.reserve(ITERATIONS);
    auto start = std::chrono::high_resolution_clock::now();
    tuner.tune();
    auto end = std::chrono::high_resolution_clock::now();
    result.total_time = std::chrono::duration<double>(end - start).count();
    const auto& all_results = tuner.getAllResults(); // Get all results
    const auto& best_result = tuner.getBestResult();
    double current_best = -std::numeric_limits<double>::infinity();
    for (const auto& res : all_results) {
        result.scores.push_back(res.score);
        current_best = std::max(current_best, res.score);
        result.best_score_history.push_back(current_best);
        memStats.update();
        result.memory_usage_kb.push_back(memStats.peak_usage_kb);
    }
    result.peak_memory_kb = memStats.peak_usage_kb;
    result.best_config = best_result.configuration;
    result.best_score = best_result.score;
    return result;
}
void saveBenchmarkToCSV(
    const std::string& filename,
    const std::vector<BenchmarkResult>& results,
    const std::vector<std::string>& strategy_names)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    file << "Iteration";
    for (const auto& name : strategy_names) {
        file << "," << name << "_score," << name << "_best_score," << name << "_memory_kb"; // Changed header
    }
    file << std::endl;
    size_t max_iters = 0;
     if (!results.empty() && !results[0].scores.empty()){
         max_iters = results[0].scores.size();
     }
    for (size_t i = 0; i < max_iters; ++i) { // Use size_t for index
        file << (i + 1); // Iteration number
        for (size_t j = 0; j < strategy_names.size(); ++j) {
            bool score_ok = i < results[j].scores.size();
            bool history_ok = i < results[j].best_score_history.size();
            bool memory_ok = i < results[j].memory_usage_kb.size();
            file << "," << (score_ok ? std::to_string(results[j].scores[i]) : "");
            file << "," << (history_ok ? std::to_string(results[j].best_score_history[i]) : "");
            file << "," << (memory_ok ? std::to_string(results[j].memory_usage_kb[i]) : ""); // Use size_t memory
        }
        file << std::endl;
    }
    file.close();
    std::string summary_filename = filename.substr(0, filename.find_last_of('.')) + "_summary.csv";
    std::ofstream summary_file(summary_filename);
    if (!summary_file.is_open()) {
        std::cerr << "Error: Could not open file " << summary_filename << std::endl;
        return;
    }
    summary_file << "Strategy,Best Score,Total Time (s),Peak Memory (KB),Best Config" << std::endl; // Changed header
    for (size_t i = 0; i < strategy_names.size(); ++i) {
        summary_file << strategy_names[i] << ","
                     << results[i].best_score << ","
                     << results[i].total_time << ","
                     << results[i].peak_memory_kb << ",\""; // Use size_t memory
        bool first = true;
        for (const auto& [name, value] : results[i].best_config) {
            if (!first) {
                summary_file << "; ";
            }
            first = false;
            summary_file << name << "=";
            if (std::holds_alternative<int>(value)) {
                summary_file << std::get<int>(value);
            } else if (std::holds_alternative<float>(value)) {
                summary_file << std::get<float>(value);
            } else if (std::holds_alternative<bool>(value)) {
                summary_file << (std::get<bool>(value) ? "true" : "false");
            } else if (std::holds_alternative<std::string>(value)) {
                summary_file << std::get<std::string>(value);
            }
        }
         if (!first) summary_file << "; ";
         summary_file << "solver=lsqr";
        summary_file << "\"" << std::endl;
    }
    summary_file.close();
}
void printBenchmarkSummary(
    const std::vector<BenchmarkResult>& results,
    const std::vector<std::string>& strategy_names)
{
    std::cout << "\n=== Benchmark Summary ===\n" << std::endl;
    std::cout << std::left << std::setw(15) << "Strategy"
              << std::setw(20) << "Best Score" // Increased width
              << std::setw(18) << "Total Time (s)" // Increased width
              << std::setw(20) << "Peak Memory (KB)"
              << "Best Configuration" << std::endl;
    std::cout << std::string(100, '-') << std::endl;
    for (size_t i = 0; i < strategy_names.size(); ++i) {
        std::cout << std::left << std::setw(15) << strategy_names[i]
                  << std::setw(20) << std::fixed << std::setprecision(10) << results[i].best_score // Increased precision
                  << std::setw(18) << std::fixed << std::setprecision(4) << results[i].total_time // Increased precision
                  << std::setw(20) << results[i].peak_memory_kb; // Use size_t memory
        std::cout << "{ ";
        bool first = true;
        for (const auto& [name, value] : results[i].best_config) {
            if (!first) {
                std::cout << ", ";
            }
            first = false;
            std::cout << name << "=";
            if (std::holds_alternative<int>(value)) {
                std::cout << std::get<int>(value);
            } else if (std::holds_alternative<float>(value)) {
                 std::cout << std::fixed << std::setprecision(8) << std::get<float>(value);
            } else if (std::holds_alternative<bool>(value)) {
                std::cout << (std::get<bool>(value) ? "true" : "false");
            } else if (std::holds_alternative<std::string>(value)) {
                std::cout << "'" << std::get<std::string>(value) << "'";
            }
        }
         if (!first) std::cout << ", ";
         std::cout << "solver=lsqr"; // Manually added
        std::cout << " }" << std::endl;
    }
     std::cout << std::string(100, '-') << std::endl;
}
void benchmarkModel(
    const std::string& model_type,
    std::vector<std::vector<double>>& X_train,
    std::vector<double>& y_train,
    std::vector<std::vector<double>>& X_test,
    std::vector<double>& y_test)
{
    std::cout << "\n=== Benchmarking " << model_type << " ===" << std::endl;
    std::cout << "Iterations: " << ITERATIONS << " | Threads: " << THREADS << "\n" << std::endl;
    std::shared_ptr<severine::ModelInterface> model;
    severine::SearchSpace searchSpace;
    if (model_type == "LinearRegression") {
        model = severine::createMLFmkModel("LinearRegression");
        searchSpace = createLinearRegressionSearchSpace();
    } else {
        throw std::invalid_argument("Unsupported model type for benchmarking: " + model_type);
    }
    auto* dataModel = dynamic_cast<severine::DataMatrixModelAdapter*>(model.get());
    if (dataModel) {
        try {
             dataModel->loadTrainingData(X_train, y_train);
             dataModel->loadTestData(X_test, y_test);
        } catch (const std::exception& e) {
             std::cerr << "Error loading data into model: " << e.what() << std::endl;
             return;
        }
    } else {
        std::cerr << "Error: Model adapter does not support DataMatrix loading." << std::endl;
        return;
    }
     model->setVerbose(false);
    std::vector<std::string> strategy_names = {"random"};
    std::vector<BenchmarkResult> results;
    for (const auto& strategy_name : strategy_names) {
        std::cout << "Running benchmark for " << strategy_name << " search..." << std::endl;
        BenchmarkResult result = runBenchmark(strategy_name, model, searchSpace);
        results.push_back(result);
        std::cout << "... " << strategy_name << " search complete." << std::endl;
    }
    printBenchmarkSummary(results, strategy_names);
    std::string output_file = OUTPUT_DIR + "/" + model_type + "_benchmark.csv";
    saveBenchmarkToCSV(output_file, results, strategy_names);
    std::cout << "Results saved to:" << std::endl;
    std::cout << " - " << output_file << std::endl;
    std::cout << " - " << output_file.substr(0, output_file.find_last_of('.')) + "_summary.csv" << std::endl;
}
int toleranceLSQRTest() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    char time_str[26];
    ctime_r(&in_time_t, time_str);
    std::cout << "=== Severine Benchmarking Tool ===" << std::endl;
    std::cout << "Started at: " << time_str;
    std::cout << "Benchmark results will be saved to: " << OUTPUT_DIR << std::endl;
    try {
        if (!std::filesystem::exists(OUTPUT_DIR)) {
            if (std::filesystem::create_directory(OUTPUT_DIR)) {
                 std::cout << "Created output directory: " << OUTPUT_DIR << std::endl;
            } else {
                 std::cerr << "Error: Could not create output directory: " << OUTPUT_DIR << std::endl;
                 return 1;
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
         std::cerr << "Filesystem error: " << e.what() << std::endl;
         return 1;
    }
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;
    std::cout << "\nGenerating synthetic dataset..." << std::endl;
    generateSyntheticData(X_train, y_train, X_test, y_test, 1000, 20, 0.1);
    if (X_train.empty() || X_train[0].empty()){
         std::cerr << "Error: Failed to generate synthetic data." << std::endl;
         return 1;
    }
    std::cout << "Dataset generated: " << X_train.size() << " training samples, "
              << X_test.size() << " test samples, " << X_train[0].size() << " features." << std::endl;
    try{
         benchmarkModel("LinearRegression", X_train, y_train, X_test, y_test);
    } catch (const std::exception& e) {
         std::cerr << "\n*** An error occurred during benchmarking: " << e.what() << " ***" << std::endl;
         return 1;
    } catch (...) {
         std::cerr << "\n*** An unknown error occurred during benchmarking ***" << std::endl;
         return 1;
    }
    std::cout << "\nBenchmarking complete!" << std::endl;
    return 0;
}