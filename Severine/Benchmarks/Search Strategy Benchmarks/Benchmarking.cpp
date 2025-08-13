#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>
#include <filesystem>
#include <mach-o/dyld.h>
#include "../../include/hyperparameter.hpp"
#include "../../include/modelAdapters.hpp"
#include "../../include/randomSearch.hpp"
#include "../../include/bayesianOptimization.hpp"
#include "../../include/tuner.hpp"
namespace severinebench {
std::string getExecutableDirectory() {
    char path[1024];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) == 0) {
        std::string execPath(path);
        size_t lastSlash = execPath.find_last_of("/\\");
        if (lastSlash != std::string::npos) {
            return execPath.substr(0, lastSlash);
        }
    }
    return ".";
}
std::string getOutputDirectory() {
    std::string homeDir = std::getenv("HOME") ? std::getenv("HOME") : "";
    if (!homeDir.empty()) {
        std::string desktopDir = homeDir + "/Desktop/SeverineBenchmarks";
        try {
            std::filesystem::create_directories(desktopDir);
            return desktopDir;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not create directory on Desktop: " << e.what() << std::endl;
        }
    }
    std::string execDir = getExecutableDirectory();
    std::string benchmarkDir = execDir + "/SeverineBenchmarks";
    try {
        std::filesystem::create_directories(benchmarkDir);
        return benchmarkDir;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not create directory in executable path: " << e.what() << std::endl;
    }
    return ".";
}
void generateSyntheticData(
    std::vector<std::vector<double>>& X_train,
    std::vector<double>& y_train,
    std::vector<std::vector<double>>& X_test,
    std::vector<double>& y_test,
    int n_samples = 1000,
    int n_features = 10,
    double noise = 0.1) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<double> noise_dist(0.0, noise);
    std::vector<double> coefficients(n_features);
    std::uniform_real_distribution<double> coef_dist(-1.0, 1.0);
    for (int i = 0; i < n_features; ++i) {
        coefficients[i] = coef_dist(rng);
    }
    X_train.resize(n_samples);
    y_train.resize(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        X_train[i].resize(n_features);
        for (int j = 0; j < n_features; ++j) {
            X_train[i][j] = coef_dist(rng);
        }
        y_train[i] = 0.0;
        for (int j = 0; j < n_features; ++j) {
            y_train[i] += X_train[i][j] * coefficients[j];
        }
        y_train[i] += noise_dist(rng);
    }
    X_test.resize(n_samples / 5);
    y_test.resize(n_samples / 5);
    for (int i = 0; i < n_samples / 5; ++i) {
        X_test[i].resize(n_features);
        for (int j = 0; j < n_features; ++j) {
            X_test[i][j] = coef_dist(rng);
        }
        y_test[i] = 0.0;
        for (int j = 0; j < n_features; ++j) {
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
    auto max_iter_param = std::make_shared<severine::IntegerHyperparameter>("max_iter", 100, 1000);
    searchSpace.addHyperparameter(max_iter_param);
    std::vector<std::string> solver_options = {"sgd", "normal_equation"};
    auto solver_param = std::make_shared<severine::CategoricalHyperparameter>("solver", solver_options);
    searchSpace.addHyperparameter(solver_param);
    return searchSpace;
}
severine::SearchSpace createRandomForestSearchSpace() {
    severine::SearchSpace searchSpace;
    auto n_estimators_param = std::make_shared<severine::IntegerHyperparameter>("n_estimators", 10, 200);
    searchSpace.addHyperparameter(n_estimators_param);
    auto max_depth_param = std::make_shared<severine::IntegerHyperparameter>("max_depth", 3, 20);
    searchSpace.addHyperparameter(max_depth_param);
    auto min_samples_split_param = std::make_shared<severine::IntegerHyperparameter>("min_samples_split", 2, 20);
    searchSpace.addHyperparameter(min_samples_split_param);
    auto max_features_param = std::make_shared<severine::IntegerHyperparameter>("max_features", 1, 10);
    searchSpace.addHyperparameter(max_features_param);
    std::vector<std::string> criterion_options = {"mse", "mae"};
    auto criterion_param = std::make_shared<severine::CategoricalHyperparameter>("criterion", criterion_options);
    searchSpace.addHyperparameter(criterion_param);
    return searchSpace;
}
severine::SearchSpace createSVMSearchSpace() {
    severine::SearchSpace searchSpace;
    auto C_param = std::make_shared<severine::FloatHyperparameter>(
        "C", 0.1f, 100.0f, severine::FloatHyperparameter::Distribution::LOG_UNIFORM);
    searchSpace.addHyperparameter(C_param);
    auto gamma_param = std::make_shared<severine::FloatHyperparameter>(
        "gamma", 0.001f, 1.0f, severine::FloatHyperparameter::Distribution::LOG_UNIFORM);
    searchSpace.addHyperparameter(gamma_param);
    auto epsilon_param = std::make_shared<severine::FloatHyperparameter>(
        "epsilon", 0.01f, 1.0f, severine::FloatHyperparameter::Distribution::LOG_UNIFORM);
    searchSpace.addHyperparameter(epsilon_param);
    std::vector<std::string> kernel_options = {"linear", "rbf"};
    auto kernel_param = std::make_shared<severine::CategoricalHyperparameter>("kernel", kernel_options);
    searchSpace.addHyperparameter(kernel_param);
    return searchSpace;
}
struct BenchmarkResult {
    std::vector<double> scores;
    std::vector<double> best_score_history;
    std::vector<double> times;
    double total_time;
    severine::Config best_config;
    double best_score;
};
BenchmarkResult runBenchmark(
    const std::string& strategy_name,
    std::shared_ptr<severine::ModelInterface> model,
    const severine::SearchSpace& searchSpace,
    int max_iterations,
    int n_threads = 1) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shared_ptr<severine::SearchStrategy> strategy;
    if (strategy_name == "random") {
        strategy = std::make_shared<severine::RandomSearch>(searchSpace, rng);
    } else if (strategy_name == "bayesian") {
        strategy = std::make_shared<severine::BayesianOptimization>(
            searchSpace, rng, 10, 2.0, severine::KernelType::RBF,
            severine::AcquisitionFunctionType::EI, true);
    } else {
        throw std::invalid_argument("Unknown strategy: " + strategy_name);
    }
    severine::TunerConfig tunerConfig;
    tunerConfig.maxIterations = max_iterations;
    tunerConfig.numThreads = n_threads;
    tunerConfig.verbose = false;
    tunerConfig.parallelStrategy = severine::ParallelizationStrategy::THREAD_POOL;
    severine::Tuner tuner(model, strategy, tunerConfig);
    auto start = std::chrono::high_resolution_clock::now();
    tuner.tune();
    auto end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end - start).count();
    const auto& results = tuner.getAllResults();
    const auto& best_result = tuner.getBestResult();
    BenchmarkResult benchmark;
    benchmark.scores.reserve(results.size());
    benchmark.best_score_history.reserve(results.size());
    benchmark.times.reserve(results.size());
    double current_best = -std::numeric_limits<double>::infinity();
    for (const auto& result : results) {
        benchmark.scores.push_back(result.score);
        current_best = std::max(current_best, result.score);
        benchmark.best_score_history.push_back(current_best);
    }
    benchmark.total_time = total_time;
    benchmark.best_config = best_result.configuration;
    benchmark.best_score = best_result.score;
    return benchmark;
}
void saveBenchmarkToCSV(
    const std::string& filename,
    const std::vector<BenchmarkResult>& results,
    const std::vector<std::string>& strategy_names,
    int iterations) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    file << "Iteration";
    for (const auto& name : strategy_names) {
        file << "," << name << "_score," << name << "_best_score";
    }
    file << std::endl;
    for (int i = 0; i < iterations; ++i) {
        file << i+1;
        for (size_t j = 0; j < strategy_names.size(); ++j) {
            if (i < results[j].scores.size()) {
                file << "," << results[j].scores[i] << "," << results[j].best_score_history[i];
            } else {
                file << ",,";
            }
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
    summary_file << "Strategy,Best Score,Total Time (s),Best Config" << std::endl;
    for (size_t i = 0; i < strategy_names.size(); ++i) {
        summary_file << strategy_names[i] << ","
                    << results[i].best_score << ","
                    << results[i].total_time << ",\"";
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
        summary_file << "\"" << std::endl;
    }
    summary_file.close();
    std::cout << "Results saved to:\n"
              << " - " << filename << "\n"
              << " - " << summary_filename << std::endl;
}
void printBenchmarkSummary(
    const std::vector<BenchmarkResult>& results,
    const std::vector<std::string>& strategy_names) {
    std::cout << "\n=== Benchmark Summary ===\n" << std::endl;
    std::cout << std::left << std::setw(15) << "Strategy"
              << std::setw(15) << "Best Score"
              << std::setw(15) << "Total Time (s)"
              << "Best Configuration" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    for (size_t i = 0; i < strategy_names.size(); ++i) {
        std::cout << std::left << std::setw(15) << strategy_names[i]
                  << std::setw(15) << std::fixed << std::setprecision(6) << results[i].best_score
                  << std::setw(15) << std::fixed << std::setprecision(2) << results[i].total_time;
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
                std::cout << std::get<float>(value);
            } else if (std::holds_alternative<bool>(value)) {
                std::cout << (std::get<bool>(value) ? "true" : "false");
            } else if (std::holds_alternative<std::string>(value)) {
                std::cout << std::get<std::string>(value);
            }
        }
        std::cout << " }" << std::endl;
    }
    std::cout << std::endl;
}
void benchmarkSingleModel(
    const std::string& model_type,
    const std::vector<std::vector<double>>& X_train,
    const std::vector<double>& y_train,
    const std::vector<std::vector<double>>& X_test,
    const std::vector<double>& y_test,
    int iterations,
    int n_threads,
    const std::string& output_dir) {
    std::cout << "\n=== Benchmarking " << model_type << " ===" << std::endl;
    std::cout << "Iterations: " << iterations << " | Threads: " << n_threads << std::endl;
    std::shared_ptr<severine::ModelInterface> model;
    severine::SearchSpace searchSpace;
    if (model_type == "LinearRegression") {
        model = severine::createMLFmkModel("LinearRegression");
        searchSpace = createLinearRegressionSearchSpace();
    } else if (model_type == "RandomForest") {
        model = severine::createMLFmkModel("RandomForest");
        searchSpace = createRandomForestSearchSpace();
    } else if (model_type == "SVM") {
        model = severine::createMLFmkModel("SVM");
        searchSpace = createSVMSearchSpace();
    } else {
        throw std::invalid_argument("Unsupported model type: " + model_type);
    }
    auto* dataModel = dynamic_cast<severine::DataMatrixModelAdapter*>(model.get());
    if (dataModel) {
        dataModel->loadTrainingData(X_train, y_train);
        dataModel->loadTestData(X_test, y_test);
    } else {
        throw std::runtime_error("Model does not support data loading");
    }
    std::vector<std::string> strategy_names = {"random", "bayesian"};
    std::vector<BenchmarkResult> results;
    for (const auto& strategy_name : strategy_names) {
        std::cout << "\nRunning benchmark for " << strategy_name << " search..." << std::endl;
        std::shared_ptr<severine::ModelInterface> model_clone = model->clone();
        BenchmarkResult result = runBenchmark(
            strategy_name, model_clone, searchSpace, iterations, n_threads);
        results.push_back(result);
    }
    printBenchmarkSummary(results, strategy_names);
    std::string output_file = output_dir + "/" + model_type + "_benchmark.csv";
    saveBenchmarkToCSV(output_file, results, strategy_names, iterations);
}
void runInteractiveBenchmark() {
    std::string output_dir = getOutputDirectory();
    std::cout << "Benchmark results will be saved to: " << output_dir << std::endl;
    int iterations = 50;
    int n_threads = 4;
    int n_samples = 1000;
    int n_features = 10;
    double noise = 0.1;
    std::cout << "=== Severine Benchmarking Tool ===\n\n";
    std::cout << "Please configure your benchmark:\n";
    std::cout << "Number of iterations (default: 50): ";
    std::string iterations_str;
    std::getline(std::cin, iterations_str);
    if (!iterations_str.empty()) {
        try {
            iterations = std::stoi(iterations_str);
        } catch (const std::exception& e) {
            std::cout << "Invalid input, using default: 50\n";
            iterations = 50;
        }
    }
    std::cout << "Number of threads (default: 4): ";
    std::string threads_str;
    std::getline(std::cin, threads_str);
    if (!threads_str.empty()) {
        try {
            n_threads = std::stoi(threads_str);
        } catch (const std::exception& e) {
            std::cout << "Invalid input, using default: 4\n";
            n_threads = 4;
        }
    }
    std::cout << "\nSelect models to benchmark:\n";
    std::cout << "1. Linear Regression\n";
    std::cout << "2. Random Forest\n";
    std::cout << "3. SVM\n";
    std::cout << "4. All models\n";
    std::cout << "Your choice (1-4): ";
    std::string model_choice;
    std::getline(std::cin, model_choice);
    if (model_choice.empty() ||
        (model_choice != "1" && model_choice != "2" &&
         model_choice != "3" && model_choice != "4")) {
        std::cout << "Invalid choice, defaulting to Linear Regression\n";
        model_choice = "1";
    }
    std::cout << "\nGenerating synthetic dataset...\n";
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;
    generateSyntheticData(X_train, y_train, X_test, y_test, n_samples, n_features, noise);
    std::cout << "Dataset generated: " << X_train.size() << " training samples, "
              << X_test.size() << " test samples, " << X_train[0].size() << " features\n\n";
    if (model_choice == "4") {
        std::vector<std::string> model_types = {"LinearRegression", "RandomForest", "SVM"};
        for (const auto& model_type : model_types) {
            benchmarkSingleModel(model_type, X_train, y_train, X_test, y_test,
                                iterations, n_threads, output_dir);
        }
        std::string combined_file = output_dir + "/combined_results.csv";
        std::ofstream combined(combined_file);
        if (combined.is_open()) {
            combined << "Model,Strategy,Best Score,Total Time (s)\n";
            for (const auto& model_type : model_types) {
                std::string summary_file = output_dir + "/" + model_type + "_benchmark_summary.csv";
                std::ifstream summary(summary_file);
                if (summary.is_open()) {
                    std::string line;
                    std::getline(summary, line);
                    while (std::getline(summary, line)) {
                        std::stringstream ss(line);
                        std::string strategy, score, time, config;
                        std::getline(ss, strategy, ',');
                        std::getline(ss, score, ',');
                        std::getline(ss, time, ',');
                        combined << model_type << "," << strategy << "," << score << "," << time << "\n";
                    }
                    summary.close();
                }
            }
            combined.close();
            std::cout << "\nCombined results saved to: " << combined_file << std::endl;
        }
    } else {
        std::string model_type;
        if (model_choice == "1") {
            model_type = "LinearRegression";
        } else if (model_choice == "2") {
            model_type = "RandomForest";
        } else {  // model_choice == "3"
            model_type = "SVM";
        }
        benchmarkSingleModel(model_type, X_train, y_train, X_test, y_test,
                           iterations, n_threads, output_dir);
    }
    std::cout << "\nAll benchmark results saved to: " << output_dir << std::endl;
}
void runCommandLineBenchmark(int argc, char** argv) {
    std::string model_type = "LinearRegression";
    int iterations = 50;
    int n_threads = 4;
    bool run_all_models = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" || arg == "-m") {
            if (i + 1 < argc) {
                model_type = argv[++i];
            }
        } else if (arg == "--iterations" || arg == "-i") {
            if (i + 1 < argc) {
                iterations = std::stoi(argv[++i]);
            }
        } else if (arg == "--threads" || arg == "-t") {
            if (i + 1 < argc) {
                n_threads = std::stoi(argv[++i]);
            }
        } else if (arg == "--all" || arg == "-a") {
            run_all_models = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]\n\n"
                      << "Options:\n"
                      << "  -m, --model MODEL        Model type (LinearRegression, RandomForest, SVM)\n"
                      << "  -i, --iterations N       Number of iterations (default: 50)\n"
                      << "  -t, --threads N          Number of threads (default: 4)\n"
                      << "  -a, --all                Run benchmark for all model types\n"
                      << "  -h, --help               Show this help message\n";
            return;
        }
    }
    std::string output_dir = getOutputDirectory();
    std::cout << "Benchmark results will be saved to: " << output_dir << std::endl;
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;
    generateSyntheticData(X_train, y_train, X_test, y_test, 1000, 10, 0.1);
    std::cout << "Generated synthetic dataset with " << X_train.size() << " training samples, "
              << X_test.size() << " test samples, " << X_train[0].size() << " features\n";
    if (run_all_models) {
        std::vector<std::string> model_types = {"LinearRegression", "RandomForest", "SVM"};
        for (const auto& model : model_types) {
            benchmarkSingleModel(model, X_train, y_train, X_test, y_test,
                              iterations, n_threads, output_dir);
        }
        std::string combined_file = output_dir + "/combined_results.csv";
        std::ofstream combined(combined_file);
        if (combined.is_open()) {
            combined << "Model,Strategy,Best Score,Total Time (s)\n";
            for (const auto& model : model_types) {
                std::string summary_file = output_dir + "/" + model + "_benchmark_summary.csv";
                std::ifstream summary(summary_file);
                if (summary.is_open()) {
                    std::string line;
                    std::getline(summary, line);
                    while (std::getline(summary, line)) {
                        std::stringstream ss(line);
                        std::string strategy, score, time, config;
                        std::getline(ss, strategy, ',');
                        std::getline(ss, score, ',');
                        std::getline(ss, time, ',');
                        combined << model << "," << strategy << "," << score << "," << time << "\n";
                    }
                    summary.close();
                }
            }
            combined.close();
            std::cout << "\nCombined results saved to: " << combined_file << std::endl;
        }
    } else {
        benchmarkSingleModel(model_type, X_train, y_train, X_test, y_test,
                           iterations, n_threads, output_dir);
    }
}
} // namespace severinebench
int testingToolMain(int argc, char** argv) {
    try {
        if (argc <= 1) {
            severinebench::runInteractiveBenchmark();
        } else if (argc > 1 && std::string(argv[1]) == "--interactive") {
            severinebench::runInteractiveBenchmark();
        } else {
            severinebench::runCommandLineBenchmark(argc, argv);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}