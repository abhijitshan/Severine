#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <memory>
#include <vector>
#include <string>
#include "include/hyperparameter.hpp"
#include "include/modelAdapters.hpp"
#include "include/randomSearch.hpp"
#include "include/bayesianOptimization.hpp"
#include "include/tuner.hpp"
void generateSyntheticData(
    std::vector<std::vector<double>>& X_train,
    std::vector<double>& y_train,
    std::vector<std::vector<double>>& X_test,
    std::vector<double>& y_test,
    int n_samples = 1000,
    int n_features = 20,
    double noise = 0.1)
{
    std::mt19937 rng(42);
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
    std::vector<std::string> solvers = {"sgd", "lsqr", "normal_equation"};
    auto solver_param = std::make_shared<severine::CategoricalHyperparameter>("solver", solvers);
    searchSpace.addHyperparameter(solver_param);
    auto max_iter_param = std::make_shared<severine::IntegerHyperparameter>("max_iter", 100, 5000);
    searchSpace.addHyperparameter(max_iter_param);
    auto tol_param = std::make_shared<severine::FloatHyperparameter>(
        "tol", 1e-6f, 1e-2f, severine::FloatHyperparameter::Distribution::LOG_UNIFORM);
    searchSpace.addHyperparameter(tol_param);
    return searchSpace;
}
class ProgressTracker {
public:
    ProgressTracker(std::shared_ptr<severine::SearchStrategy> baseStrategy)
        : baseStrategy_(baseStrategy),
          startTime_(std::chrono::high_resolution_clock::now())
    {
        std::cout << "\n=== Severine Hyperparameter Optimization Demo ===\n" << std::endl;
        std::cout << std::left
                  << std::setw(5) << "Iter"
                  << std::setw(12) << "Score"
                  << std::setw(15) << "Alpha"
                  << std::setw(14) << "FitIntercept"
                  << std::setw(10) << "MaxIter"
                  << std::setw(15) << "Solver"
                  << std::setw(15) << "Tolerance"
                  << std::setw(7) << "Time(s)"
                  << std::setw(15) << "Improvement"
                  << std::endl;
        std::cout << std::string(100, '-') << std::endl;
    }
    severine::Config nextConfiguration() {
        return baseStrategy_->nextConfiguration();
    }
    void update(const severine::EvaluationResult& result) {
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - startTime_).count();
        const auto& config = result.configuration;
        float alpha = 0.0f;
        bool fit_intercept = true;
        int max_iter = 1000;
        float tol = 1e-4f;
        std::string solver = "sgd";
        if (config.find("alpha") != config.end() && std::holds_alternative<float>(config.at("alpha"))) {
            alpha = std::get<float>(config.at("alpha"));
        }
        if (config.find("fit_intercept") != config.end() && std::holds_alternative<bool>(config.at("fit_intercept"))) {
            fit_intercept = std::get<bool>(config.at("fit_intercept"));
        }
        if (config.find("max_iter") != config.end() && std::holds_alternative<int>(config.at("max_iter"))) {
            max_iter = std::get<int>(config.at("max_iter"));
        }
        if (config.find("tol") != config.end() && std::holds_alternative<float>(config.at("tol"))) {
            tol = std::get<float>(config.at("tol"));
        }
        if (config.find("solver") != config.end() && std::holds_alternative<std::string>(config.at("solver"))) {
            solver = std::get<std::string>(config.at("solver"));
        }
        std::string improvement = "";
        if (result.score > bestScore_) {
            improvement = " ← NEW BEST!";
            bestScore_ = result.score;
            bestConfig_ = result.configuration;
            bestIteration_ = result.iteration + 1;
        }
        std::cout << std::left
                  << std::setw(5) << (result.iteration + 1)
                  << std::fixed << std::setprecision(8) << std::setw(12) << result.score
                  << std::scientific << std::setprecision(4) << std::setw(15) << alpha
                  << std::setw(14) << (fit_intercept ? "true" : "false")
                  << std::setw(10) << max_iter
                  << std::setw(15) << solver
                  << std::scientific << std::setprecision(4) << std::setw(15) << tol
                  << std::fixed << std::setprecision(2) << std::setw(7) << elapsed
                  << improvement
                  << std::endl;
        baseStrategy_->update(result);
    }
    void printSummary() const {
        auto now = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double>(now - startTime_).count();
        std::cout << "\n=== Hyperparameter Tuning Complete ===\n" << std::endl;
        std::cout << "Total time: " << std::fixed << std::setprecision(2) << totalTime << " seconds" << std::endl;
        std::cout << "Best score: " << std::fixed << std::setprecision(10) << bestScore_
                  << " (found at iteration " << bestIteration_ << ")" << std::endl;
        std::cout << "\nBest configuration:" << std::endl;
        for (const auto& [name, value] : bestConfig_) {
            std::cout << "  " << name << " = ";
            if (std::holds_alternative<int>(value)) {
                std::cout << std::get<int>(value);
            } else if (std::holds_alternative<float>(value)) {
                std::cout << std::scientific << std::setprecision(6) << std::get<float>(value);
            } else if (std::holds_alternative<bool>(value)) {
                std::cout << (std::get<bool>(value) ? "true" : "false");
            } else if (std::holds_alternative<std::string>(value)) {
                std::cout << std::get<std::string>(value);
            }
            std::cout << std::endl;
        }
    }
    const severine::EvaluationResult& getBestResult() const {
        return baseStrategy_->getBestResult();
    }
    std::shared_ptr<severine::SearchStrategy> getBaseStrategy() {
        return baseStrategy_;
    }
private:
    std::shared_ptr<severine::SearchStrategy> baseStrategy_;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime_;
    double bestScore_ = -std::numeric_limits<double>::infinity();
    severine::Config bestConfig_;
    int bestIteration_ = 0;
};
int main() {
    int iterations = 20;
    std::cout << "=== Severine Hyperparameter Optimization Demo ===" << std::endl;
    std::cout << "Algorithm: Bayesian Optimization | Model: Linear Regression" << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;
    std::cout << "\nGenerating synthetic dataset..." << std::endl;
    generateSyntheticData(X_train, y_train, X_test, y_test, 1000, 20, 0.1);
    std::cout << "Dataset generated: " << X_train.size() << " training samples, "
              << X_test.size() << " test samples, " << X_train[0].size() << " features" << std::endl;
    std::shared_ptr<severine::ModelInterface> model = severine::createMLFmkModel("LinearRegression");
    severine::SearchSpace searchSpace = createLinearRegressionSearchSpace();
    auto* dataModel = dynamic_cast<severine::DataMatrixModelAdapter*>(model.get());
    if (dataModel) {
        dataModel->loadTrainingData(X_train, y_train);
        dataModel->loadTestData(X_test, y_test);
        dataModel->setVerbose(false);
    } else {
        std::cerr << "Error: Model does not support data loading" << std::endl;
        return 1;
    }
    std::random_device rd;
    std::mt19937 rng(rd());
    std::cout << "\nInitializing Bayesian Optimization with MATERN kernel and UCB acquisition function..." << std::endl;
    auto baseStrategy = std::make_shared<severine::BayesianOptimization>(
        searchSpace, rng, 5, 2.0, severine::KernelType::MATERN,
        severine::AcquisitionFunctionType::UCB, true);
    auto progress = std::make_shared<ProgressTracker>(baseStrategy);
    std::cout << "\nStarting hyperparameter tuning process...\n" << std::endl;
    severine::TunerConfig tunerConfig;
    tunerConfig.maxIterations = iterations;
    tunerConfig.numThreads = 1;
    tunerConfig.verbose = false;
    tunerConfig.parallelStrategy = severine::ParallelizationStrategy::NONE;
    severine::Tuner tuner(model, progress->getBaseStrategy(), tunerConfig);
    for (int i = 0; i < iterations; i++) {
        severine::Config config = progress->nextConfiguration();
        model->configure(config);
        model->train();
        double score = model->evaluate();
        severine::EvaluationResult result{config, score, i};
        progress->update(result);
    }
    progress->printSummary();
    std::cout << "\n=== Demonstration Complete ===" << std::endl;
    return 0;
}