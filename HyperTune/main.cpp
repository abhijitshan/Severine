//
//  main_bayesian.cpp
//  HyperTune
//
//  Created on 18/04/25.
//

#include "include/hyperparameter.hpp"
#include "include/bayesianOptimization.hpp"
#include "include/tuner.hpp"
#include "include/modelAdapters.hpp"
#include "include/randomSearch.hpp"
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <chrono>

using namespace hypertune;

// Helper function to generate synthetic data
void generateSyntheticData(
    std::vector<std::vector<double>>& X_train,
    std::vector<double>& y_train,
    std::vector<std::vector<double>>& X_test,
    std::vector<double>& y_test,
    int n_samples = 1000,
    int n_features = 10,
    double noise = 0.1) {
    
    // Set up RNG
    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<double> noise_dist(0.0, noise);
    
    // Generate true coefficients
    std::vector<double> coefficients(n_features);
    std::uniform_real_distribution<double> coef_dist(-1.0, 1.0);
    for (int i = 0; i < n_features; ++i) {
        coefficients[i] = coef_dist(rng);
    }
    
    // Generate training data
    X_train.resize(n_samples);
    y_train.resize(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        X_train[i].resize(n_features);
        for (int j = 0; j < n_features; ++j) {
            X_train[i][j] = coef_dist(rng);
        }
        
        // Calculate y = X * coefficients + noise
        y_train[i] = 0.0;
        for (int j = 0; j < n_features; ++j) {
            y_train[i] += X_train[i][j] * coefficients[j];
        }
        y_train[i] += noise_dist(rng);
    }
    
    // Generate test data
    X_test.resize(n_samples / 5);
    y_test.resize(n_samples / 5);
    
    for (int i = 0; i < n_samples / 5; ++i) {
        X_test[i].resize(n_features);
        for (int j = 0; j < n_features; ++j) {
            X_test[i][j] = coef_dist(rng);
        }
        
        // Calculate y = X * coefficients + noise
        y_test[i] = 0.0;
        for (int j = 0; j < n_features; ++j) {
            y_test[i] += X_test[i][j] * coefficients[j];
        }
        y_test[i] += noise_dist(rng);
    }
}

// Example 1: Tune Linear Regression with Bayesian Optimization
void tuneLinearRegressionWithBayesianOpt() {
    std::cout << "=== Tuning Linear Regression with Bayesian Optimization ===" << std::endl;
    
    // Generate synthetic data
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;
    generateSyntheticData(X_train, y_train, X_test, y_test);
    
    // Create a search space
    std::cout << "Creating search space..." << std::endl;
    SearchSpace searchSpace;
    
    // Add hyperparameters for linear regression
    std::cout << "Adding hyperparameters..." << std::endl;
    searchSpace.addHyperparameter(
        std::make_shared<BooleanHyperparameter>("fit_intercept"));
    searchSpace.addHyperparameter(
        std::make_shared<FloatHyperparameter>("alpha", 0.0001f, 10.0f,
                                           FloatHyperparameter::Distribution::LOG_UNIFORM));
    searchSpace.addHyperparameter(
        std::make_shared<CategoricalHyperparameter>(
            "solver", std::vector<std::string>{"sgd", "normal_equation"}));
    searchSpace.addHyperparameter(
        std::make_shared<IntegerHyperparameter>("max_iter", 100, 2000));
    
    // Set up RNG
    std::cout << "Setting up RNG..." << std::endl;
    std::random_device rd;
    std::mt19937 rng(rd());
    
    // Create model
    std::cout << "Creating model..." << std::endl;
    auto model = std::make_shared<LinearRegressionAdapter>();
    model->setVerbose(true);
    model->loadTrainingData(X_train, y_train);
    model->loadTestData(X_test, y_test);
    
    // Create Bayesian Optimization search strategy
    std::cout << "Creating Bayesian Optimization strategy..." << std::endl;
    auto strategy = std::make_shared<BayesianOptimization>(searchSpace, rng, 5, 2.0);
    
    // Configure the tuner
    std::cout << "Configuring tuner..." << std::endl;
    TunerConfig tunerConfig;
    tunerConfig.maxIterations = 15;
    tunerConfig.numThreads = 4;
    tunerConfig.earlyStopStrategy = EarlyStoppingStrategy::NO_IMPROVEMENT;
    tunerConfig.earlyStoppingPatience = 3;
    
    // Create and run the tuner
    std::cout << "Creating tuner..." << std::endl;
    Tuner tuner(model, strategy, tunerConfig);
    
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Running Bayesian Optimization tuner..." << std::endl;
    tuner.tune();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Get the best result
    std::cout << "Getting best result..." << std::endl;
    const EvaluationResult& best = tuner.getBestResult();
    std::cout << "\nBest configuration found:" << std::endl;
    for (const auto& [name, value] : best.configuration) {
        std::cout << "  " << name << ": ";
        if (std::holds_alternative<int>(value)) {
            std::cout << std::get<int>(value);
        } else if (std::holds_alternative<float>(value)) {
            std::cout << std::get<float>(value);
        } else if (std::holds_alternative<bool>(value)) {
            std::cout << (std::get<bool>(value) ? "true" : "false");
        } else if (std::holds_alternative<std::string>(value)) {
            std::cout << std::get<std::string>(value);
        }
        std::cout << std::endl;
    }
    std::cout << "Score: " << best.score << std::endl;
    std::cout << "Optimization completed in " << duration << " ms" << std::endl;
    std::cout << std::endl;
}

// Example 2: Tune Random Forest with Bayesian Optimization
void tuneRandomForestWithBayesianOpt() {
    std::cout << "=== Tuning Random Forest with Bayesian Optimization ===" << std::endl;
    
    // Generate synthetic data
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;
    generateSyntheticData(X_train, y_train, X_test, y_test);
    
    // Create a search space
    std::cout << "Creating search space..." << std::endl;
    SearchSpace searchSpace;
    
    // Add hyperparameters for random forest
    std::cout << "Adding hyperparameters..." << std::endl;
    searchSpace.addHyperparameter(
        std::make_shared<IntegerHyperparameter>("n_estimators", 10, 200));
    searchSpace.addHyperparameter(
        std::make_shared<IntegerHyperparameter>("max_depth", 3, 20));
    searchSpace.addHyperparameter(
        std::make_shared<IntegerHyperparameter>("min_samples_split", 2, 20));
    searchSpace.addHyperparameter(
        std::make_shared<IntegerHyperparameter>("max_features", 0, 10));
    
    // Set up RNG
    std::cout << "Setting up RNG..." << std::endl;
    std::random_device rd;
    std::mt19937 rng(rd());
    
    // Create model
    std::cout << "Creating model..." << std::endl;
    auto model = std::make_shared<RandomForestAdapter>();
    model->setVerbose(true);
    model->loadTrainingData(X_train, y_train);
    model->loadTestData(X_test, y_test);
    
    // Create Bayesian Optimization search strategy
    std::cout << "Creating Bayesian Optimization strategy..." << std::endl;
    auto strategy = std::make_shared<BayesianOptimization>(searchSpace, rng, 8, 1.5);
    
    // Configure the tuner
    std::cout << "Configuring tuner..." << std::endl;
    TunerConfig tunerConfig;
    tunerConfig.maxIterations = 20;
    tunerConfig.numThreads = 4;
    tunerConfig.earlyStopStrategy = EarlyStoppingStrategy::NO_IMPROVEMENT;
    tunerConfig.earlyStoppingPatience = 5;
    
    // Create and run the tuner
    std::cout << "Creating tuner..." << std::endl;
    Tuner tuner(model, strategy, tunerConfig);
    
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Running Bayesian Optimization tuner..." << std::endl;
    tuner.tune();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Get the best result
    std::cout << "Getting best result..." << std::endl;
    const EvaluationResult& best = tuner.getBestResult();
    std::cout << "\nBest configuration found:" << std::endl;
    for (const auto& [name, value] : best.configuration) {
        std::cout << "  " << name << ": ";
        if (std::holds_alternative<int>(value)) {
            std::cout << std::get<int>(value);
        } else if (std::holds_alternative<float>(value)) {
            std::cout << std::get<float>(value);
        } else if (std::holds_alternative<bool>(value)) {
            std::cout << (std::get<bool>(value) ? "true" : "false");
        } else if (std::holds_alternative<std::string>(value)) {
            std::cout << std::get<std::string>(value);
        }
        std::cout << std::endl;
    }
    std::cout << "Score: " << best.score << std::endl;
    std::cout << "Optimization completed in " << duration << " ms" << std::endl;
}

// Compare Bayesian Optimization with Random Search
void compareBayesianVsRandom() {
    std::cout << "\n=== Comparing Bayesian Optimization vs Random Search ===" << std::endl;
    
    // Generate synthetic data
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;
    generateSyntheticData(X_train, y_train, X_test, y_test, 1500, 12, 0.15);
    
    // Create a search space
    SearchSpace searchSpace;
    
    // Add hyperparameters (same for both methods)
    searchSpace.addHyperparameter(
        std::make_shared<FloatHyperparameter>("C", 0.01f, 100.0f,
                                           FloatHyperparameter::Distribution::LOG_UNIFORM));
    searchSpace.addHyperparameter(
        std::make_shared<FloatHyperparameter>("gamma", 0.001f, 10.0f,
                                           FloatHyperparameter::Distribution::LOG_UNIFORM));
    searchSpace.addHyperparameter(
        std::make_shared<FloatHyperparameter>("epsilon", 0.01f, 1.0f,
                                           FloatHyperparameter::Distribution::UNIFORM));
    searchSpace.addHyperparameter(
        std::make_shared<CategoricalHyperparameter>(
            "kernel", std::vector<std::string>{"linear", "rbf"}));
    
    // Set up RNG with fixed seed for fair comparison
    unsigned seed = 12345;
    std::mt19937 rng(seed);
    
    // Configure tuner settings (same for both methods)
    TunerConfig tunerConfig;
    tunerConfig.maxIterations = 25;
    tunerConfig.numThreads = 4;
    tunerConfig.earlyStopStrategy = EarlyStoppingStrategy::NONE; // Disable early stopping for fair comparison
    
    // First, run Random Search
    std::cout << "\nRunning Random Search..." << std::endl;
    
    auto modelRandom = std::make_shared<SVMAdapter>();
    modelRandom->setVerbose(false); // Reduce output verbosity
    modelRandom->loadTrainingData(X_train, y_train);
    modelRandom->loadTestData(X_test, y_test);
    
    auto randomStrategy = std::make_shared<RandomSearch>(searchSpace, rng);
    Tuner randomTuner(modelRandom, randomStrategy, tunerConfig);
    
    auto randomStart = std::chrono::high_resolution_clock::now();
    randomTuner.tune();
    auto randomEnd = std::chrono::high_resolution_clock::now();
    auto randomDuration = std::chrono::duration_cast<std::chrono::milliseconds>(randomEnd - randomStart).count();
    
    const EvaluationResult& randomBest = randomTuner.getBestResult();
    
    // Next, run Bayesian Optimization
    std::cout << "\nRunning Bayesian Optimization..." << std::endl;
    
    // Reset RNG with same seed for fair comparison
    rng.seed(seed);
    
    auto modelBayes = std::make_shared<SVMAdapter>();
    modelBayes->setVerbose(false); // Reduce output verbosity
    modelBayes->loadTrainingData(X_train, y_train);
    modelBayes->loadTestData(X_test, y_test);
    
    auto bayesStrategy = std::make_shared<BayesianOptimization>(searchSpace, rng, 5, 2.0);
    Tuner bayesTuner(modelBayes, bayesStrategy, tunerConfig);
    
    auto bayesStart = std::chrono::high_resolution_clock::now();
    bayesTuner.tune();
    auto bayesEnd = std::chrono::high_resolution_clock::now();
    auto bayesDuration = std::chrono::duration_cast<std::chrono::milliseconds>(bayesEnd - bayesStart).count();
    
    const EvaluationResult& bayesBest = bayesTuner.getBestResult();
    
    // Print comparison results
    std::cout << "\n=== Optimization Results Comparison ===" << std::endl;
    
    std::cout << "\nRandom Search Results:" << std::endl;
    std::cout << "  Best Score: " << randomBest.score << std::endl;
    std::cout << "  Execution Time: " << randomDuration << " ms" << std::endl;
    std::cout << "  Best Configuration:" << std::endl;
    for (const auto& [name, value] : randomBest.configuration) {
        std::cout << "    " << name << ": ";
        if (std::holds_alternative<int>(value)) {
            std::cout << std::get<int>(value);
        } else if (std::holds_alternative<float>(value)) {
            std::cout << std::get<float>(value);
        } else if (std::holds_alternative<bool>(value)) {
            std::cout << (std::get<bool>(value) ? "true" : "false");
        } else if (std::holds_alternative<std::string>(value)) {
            std::cout << std::get<std::string>(value);
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nBayesian Optimization Results:" << std::endl;
    std::cout << "  Best Score: " << bayesBest.score << std::endl;
    std::cout << "  Execution Time: " << bayesDuration << " ms" << std::endl;
    std::cout << "  Best Configuration:" << std::endl;
    for (const auto& [name, value] : bayesBest.configuration) {
        std::cout << "    " << name << ": ";
        if (std::holds_alternative<int>(value)) {
            std::cout << std::get<int>(value);
        } else if (std::holds_alternative<float>(value)) {
            std::cout << std::get<float>(value);
        } else if (std::holds_alternative<bool>(value)) {
            std::cout << (std::get<bool>(value) ? "true" : "false");
        } else if (std::holds_alternative<std::string>(value)) {
            std::cout << std::get<std::string>(value);
        }
        std::cout << std::endl;
    }
    
    // Performance comparison
    double scoreDiff = bayesBest.score - randomBest.score;
    double timeDiff = static_cast<double>(bayesDuration - randomDuration) / randomDuration * 100.0;
    
    std::cout << "\nPerformance Difference:" << std::endl;
    std::cout << "  Score Improvement: " << (scoreDiff >= 0 ? "+" : "") << scoreDiff
              << " (" << (scoreDiff >= 0 ? "+" : "") << (scoreDiff / std::abs(randomBest.score) * 100.0)
              << "%)" << std::endl;
    std::cout << "  Time Difference: " << (timeDiff >= 0 ? "+" : "") << timeDiff << "%" << std::endl;
}

int main() {
    std::cout << "=== Starting HyperTune Bayesian Optimization Examples ===" << std::endl;
    
    // Tune Linear Regression with Bayesian Optimization
    tuneLinearRegressionWithBayesianOpt();
    
    // Tune Random Forest with Bayesian Optimization
    tuneRandomForestWithBayesianOpt();
    
    // Compare Bayesian Optimization with Random Search
    compareBayesianVsRandom();
    
    std::cout << "\n=== HyperTune Bayesian Optimization Examples completed ===" << std::endl;
    return 0;
}
