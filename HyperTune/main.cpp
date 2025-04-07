//
//  exampleUsage.cpp
//  HyperTune
//
//  Created on 07/04/25.
//

#include "include/hyperparameter.hpp"
#include "include/randomSearch.hpp"
#include "include/tuner.hpp"
#include "include/modelAdapters.hpp"
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

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

// Example usage for linear regression
void tuneLinearRegression() {
    std::cout << "=== Tuning Linear Regression ===" << std::endl;
    
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
    
    // Create random search strategy
    std::cout << "Creating search strategy..." << std::endl;
    auto strategy = std::make_shared<RandomSearch>(searchSpace, rng);
    
    // Configure the tuner
    std::cout << "Configuring tuner..." << std::endl;
    TunerConfig tunerConfig;
    tunerConfig.maxIterations = 10;
    tunerConfig.numThreads = 4;
    tunerConfig.earlyStopStrategy = EarlyStoppingStrategy::NO_IMPROVEMENT;
    tunerConfig.earlyStoppingPatience = 3;
    
    // Create and run the tuner
    std::cout << "Creating tuner..." << std::endl;
    Tuner tuner(model, strategy, tunerConfig);
    
    std::cout << "Running tuner..." << std::endl;
    tuner.tune();
    
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
}

// Example usage for random forest
void tuneRandomForest() {
    std::cout << "\n=== Tuning Random Forest ===" << std::endl;
    
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
        std::make_shared<IntegerHyperparameter>("max_features", -1, 10));
    
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
    
    // Create random search strategy
    std::cout << "Creating search strategy..." << std::endl;
    auto strategy = std::make_shared<RandomSearch>(searchSpace, rng);
    
    // Configure the tuner
    std::cout << "Configuring tuner..." << std::endl;
    TunerConfig tunerConfig;
    tunerConfig.maxIterations = 10;
    tunerConfig.numThreads = 4;
    tunerConfig.earlyStopStrategy = EarlyStoppingStrategy::NO_IMPROVEMENT;
    tunerConfig.earlyStoppingPatience = 3;
    
    // Create and run the tuner
    std::cout << "Creating tuner..." << std::endl;
    Tuner tuner(model, strategy, tunerConfig);
    
    std::cout << "Running tuner..." << std::endl;
    tuner.tune();
    
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
}

int main() {
    std::cout << "=== Starting HyperTune Model Examples ===" << std::endl;
    
    // Tune Linear Regression
    tuneLinearRegression();
    
    // Tune Random Forest
    tuneRandomForest();
    
    std::cout << "=== HyperTune examples completed ===" << std::endl;
    return 0;
}
