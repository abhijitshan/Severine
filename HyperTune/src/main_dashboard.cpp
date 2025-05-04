//
//  main_dashboard.cpp
//  HyperTune
//
//  Created on 18/04/25.
//

#include "../include/hyperparameter.hpp"
#include "../include/bayesianOptimization.hpp"
#include "../include/tuner.hpp"
#include "../include/modelAdapters.hpp"
#include "../include/randomSearch.hpp"
#include "../include/tunerServer.hpp" // This file is in the same directory
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#include <thread>

using namespace hypertune;

// Helper function to generate synthetic data (same as in your original code)
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

// Modified comparison function that uses the dashboard
void compareBayesianVsRandomWithDashboard(TunerServer& server) {
    std::cout << "\n=== Comparing Bayesian Optimization vs Random Search with Dashboard ===" << std::endl;
    
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
    
    // First, run Bayesian Optimization
    std::cout << "\nRunning Bayesian Optimization..." << std::endl;
    
    auto modelBayes = std::make_shared<SVMAdapter>();
    modelBayes->setVerbose(true); 
    modelBayes->loadTrainingData(X_train, y_train);
    modelBayes->loadTestData(X_test, y_test);
    
    auto bayesStrategy = std::make_shared<BayesianOptimization>(searchSpace, rng, 5, 2.0);
    auto bayesTuner = std::make_shared<Tuner>(modelBayes, bayesStrategy, tunerConfig);
    
    // Register the tuner with our dashboard server
    server.registerTuner(bayesTuner);
    
    auto bayesStart = std::chrono::high_resolution_clock::now();
    bayesTuner->tune();
    auto bayesEnd = std::chrono::high_resolution_clock::now();
    auto bayesDuration = std::chrono::duration_cast<std::chrono::milliseconds>(bayesEnd - bayesStart).count();
    
    const EvaluationResult& bayesBest = bayesTuner->getBestResult();
    
    // Print results
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
    
    std::cout << "\nDashboard is running. Press Enter to stop..." << std::endl;
    std::cin.get();
}

int main() {
    std::cout << "=== Starting HyperTune Dashboard Example ===" << std::endl;
    
    // Create and start the server
    TunerServer server(8080);
    server.start();
    
    std::cout << "Dashboard server started at http://localhost:8080" << std::endl;
    
    // Run the comparison with dashboard
    compareBayesianVsRandomWithDashboard(server);
    
    // Stop the server
    server.stop();
    
    std::cout << "\n=== HyperTune Dashboard Example completed ===" << std::endl;
    return 0;
}