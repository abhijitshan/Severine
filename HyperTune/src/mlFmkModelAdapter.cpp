//
//  mlFmkModelAdapter.cpp
//  HyperTune
//
//  Created by Abhijit Shanbhag on 07/04/25.
//
#include "../include/modelInterface.hpp"
#include "../include/modelAdapters.hpp"
#include <stdexcept>
#include <iostream>
#include <chrono>

namespace hypertune {

/// Implementing the templates defined in `modelInterface.hpp`
void MLFmkModelAdapter::train() {
    auto start = std::chrono::high_resolution_clock::now();
    trainModel();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    if (verbose_) {
        std::cout << "Training Completed in " << duration.count() << " ms\n";
    }
}

double MLFmkModelAdapter::evaluate() {
    auto start = std::chrono::high_resolution_clock::now();
    double score = evaluateModel();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    if (verbose_) {
        std::cout << "Evaluation Completed in " << duration.count() << " ms\n";
        std::cout << "Score Obtained: \t\t" << score << "\n";
    }
    return score;
}

void MLFmkModelAdapter::configure(const Config& hyperparameters) {
    config_ = hyperparameters;
    applyHyperparameters(config_);
    if (verbose_) {
        std::cout << "Model Configured with Hyperparameters:\n";
        for (const auto& [name, value] : hyperparameters) {
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
    }
}

std::string MLFmkModelAdapter::toString() const {
    /// Generic implementation, can be overridden by subclasses
    std::string result = "MLFmkModelAdapter with configuration:\n";
    for (const auto& [name, value] : config_) {
        result += "  " + name + ": ";
        if (std::holds_alternative<int>(value)) {
            result += std::to_string(std::get<int>(value));
        } else if (std::holds_alternative<float>(value)) {
            result += std::to_string(std::get<float>(value));
        } else if (std::holds_alternative<bool>(value)) {
            result += std::get<bool>(value) ? "true" : "false";
        } else if (std::holds_alternative<std::string>(value)) {
            result += std::get<std::string>(value);
        }
        result += "\n";
    }
    return result;
}

/// Factory function implementation
std::unique_ptr<ModelInterface> createMLFmkModel(const std::string& modelType) {
    /// This will be expanded as more model types are added
    if (modelType == "LinearRegression") {
        return std::unique_ptr<ModelInterface>(new LinearRegressionAdapter());
    } else if (modelType == "RandomForest") {
        return std::unique_ptr<ModelInterface>(new RandomForestAdapter());
    } else if (modelType == "SVM") {
        return std::unique_ptr<ModelInterface>(new SVMAdapter());
    } else if (modelType == "NeuralNetwork") {
        return std::unique_ptr<ModelInterface>(new NeuralNetworkAdapter());
    } else {
        throw std::invalid_argument("Unknown model type: " + modelType);
    }
}

} // namespace hypertune
