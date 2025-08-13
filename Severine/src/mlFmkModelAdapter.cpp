#include "../include/modelInterface.hpp"
#include "../include/modelAdapters.hpp"
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <numeric>
#include <cmath>
namespace severine {
void MLFmkModelAdapter::train() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto start = std::chrono::high_resolution_clock::now();
    trainModel();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    if (verbose_) {
        std::cout << "Training Completed in " << duration.count() << " ms\n";
    }
}
double MLFmkModelAdapter::evaluate() {
    std::lock_guard<std::mutex> lock(mutex_);
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
    std::lock_guard<std::mutex> lock(mutex_);
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
    std::lock_guard<std::mutex> lock(mutex_);
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
std::shared_ptr<ModelInterface> MLFmkModelAdapter::clone() const {
    return cloneImpl();
}
std::unique_ptr<ModelInterface> createMLFmkModel(const std::string& modelType) {
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
DataMatrixModelAdapter::DataMatrixModelAdapter() : dataLoaded_(false) {}
void DataMatrixModelAdapter::loadTrainingData(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    std::lock_guard<std::mutex> lock(mutex_); // Lock acquired here
    if (X.empty() || y.empty() || X.size() != y.size()) {
        throw std::invalid_argument("Invalid training data dimensions");
    }
    xTrain_ = X;
    yTrain_ = y;
    dataLoaded_ = true;
    if (verbose_) {
        std::cout << "Loaded training data: " << X.size() << " samples, "
                  << (X.empty() ? 0 : X[0].size()) << " features" << std::endl;
    }
} // Lock released here
void DataMatrixModelAdapter::loadTestData(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    std::lock_guard<std::mutex> lock(mutex_); // Lock acquired here
    if (X.empty() || y.empty() || X.size() != y.size()) {
        throw std::invalid_argument("Invalid test data dimensions");
    }
    xTest_ = X;
    yTest_ = y;
    yPred_.clear();
    if (verbose_) {
        std::cout << "Loaded test data: " << X.size() << " samples, "
                  << (X.empty() ? 0 : X[0].size()) << " features" << std::endl;
    }
} // Lock released here
double DataMatrixModelAdapter::calculateMSE() const {
    if (!dataLoaded_) {
        throw std::runtime_error("Data not loaded for MSE calculation.");
    }
    if (yPred_.empty()) {
         throw std::runtime_error("Predictions not available for MSE calculation. Call evaluate() first.");
    }
     if (yPred_.size() != yTest_.size()) {
         throw std::runtime_error("Prediction vector size does not match test data size for MSE.");
     }
    double sumSquaredErrors = 0.0;
    for (size_t i = 0; i < yTest_.size(); ++i) {
        double error = yTest_[i] - yPred_[i];
        sumSquaredErrors += error * error;
    }
    return sumSquaredErrors / yTest_.size();
}
double DataMatrixModelAdapter::calculateRMSE() const {
    if (!dataLoaded_) {
         throw std::runtime_error("Data not loaded for RMSE calculation.");
    }
     if (yPred_.empty()) {
         throw std::runtime_error("Predictions not available for RMSE calculation. Call evaluate() first.");
     }
    double mse = calculateMSE(); // This is safe now
    return std::sqrt(mse);
}
double DataMatrixModelAdapter::calculateMAE() const {
     if (!dataLoaded_) {
        throw std::runtime_error("Data not loaded for MAE calculation.");
    }
    if (yPred_.empty()) {
         throw std::runtime_error("Predictions not available for MAE calculation. Call evaluate() first.");
    }
     if (yPred_.size() != yTest_.size()) {
         throw std::runtime_error("Prediction vector size does not match test data size for MAE.");
     }
    double sumAbsoluteErrors = 0.0;
    for (size_t i = 0; i < yTest_.size(); ++i) {
        double error = std::abs(yTest_[i] - yPred_[i]);
        sumAbsoluteErrors += error;
    }
    return sumAbsoluteErrors / yTest_.size();
}
double DataMatrixModelAdapter::calculateR2() const {
    if (!dataLoaded_) {
        throw std::runtime_error("Data not loaded for R2 calculation.");
    }
    if (yPred_.empty()) {
        throw std::runtime_error("Predictions not available for R2 calculation. Call evaluate() first.");
    }
     if (yPred_.size() != yTest_.size()) {
         throw std::runtime_error("Prediction vector size does not match test data size for R2.");
     }
     if (yTest_.empty()){
         throw std::runtime_error("Cannot calculate R2 with empty test data.");
     }
    double meanY = std::accumulate(yTest_.begin(), yTest_.end(), 0.0) / yTest_.size();
    double ssTot = 0.0; // Total sum of squares
    double ssRes = 0.0;
    for (size_t i = 0; i < yTest_.size(); ++i) {
        double diffTotal = yTest_[i] - meanY;
        ssTot += diffTotal * diffTotal;
        double diffRes = yTest_[i] - yPred_[i];
        ssRes += diffRes * diffRes;
    }
    if (ssTot == 0.0) {
        return (ssRes == 0.0) ? 1.0 : 0.0;
    }
    return 1.0 - (ssRes / ssTot); // Standard R2 formula
}
void DataMatrixModelAdapter::applyHyperparameters(const Config& hyperparameters) {
    config_ = hyperparameters;
}
void DataMatrixModelAdapter::trainModel() {
    if (!dataLoaded_)
        throw std::runtime_error("Training data not loaded.");
    yPred_.clear();
}
double DataMatrixModelAdapter::evaluateModel() {
     if (!dataLoaded_)
        throw std::runtime_error("Test data not loaded.");
    if (yPred_.size() != xTest_.size()){
         yPred_.resize(xTest_.size());
    }
    if (yPred_.empty() && !xTest_.empty()){
         double meanY = std::accumulate(yTrain_.begin(), yTrain_.end(), 0.0) / yTrain_.size();
         std::fill(yPred_.begin(), yPred_.end(), meanY);
    }
    if (yTest_.empty()) return 0.0;
    return calculateR2();
}
std::shared_ptr<MLFmkModelAdapter> DataMatrixModelAdapter::cloneDataMatrixImpl() const{
    throw std::runtime_error("cloneDataMatrixImpl() must be implemented in derived DataMatrixModelAdapter classes.");
}
} // namespace severine