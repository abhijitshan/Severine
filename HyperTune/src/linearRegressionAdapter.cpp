//
//  linearRegressionAdapter.cpp
//  HyperTune
//
//  Created by Abhijit Shanbhag on 07/04/25.
//

#include "../include/modelAdapters.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <random>

namespace hypertune {

/// DataMatrixModelAdapter implementation
DataMatrixModelAdapter::DataMatrixModelAdapter() : dataLoaded_(false) {}

void DataMatrixModelAdapter::loadTrainingData(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    if (X.empty() || y.empty() || X.size() != y.size()) {
        throw std::invalid_argument("Invalid training data dimensions");
    }
    
    xTrain_ = X;
    yTrain_ = y;
    dataLoaded_ = true;
    
    if (verbose_) {
        std::cout << "Loaded training data: " << X.size() << " samples, "
                  << X[0].size() << " features" << std::endl;
    }
}

void DataMatrixModelAdapter::loadTestData(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    if (X.empty() || y.empty() || X.size() != y.size()) {
        throw std::invalid_argument("Invalid test data dimensions");
    }
    
    xTest_ = X;
    yTest_ = y;
    
    if (verbose_) {
        std::cout << "Loaded test data: " << X.size() << " samples, "
                  << X[0].size() << " features" << std::endl;
    }
}

double DataMatrixModelAdapter::calculateMSE() const {
    if (yTest_.size() != yPred_.size() || yTest_.empty()) {
        throw std::runtime_error("Cannot calculate MSE: test data or predictions missing");
    }
    
    double sum_sq_error = 0.0;
    for (size_t i = 0; i < yTest_.size(); ++i) {
        double error = yTest_[i] - yPred_[i];
        sum_sq_error += error * error;
    }
    
    return sum_sq_error / yTest_.size();
}

double DataMatrixModelAdapter::calculateRMSE() const {
    return std::sqrt(calculateMSE());
}

double DataMatrixModelAdapter::calculateMAE() const {
    if (yTest_.size() != yPred_.size() || yTest_.empty()) {
        throw std::runtime_error("Cannot calculate MAE: test data or predictions missing");
    }
    
    double sum_abs_error = 0.0;
    for (size_t i = 0; i < yTest_.size(); ++i) {
        sum_abs_error += std::abs(yTest_[i] - yPred_[i]);
    }
    
    return sum_abs_error / yTest_.size();
}

double DataMatrixModelAdapter::calculateR2() const {
    if (yTest_.size() != yPred_.size() || yTest_.empty()) {
        throw std::runtime_error("Cannot calculate R2: test data or predictions missing");
    }
    
    /// Calculate mean of y_test
    double y_mean = std::accumulate(yTest_.begin(), yTest_.end(), 0.0) / yTest_.size();
    
    /// Calculate total sum of squares
    double ss_total = 0.0;
    for (const auto& y : yTest_) {
        double diff = y - y_mean;
        ss_total += diff * diff;
    }
    
    /// Calculate residual sum of squares
    double ss_residual = 0.0;
    for (size_t i = 0; i < yTest_.size(); ++i) {
        double diff = yTest_[i] - yPred_[i];
        ss_residual += diff * diff;
    }
    
    /// Calculate R^2
    if (ss_total < 1e-10) { /// Avoid division by zero
        return 0.0;
    }
    
    return 1.0 - (ss_residual / ss_total);
}

/// Empty implementations for abstract methods in DataMatrixModelAdapter
void DataMatrixModelAdapter::applyHyperparameters(const Config& hyperparameters) {
    /// Base implementation does nothing - to be overridden by derived classes
    (void)hyperparameters; /// Avoid unused parameter warning
}

void DataMatrixModelAdapter::trainModel() {
    /// Base implementation does nothing - to be overridden by derived classes
}

double DataMatrixModelAdapter::evaluateModel() {
    /// Base implementation returns 0 - to be overridden by derived classes
    return 0.0;
}

/// LinearRegressionAdapter implementation
LinearRegressionAdapter::LinearRegressionAdapter()
    : DataMatrixModelAdapter(),
      intercept_(0.0),
      fitIntercept_(true),
      alpha_(0.0),
      solver_("sgd"),
      maxIter_(1000) {}

std::string LinearRegressionAdapter::toString() const {
    std::string result = "LinearRegression Model with hyperparameters:\n";
    result += "  fit_intercept: " + std::string(fitIntercept_ ? "true" : "false") + "\n";
    result += "  alpha: " + std::to_string(alpha_) + "\n";
    result += "  solver: " + solver_ + "\n";
    result += "  max_iter: " + std::to_string(maxIter_) + "\n";
    
    if (!coefficients_.empty()) {
        result += "Model trained with " + std::to_string(coefficients_.size()) + " coefficients\n";
        result += "Intercept: " + std::to_string(intercept_) + "\n";
    } else {
        result += "Model not yet trained\n";
    }
    
    return result;
}

void LinearRegressionAdapter::applyHyperparameters(const Config& hyperparameters) {
    /// Set default values
    fitIntercept_ = true;
    alpha_ = 0.0;
    solver_ = "sgd";
    maxIter_ = 1000;
    
    /// Apply custom hyperparameters if provided
    for (const auto& [name, value] : hyperparameters) {
        if (name == "fit_intercept" && std::holds_alternative<bool>(value)) {
            fitIntercept_ = std::get<bool>(value);
        } else if (name == "alpha" && std::holds_alternative<float>(value)) {
            alpha_ = std::get<float>(value);
        } else if (name == "solver" && std::holds_alternative<std::string>(value)) {
            solver_ = std::get<std::string>(value);
        } else if (name == "max_iter" && std::holds_alternative<int>(value)) {
            maxIter_ = std::get<int>(value);
        }
    }
    
    if (verbose_) {
        std::cout << "Linear Regression configured with:\n"
                  << "  fit_intercept: " << (fitIntercept_ ? "true" : "false") << "\n"
                  << "  alpha: " << alpha_ << "\n"
                  << "  solver: " << solver_ << "\n"
                  << "  max_iter: " << maxIter_ << std::endl;
    }
}

void LinearRegressionAdapter::trainModel() {
    if (!dataLoaded_) {
        throw std::runtime_error("Cannot train model: no data loaded");
    }
    
    if (xTrain_.empty() || xTrain_[0].empty()) {
        throw std::runtime_error("Cannot train model: empty training data");
    }
    
    /// Get dimensions
    const size_t n_samples = xTrain_.size();
    const size_t n_features = xTrain_[0].size();
    
    /// Initialize coefficients
    coefficients_.resize(n_features, 0.0);
    intercept_ = 0.0;
    
    if (solver_ == "sgd") {
        /// Stochastic Gradient Descent implementation
        std::vector<size_t> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        
        /// Learning rate (could be a hyperparameter)
        double learning_rate = 0.01;
        
        /// Create a random number generator for shuffling
        std::random_device rd;
        std::mt19937 g(rd());
        
        for (int iter = 0; iter < maxIter_; ++iter) {
            /// Shuffle indices for stochastic updates using modern C++ approach
            std::shuffle(indices.begin(), indices.end(), g);
            
            for (size_t idx : indices) {
                /// Compute prediction
                double y_pred = intercept_;
                for (size_t j = 0; j < n_features; ++j) {
                    y_pred += coefficients_[j] * xTrain_[idx][j];
                }
                
                /// Compute error
                double error = yTrain_[idx] - y_pred;
                
                /// Update intercept
                if (fitIntercept_) {
                    intercept_ += learning_rate * error;
                }
                
                /// Update coefficients with regularization (L2 penalty)
                for (size_t j = 0; j < n_features; ++j) {
                    coefficients_[j] = coefficients_[j] * (1 - learning_rate * alpha_) +
                                      learning_rate * error * xTrain_[idx][j];
                }
            }
            
            /// Early stopping could be implemented here based on convergence
        }
    } else if (solver_ == "normal_equation") {
        /// Closed-form solution using normal equation
        /// This is a simplified implementation; in practice, you'd use a linear algebra library
        
        /// Implement a very basic normal equation solver
        /// X^T * X * beta = X^T * y
        
        /// Not implementing this for brevity, but in a real implementation
        /// you would solve the normal equation here
    } else {
        throw std::invalid_argument("Unsupported solver: " + solver_);
    }
    
    if (verbose_) {
        std::cout << "Linear Regression model trained with " << n_features
                  << " features and " << n_samples << " samples" << std::endl;
    }
}

double LinearRegressionAdapter::evaluateModel() {
    if (xTest_.empty() || yTest_.empty()) {
        throw std::runtime_error("Cannot evaluate model: no test data");
    }
    
    const size_t n_samples = xTest_.size();
    yPred_.resize(n_samples);
    
    /// Make predictions
    for (size_t i = 0; i < n_samples; ++i) {
        yPred_[i] = intercept_;
        for (size_t j = 0; j < coefficients_.size(); ++j) {
            yPred_[i] += coefficients_[j] * xTest_[i][j];
        }
    }
    
    /// Calculate R^2 score (higher is better)
    double r2 = calculateR2();
    
    if (verbose_) {
        double mse = calculateMSE();
        double rmse = calculateRMSE();
        double mae = calculateMAE();
        
        std::cout << "Model evaluation metrics:\n"
                  << "  R^2: " << r2 << "\n"
                  << "  MSE: " << mse << "\n"
                  << "  RMSE: " << rmse << "\n"
                  << "  MAE: " << mae << std::endl;
    }
    
    /// Return R^2 score as our model performance metric
    return r2;
}

}
