//
//  svmAdapter.cpp
//  HyperTune
//
//  Created by Abhijit Shanbhag on 07/04/25.
//

#include "../include/modelAdapters.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include <stdexcept>
#include <numeric>

namespace hypertune {

SVMAdapter::SVMAdapter()
    : DataMatrixModelAdapter(),
      intercept_(0.0),
      C_(1.0),
      kernel_("rbf"),
      gamma_(0.1),
      epsilon_(0.1),
      maxIter_(1000),
      tol_(1e-3) {}

std::string SVMAdapter::toString() const {
    std::string result = "SVM Model with hyperparameters:\n";
    result += "  C: " + std::to_string(C_) + "\n";
    result += "  kernel: " + kernel_ + "\n";
    result += "  gamma: " + std::to_string(gamma_) + "\n";
    result += "  epsilon: " + std::to_string(epsilon_) + "\n";
    result += "  max_iter: " + std::to_string(maxIter_) + "\n";
    result += "  tolerance: " + std::to_string(tol_) + "\n";
    
    if (!supportVectors_.empty()) {
        result += "Model trained with " + std::to_string(supportVectors_.size()) + " support vectors\n";
    } else {
        result += "Model not yet trained\n";
    }
    
    return result;
}

void SVMAdapter::applyHyperparameters(const Config& hyperparameters) {
    /// Set default values
    C_ = 1.0;
    kernel_ = "rbf";
    gamma_ = 0.1;
    epsilon_ = 0.1;
    maxIter_ = 1000;
    tol_ = 1e-3;
    
    /// Apply custom hyperparameters if provided
    for (const auto& [name, value] : hyperparameters) {
        if (name == "C" && std::holds_alternative<float>(value)) {
            C_ = std::get<float>(value);
        } else if (name == "kernel" && std::holds_alternative<std::string>(value)) {
            kernel_ = std::get<std::string>(value);
        } else if (name == "gamma" && std::holds_alternative<float>(value)) {
            gamma_ = std::get<float>(value);
        } else if (name == "epsilon" && std::holds_alternative<float>(value)) {
            epsilon_ = std::get<float>(value);
        } else if (name == "max_iter" && std::holds_alternative<int>(value)) {
            maxIter_ = std::get<int>(value);
        } else if (name == "tol" && std::holds_alternative<float>(value)) {
            tol_ = std::get<float>(value);
        }
    }
    
    if (verbose_) {
        std::cout << "SVM configured with:\n"
                  << "  C: " << C_ << "\n"
                  << "  kernel: " << kernel_ << "\n"
                  << "  gamma: " << gamma_ << "\n"
                  << "  epsilon: " << epsilon_ << "\n"
                  << "  max_iter: " << maxIter_ << "\n"
                  << "  tolerance: " << tol_ << std::endl;
    }
}

void SVMAdapter::trainModel() {
    if (!dataLoaded_) {
        throw std::runtime_error("Cannot train model: no data loaded");
    }
    
    if (xTrain_.empty() || xTrain_[0].empty()) {
        throw std::runtime_error("Cannot train model: empty training data");
    }
    
    /// Get dimensions
    const size_t nSamples = xTrain_.size();
    const size_t nFeatures = xTrain_[0].size();
    
    /// This is a simplified implementation that doesn't actually implement SVM training
    /// In a real implementation, you would implement the SVM optimization algorithm
    
    /// Instead, we'll just create some random support vectors for demonstration
    supportVectors_.clear();
    dualCoeff.clear();
    
    /// Select a subset of training samples as support vectors (simplified)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, nSamples - 1);
    std::uniform_real_distribution<> coef_distrib(-C_, C_);
    
    /// Number of support vectors (simplified - would be determined by the algorithm)
    int nSupVectors = std::min(static_cast<size_t>(30), nSamples / 2);
    
    /// Create support vectors
    for (int i = 0; i < nSupVectors; i++) {
        int idx = distrib(gen);
        supportVectors_.push_back(idx);
        dualCoeff.push_back(coef_distrib(gen));
    }
    
    /// Calculate intercept (simplified)
    intercept_ = 0.0;
    
    if (verbose_) {
        std::cout << "SVM model trained with " << supportVectors_.size()
                  << " support vectors from " << nSamples << " samples" << std::endl;
    }
}

double SVMAdapter::evaluateModel() {
    if (xTest_.empty() || yTest_.empty()) {
        throw std::runtime_error("Cannot evaluate model: no test data");
    }
    
    const size_t n_samples = xTest_.size();
    yPred_.resize(n_samples);
    
    /// Make predictions (simplified)
    for (size_t i = 0; i < n_samples; ++i) {
        /// This is a very simplified prediction - not actual SVM
        /// In a real implementation, you would use the kernel trick
        /// and compute the decision function
        
        /// For demonstration, we'll just use a weighted sum of distances to support vectors
        yPred_[i] = intercept_;
        
        for (size_t j = 0; j < supportVectors_.size(); ++j) {
            size_t sv_idx = supportVectors_[j];
            double kernel_value = 0.0;
            
            /// Compute kernel value (simplified)
            if (kernel_ == "linear") {
                /// Linear kernel: K(x,y) = <x,y>
                kernel_value = 0.0;
                for (size_t k = 0; k < xTrain_[sv_idx].size(); ++k) {
                    kernel_value += xTrain_[sv_idx][k] * xTest_[i][k];
                }
            } else if (kernel_ == "rbf") {
                /// RBF kernel: K(x,y) = exp(-gamma||x-y||^2)
                double dist_sq = 0.0;
                for (size_t k = 0; k < xTrain_[sv_idx].size(); ++k) {
                    double diff = xTrain_[sv_idx][k] - xTest_[i][k];
                    dist_sq += diff * diff;
                }
                kernel_value = std::exp(-gamma_ * dist_sq);
            }
            
            yPred_[i] += dualCoeff[j] * kernel_value;
        }
    }
    
    /// Calculate R^2 score (higher is better)
    double r2 = calculateR2();
    
    if (verbose_) {
        double mse = calculateMSE();
        double rmse = calculateRMSE();
        double mae = calculateMAE();
        
        std::cout << "SVM evaluation metrics:\n"
                  << "  R^2: " << r2 << "\n"
                  << "  MSE: " << mse << "\n"
                  << "  RMSE: " << rmse << "\n"
                  << "  MAE: " << mae << std::endl;
    }
    
    /// Return R^2 score as our model performance metric
    return r2;
}

} /// namespace hypertune
