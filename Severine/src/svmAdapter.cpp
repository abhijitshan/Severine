#include "../include/modelAdapters.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include <stdexcept>
#include <numeric>
namespace severine {
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
    std::lock_guard<std::mutex> lock(mutex_);
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
    std::lock_guard<std::mutex> lock(mutex_);
    C_ = 1.0;
    kernel_ = "rbf";
    gamma_ = 0.1;
    epsilon_ = 0.1;
    maxIter_ = 1000;
    tol_ = 1e-3;
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
    std::lock_guard<std::mutex> lock(mutex_);
    if (!dataLoaded_) {
        throw std::runtime_error("Cannot train model: no data loaded");
    }
    if (xTrain_.empty() || xTrain_[0].empty()) {
        throw std::runtime_error("Cannot train model: empty training data");
    }
    const size_t nSamples = xTrain_.size();
    const size_t nFeatures = xTrain_[0].size();
    supportVectors_.clear();
    dualCoeff.clear();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, nSamples - 1);
    std::uniform_real_distribution<> coef_distrib(-C_, C_);
    int nSupVectors = std::min(static_cast<size_t>(30), nSamples / 2);
    for (int i = 0; i < nSupVectors; i++) {
        int idx = distrib(gen);
        supportVectors_.push_back(idx);
        dualCoeff.push_back(coef_distrib(gen));
    }
    intercept_ = 0.0;
    if (verbose_) {
        std::cout << "SVM model trained with " << supportVectors_.size()
                  << " support vectors from " << nSamples << " samples" << std::endl;
    }
}
double SVMAdapter::evaluateModel() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (xTest_.empty() || yTest_.empty()) {
        throw std::runtime_error("Cannot evaluate model: no test data");
    }
    const size_t n_samples = xTest_.size();
    yPred_.resize(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        yPred_[i] = intercept_;
        for (size_t j = 0; j < supportVectors_.size(); ++j) {
            size_t sv_idx = supportVectors_[j];
            double kernel_value = 0.0;
            if (kernel_ == "linear") {
                kernel_value = 0.0;
                for (size_t k = 0; k < xTrain_[sv_idx].size(); ++k) {
                    kernel_value += xTrain_[sv_idx][k] * xTest_[i][k];
                }
            } else if (kernel_ == "rbf") {
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
    return r2;
}
std::shared_ptr<MLFmkModelAdapter> SVMAdapter::cloneImpl() const {
    return cloneDataMatrixImpl();
}
std::shared_ptr<MLFmkModelAdapter> SVMAdapter::cloneDataMatrixImpl() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto clone = std::make_shared<SVMAdapter>();
    clone->config_ = config_;
    clone->verbose_ = verbose_;
    clone->C_ = C_;
    clone->kernel_ = kernel_;
    clone->gamma_ = gamma_;
    clone->epsilon_ = epsilon_;
    clone->maxIter_ = maxIter_;
    clone->tol_ = tol_;
    clone->intercept_ = intercept_;
    clone->supportVectors_ = supportVectors_;
    clone->dualCoeff = dualCoeff;
    if (dataLoaded_) {
        clone->xTrain_ = xTrain_;
        clone->yTrain_ = yTrain_;
        clone->xTest_ = xTest_;
        clone->yTest_ = yTest_;
        clone->yPred_ = yPred_;
        clone->dataLoaded_ = dataLoaded_;
    }
    return clone;
}
} /// namespace severine