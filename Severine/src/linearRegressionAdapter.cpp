#include "../include/modelAdapters.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <random>
namespace severine {
LinearRegressionAdapter::LinearRegressionAdapter()
    : DataMatrixModelAdapter(),
      intercept_(0.0),
      fitIntercept_(true),
      alpha_(0.0),
      solver_("sgd"),
      maxIter_(1000) {
}
std::string LinearRegressionAdapter::toString() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string result = "LinearRegression Model with hyperparameters:\n";
    result += "  fit_intercept: " + std::string(fitIntercept_ ? "true" : "false") + "\n";
    result += "  alpha: " + std::to_string(alpha_) + "\n";
    result += "  solver: " + solver_ + "\n";
    result += "  max_iter: " + std::to_string(maxIter_) + "\n";
    if (config_.find("tol") != config_.end() && std::holds_alternative<float>(config_.at("tol"))) {
        result += "  tol: " + std::to_string(std::get<float>(config_.at("tol"))) + "\n";
    }
    if (!coefficients_.empty()) {
        result += "Model trained with " + std::to_string(coefficients_.size()) + " coefficients\n";
        result += "Intercept: " + std::to_string(intercept_) + "\n";
    } else {
        result += "Model not yet trained\n";
    }
    return result;
}
void LinearRegressionAdapter::applyHyperparameters(const Config& hyperparameters) {
    fitIntercept_ = true;
    alpha_ = 0.0;
    solver_ = "lsqr";
    maxIter_ = 1000;
    config_["tol"] = 1e-4f;
    for (const auto& [name, value] : hyperparameters) {
        if (name == "fit_intercept" && std::holds_alternative<bool>(value)) {
            fitIntercept_ = std::get<bool>(value);
        } else if (name == "alpha" && std::holds_alternative<float>(value)) {
            alpha_ = std::get<float>(value);
        } else if (name == "solver" && std::holds_alternative<std::string>(value)) {
            solver_ = std::get<std::string>(value);
        } else if (name == "max_iter" && std::holds_alternative<int>(value)) {
            maxIter_ = std::get<int>(value);
        } else if (name == "tol" && std::holds_alternative<float>(value)) {
            config_["tol"] = value;
        }
    }
    if (verbose_) {
        std::cout << "Linear Regression configured with:\n"
                  << "  fit_intercept: " << (fitIntercept_ ? "true" : "false") << "\n"
                  << "  alpha: " << alpha_ << "\n"
                  << "  solver: " << solver_ << "\n"
                  << "  max_iter: " << maxIter_ << "\n";
        if (config_.find("tol") != config_.end() && std::holds_alternative<float>(config_.at("tol"))) {
            std::cout << "  tol: " << std::get<float>(config_.at("tol")) << std::endl;
        }
    }
}
void LinearRegressionAdapter::trainModel() {
    if (!dataLoaded_) {
        throw std::runtime_error("Cannot train model: no data loaded");
    }
    if (xTrain_.empty() || xTrain_[0].empty()) {
        throw std::runtime_error("Cannot train model: empty training data");
    }
    float tolerance = 1e-4f; // Default tolerance
    if (config_.find("tol") != config_.end() && std::holds_alternative<float>(config_.at("tol"))) {
        tolerance = std::get<float>(config_.at("tol"));
    }
    const size_t n_samples = xTrain_.size();
    const size_t n_features = xTrain_[0].size();
    coefficients_.resize(n_features, 0.0);
    intercept_ = 0.0;
    if (solver_ == "sgd") {
        std::vector<size_t> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        double learning_rate = 0.01;
        std::random_device rd;
        std::mt19937 g(rd());
        for (int iter = 0; iter < maxIter_; ++iter) {
            std::shuffle(indices.begin(), indices.end(), g);
            double total_error = 0.0;
            for (size_t idx : indices) {
                double y_pred = intercept_;
                for (size_t j = 0; j < n_features; ++j) {
                    y_pred += coefficients_[j] * xTrain_[idx][j];
                }
                double error = yTrain_[idx] - y_pred;
                total_error += error * error;
                if (fitIntercept_) {
                    intercept_ += learning_rate * error;
                }
                for (size_t j = 0; j < n_features; ++j) {
                    coefficients_[j] = coefficients_[j] * (1 - learning_rate * alpha_) +
                                      learning_rate * error * xTrain_[idx][j];
                }
            }
            double rmse = std::sqrt(total_error / n_samples);
            if (rmse < tolerance) {
                if (verbose_) {
                    std::cout << "SGD converged after " << iter + 1 << " iterations. RMSE: " << rmse << std::endl;
                }
                break;
            }
        }
    } else if (solver_ == "normal_equation") {
        std::vector<std::vector<double>> xtx(n_features, std::vector<double>(n_features, 0.0));
        std::vector<double> xty(n_features, 0.0);
        for (size_t i = 0; i < n_samples; ++i) {
            for (size_t j = 0; j < n_features; ++j) {
                for (size_t k = 0; k < n_features; ++k) {
                    xtx[j][k] += xTrain_[i][j] * xTrain_[i][k];
                }
                xty[j] += xTrain_[i][j] * yTrain_[i];
            }
        }
        for (size_t j = 0; j < n_features; ++j) {
            xtx[j][j] += alpha_;
        }
        for (size_t i = 0; i < n_features; ++i) {
            size_t max_row = i;
            double max_val = std::abs(xtx[i][i]);
            for (size_t j = i + 1; j < n_features; ++j) {
                if (std::abs(xtx[j][i]) > max_val) {
                    max_val = std::abs(xtx[j][i]);
                    max_row = j;
                }
            }
            if (max_row != i) {
                std::swap(xtx[i], xtx[max_row]);
                std::swap(xty[i], xty[max_row]);
            }
            for (size_t j = i + 1; j < n_features; ++j) {
                double factor = xtx[j][i] / xtx[i][i];
                xty[j] -= factor * xty[i];
                for (size_t k = i; k < n_features; ++k) {
                    xtx[j][k] -= factor * xtx[i][k];
                }
            }
        }
        for (int i = n_features - 1; i >= 0; --i) {
            double sum = 0.0;
            for (size_t j = i + 1; j < n_features; ++j) {
                sum += xtx[i][j] * coefficients_[j];
            }
            coefficients_[i] = (xty[i] - sum) / xtx[i][i];
        }
        if (fitIntercept_) {
            double sum = 0.0;
            for (size_t i = 0; i < n_samples; ++i) {
                double y_pred = 0.0;
                for (size_t j = 0; j < n_features; ++j) {
                    y_pred += coefficients_[j] * xTrain_[i][j];
                }
                sum += yTrain_[i] - y_pred;
            }
            intercept_ = sum / n_samples;
        }
    } else if (solver_ == "lsqr") {
        size_t effective_features = n_features + (fitIntercept_ ? 1 : 0);
        std::vector<double> beta(effective_features, 0.0);
        double prev_mse = std::numeric_limits<double>::max();
        for (int iter = 0; iter < maxIter_; ++iter) {
            std::vector<double> residuals(n_samples);
            double total_squared_error = 0.0;
            for (size_t i = 0; i < n_samples; ++i) {
                double y_pred = fitIntercept_ ? beta[0] : 0.0;
                for (size_t j = 0; j < n_features; ++j) {
                    y_pred += beta[j + (fitIntercept_ ? 1 : 0)] * xTrain_[i][j];
                }
                residuals[i] = yTrain_[i] - y_pred;
                total_squared_error += residuals[i] * residuals[i];
            }
            double current_mse = total_squared_error / n_samples;
            if (std::abs(prev_mse - current_mse) < tolerance) {
                if (verbose_) {
                    std::cout << "LSQR converged after " << iter + 1 << " iterations. MSE: " << current_mse << std::endl;
                }
                break;
            }
            prev_mse = current_mse;
            std::vector<double> gradient(effective_features, 0.0);
            if (fitIntercept_) {
                for (size_t i = 0; i < n_samples; ++i) {
                    gradient[0] -= 2.0 * residuals[i] / n_samples;
                }
            }
            for (size_t j = 0; j < n_features; ++j) {
                for (size_t i = 0; i < n_samples; ++i) {
                    gradient[j + (fitIntercept_ ? 1 : 0)] -= 2.0 * residuals[i] * xTrain_[i][j] / n_samples;
                }
                if (alpha_ > 0) {
                    gradient[j + (fitIntercept_ ? 1 : 0)] += 2.0 * alpha_ * beta[j + (fitIntercept_ ? 1 : 0)];
                }
            }
            double step_size = 0.01;
            bool found_better = false;
            for (int attempt = 0; attempt < 10; ++attempt) {
                std::vector<double> new_beta = beta;
                for (size_t j = 0; j < effective_features; ++j) {
                    new_beta[j] -= step_size * gradient[j];
                }
                double new_mse = 0.0;
                for (size_t i = 0; i < n_samples; ++i) {
                    double y_pred = fitIntercept_ ? new_beta[0] : 0.0;
                    for (size_t j = 0; j < n_features; ++j) {
                        y_pred += new_beta[j + (fitIntercept_ ? 1 : 0)] * xTrain_[i][j];
                    }
                    double error = yTrain_[i] - y_pred;
                    new_mse += error * error;
                }
                new_mse /= n_samples;
                if (new_mse < current_mse) {
                    beta = new_beta;
                    found_better = true;
                    break;
                }
                step_size *= 0.5;
            }
            if (!found_better) {
                if (verbose_) {
                    std::cout << "LSQR reached local minimum after " << iter + 1 << " iterations" << std::endl;
                }
                break;
            }
        }
        if (fitIntercept_) {
            intercept_ = beta[0];
            for (size_t j = 0; j < n_features; ++j) {
                coefficients_[j] = beta[j + 1];
            }
        } else {
            for (size_t j = 0; j < n_features; ++j) {
                coefficients_[j] = beta[j];
            }
        }
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
    for (size_t i = 0; i < n_samples; ++i) {
        yPred_[i] = intercept_;
        for (size_t j = 0; j < coefficients_.size(); ++j) {
            yPred_[i] += coefficients_[j] * xTest_[i][j];
        }
    }
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
    return r2;
}
std::shared_ptr<MLFmkModelAdapter> LinearRegressionAdapter::cloneImpl() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cloneDataMatrixImpl();
}
std::shared_ptr<MLFmkModelAdapter> LinearRegressionAdapter::cloneDataMatrixImpl() const {
    auto clone = std::make_shared<LinearRegressionAdapter>();
    clone->config_ = config_;
    clone->verbose_ = verbose_;
    clone->fitIntercept_ = fitIntercept_;
    clone->alpha_ = alpha_;
    clone->solver_ = solver_;
    clone->maxIter_ = maxIter_;
    clone->intercept_ = intercept_;
    if (!coefficients_.empty()) {
        clone->coefficients_.resize(coefficients_.size());
        std::copy(coefficients_.begin(), coefficients_.end(), clone->coefficients_.begin());
    }
    if (dataLoaded_) {
        clone->xTrain_.resize(xTrain_.size());
        for (size_t i = 0; i < xTrain_.size(); i++) {
            clone->xTrain_[i].resize(xTrain_[i].size());
            std::copy(xTrain_[i].begin(), xTrain_[i].end(), clone->xTrain_[i].begin());
        }
        clone->yTrain_.resize(yTrain_.size());
        std::copy(yTrain_.begin(), yTrain_.end(), clone->yTrain_.begin());
        clone->xTest_.resize(xTest_.size());
        for (size_t i = 0; i < xTest_.size(); i++) {
            clone->xTest_[i].resize(xTest_[i].size());
            std::copy(xTest_[i].begin(), xTest_[i].end(), clone->xTest_[i].begin());
        }
        clone->yTest_.resize(yTest_.size());
        std::copy(yTest_.begin(), yTest_.end(), clone->yTest_.begin());
        if (!yPred_.empty()) {
            clone->yPred_.resize(yPred_.size());
            std::copy(yPred_.begin(), yPred_.end(), clone->yPred_.begin());
        }
        clone->dataLoaded_ = true;
    }
    return clone;
}
} // namespace severine