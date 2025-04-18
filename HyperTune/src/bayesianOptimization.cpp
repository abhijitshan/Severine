//
//  bayesianOptimization.cpp
//  HyperTune
//
//  Created by Abhijit Shanbhag on 18/04/25.
//


#include "../include/bayesianOptimization.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>

namespace hypertune {

///
/// BayesianOptimization Implementation
///

BayesianOptimization::BayesianOptimization(
    const SearchSpace& searchSpace,
    std::mt19937& rng,
    int initialSamples,
    double explorationFactor)
    : SearchStrategy(searchSpace, rng),
      initialSamples_(initialSamples),
      explorationFactor_(explorationFactor),
      initCounter_(0)
{
    /// Initialize parameter indices and types
    int idx = 0;
    for (const auto& [name, param] : searchSpace_.getHyperparameters()) {
        paramIndices_[name] = idx++;
        paramTypes_[name] = param->getType();
    }
    
    /// Initialize surrogate model and acquisition function
    model_ = std::make_unique<GaussianProcess>();
    
    /// We'll use UCB as the default acquisition function
    acquisitionFn_ = std::make_unique<UpperConfidenceBound>(explorationFactor_);
}

BayesianOptimization::~BayesianOptimization() = default;

Config BayesianOptimization::nextConfiguration() {
    /// Use random search for initial samples
    if (needsInitialSampling()) {
        initCounter_++;
        return searchSpace_.sampleConfiguration(rng_);
    }
    
    /// Fit the GP model to the data collected so far
    model_->fit(X_, y_);
    
    /// Define bounds for optimization (normalized to [0,1])
    int dims = paramIndices_.size();
    Matrix bounds(dims, 2);
    for (int i = 0; i < dims; i++) {
        bounds(i, 0) = 0.0; /// Lower bound
        bounds(i, 1) = 1.0; /// Upper bound
    }
    
    /// Find the configuration that maximizes the acquisition function
    Vector x_next = acquisitionFn_->maximize(*model_, bounds, rng_);
    
    /// Convert the normalized vector back to a configuration
    Config nextConfig = denormalizeConfig(x_next);
    
    iteration_++;
    return nextConfig;
}

void BayesianOptimization::update(const EvaluationResult& result) {
    results_.push_back(result);
    
    /// Add the new sample to our training data
    Vector x = normalizeConfig(result.configuration);
    double y_value = result.score;
    
    /// If this is the first update, initialize matrices
    if (X_.rows() == 0) {
        X_ = Matrix(1, paramIndices_.size());
        for (size_t i = 0; i < x.size(); i++) {
            X_(0, i) = x(i);
        }
        
        y_ = Vector(1);
        y_(0) = y_value;
    } else {
        /// Add the new sample to the existing matrices
        Matrix new_X(X_.rows() + 1, X_.cols());
        
        /// Copy existing data
        for (size_t i = 0; i < X_.rows(); i++) {
            for (size_t j = 0; j < X_.cols(); j++) {
                new_X(i, j) = X_(i, j);
            }
        }
        
        /// Add new row
        for (size_t j = 0; j < x.size(); j++) {
            new_X(X_.rows(), j) = x(j);
        }
        
        X_ = new_X;
        
        /// Add new y value
        Vector new_y(y_.size() + 1);
        for (size_t i = 0; i < y_.size(); i++) {
            new_y(i) = y_(i);
        }
        new_y(y_.size()) = y_value;
        
        y_ = new_y;
    }
    
    /// Update the best value for the acquisition function if using Expected Improvement
    if (auto* ei = dynamic_cast<ExpectedImprovement*>(acquisitionFn_.get())) {
        ei->updateBestValue(results_.back().score);
    }
}

Vector BayesianOptimization::normalizeConfig(const Config& config) const {
    Vector x(paramIndices_.size(), 0.0);
    
    for (const auto& [name, value] : config) {
        auto it = paramIndices_.find(name);
        if (it != paramIndices_.end()) {
            int idx = it->second;
            auto typeIt = paramTypes_.find(name);
            Hyperparameter::Type type = typeIt->second;
            
            /// Normalize based on parameter type
            switch (type) {
                case Hyperparameter::Type::INTEGER: {
                    int intValue = std::get<int>(value);
                    const auto* param = dynamic_cast<const IntegerHyperparameter*>(
                        searchSpace_.getHyperparameters().at(name).get());
                    int lower = param->getLower();
                    int upper = param->getUpper();
                    x(idx) = static_cast<double>(intValue - lower) / (upper - lower);
                    break;
                }
                case Hyperparameter::Type::FLOAT: {
                    float floatValue = std::get<float>(value);
                    const auto* param = dynamic_cast<const FloatHyperparameter*>(
                        searchSpace_.getHyperparameters().at(name).get());
                    float lower = param->getLower();
                    float upper = param->getUpper();
                    x(idx) = static_cast<double>(floatValue - lower) / (upper - lower);
                    break;
                }
                case Hyperparameter::Type::BOOLEAN: {
                    bool boolValue = std::get<bool>(value);
                    x(idx) = boolValue ? 1.0 : 0.0;
                    break;
                }
                case Hyperparameter::Type::CATEGORICAL: {
                    std::string strValue = std::get<std::string>(value);
                    const auto* param = dynamic_cast<const CategoricalHyperparameter*>(
                        searchSpace_.getHyperparameters().at(name).get());
                    const auto& values = param->getValues();
                    auto it = std::find(values.begin(), values.end(), strValue);
                    int pos = std::distance(values.begin(), it);
                    x(idx) = static_cast<double>(pos) / (values.size() - 1);
                    break;
                }
            }
        }
    }
    
    return x;
}

Config BayesianOptimization::denormalizeConfig(const Vector& x) const {
    Config config;
    
    for (const auto& [name, idx] : paramIndices_) {
        auto typeIt = paramTypes_.find(name);
        Hyperparameter::Type type = typeIt->second;
        
        /// Denormalize based on parameter type
        switch (type) {
            case Hyperparameter::Type::INTEGER: {
                const auto* param = dynamic_cast<const IntegerHyperparameter*>(
                    searchSpace_.getHyperparameters().at(name).get());
                int lower = param->getLower();
                int upper = param->getUpper();
                int value = static_cast<int>(std::round(x(idx) * (upper - lower) + lower));
                value = std::max(lower, std::min(upper, value)); /// Clamp to bounds
                config[name] = value;
                break;
            }
            case Hyperparameter::Type::FLOAT: {
                const auto* param = dynamic_cast<const FloatHyperparameter*>(
                    searchSpace_.getHyperparameters().at(name).get());
                float lower = param->getLower();
                float upper = param->getUpper();
                float value = static_cast<float>(x(idx) * (upper - lower) + lower);
                value = std::max(lower, std::min(upper, value)); /// Clamp to bounds
                config[name] = value;
                break;
            }
            case Hyperparameter::Type::BOOLEAN: {
                bool value = x(idx) >= 0.5;
                config[name] = value;
                break;
            }
            case Hyperparameter::Type::CATEGORICAL: {
                const auto* param = dynamic_cast<const CategoricalHyperparameter*>(
                    searchSpace_.getHyperparameters().at(name).get());
                const auto& values = param->getValues();
                int pos = static_cast<int>(std::round(x(idx) * (values.size() - 1)));
                pos = std::max(0, std::min(static_cast<int>(values.size()) - 1, pos));
                config[name] = values[pos];
                break;
            }
        }
    }
    
    return config;
}

bool BayesianOptimization::needsInitialSampling() const {
    return initCounter_ < initialSamples_;
}

///
/// GaussianProcess Implementation
///

GaussianProcess::GaussianProcess(double alpha)
    : lengthScale_(1.0),
      signalVariance_(1.0),
      noiseVariance_(0.1),
      alpha_(alpha),
      fitted_(false)
{
}

void GaussianProcess::fit(const Matrix& X, const Vector& y) {
    /// Store the training data
    X_ = X;
    y_ = y;
    
    /// Compute the kernel matrix
    K_ = computeKernel(X_, X_);
    
    /// Add noise variance to the diagonal for numerical stability
    for (size_t i = 0; i < K_.rows(); ++i) {
        K_(i, i) += noiseVariance_ + alpha_;
    }
    
    /// Compute the inverse of the kernel matrix
    K_inv_ = K_.inverse();
    
    fitted_ = true;
    
    /// Optimize hyperparameters if we have enough data
    if (X_.rows() >= 5) {
        optimizeHyperparameters();
    }
}

std::pair<double, double> GaussianProcess::predict(const Vector& x) const {
    if (!fitted_) {
        throw std::runtime_error("GaussianProcess model has not been fitted yet");
    }
    
    /// Convert x to a matrix for kernel computation
    Matrix x_mat(1, x.size());
    for (size_t i = 0; i < x.size(); i++) {
        x_mat(0, i) = x(i);
    }
    
    /// Compute kernel between x and training points
    Matrix k_star = computeKernel(x_mat, X_);
    
    /// Compute mean prediction: k_star * K_inv * y
    double mean = 0.0;
    for (size_t i = 0; i < K_inv_.rows(); i++) {
        double sum = 0.0;
        for (size_t j = 0; j < K_inv_.cols(); j++) {
            sum += k_star(0, j) * K_inv_(j, i);
        }
        mean += sum * y_(i);
    }
    
    /// Compute variance: k(x,x) - k_star * K_inv * k_star^T
    double k_x_x = computeKernel(x_mat, x_mat)(0, 0);
    double var = k_x_x;
    
    for (size_t i = 0; i < K_inv_.rows(); i++) {
        for (size_t j = 0; j < K_inv_.cols(); j++) {
            var -= k_star(0, i) * K_inv_(i, j) * k_star(0, j);
        }
    }
    
    /// Ensure variance is positive
    var = std::max(0.0, var);
    
    return {mean, var};
}

Matrix GaussianProcess::computeKernel(const Matrix& X1, const Matrix& X2) const {
    Matrix K(X1.rows(), X2.rows());
    
    /// RBF (Gaussian) kernel
    for (size_t i = 0; i < X1.rows(); ++i) {
        for (size_t j = 0; j < X2.rows(); ++j) {
            double sq_dist = 0.0;
            for (size_t k = 0; k < X1.cols(); ++k) {
                double diff = X1(i, k) - X2(j, k);
                sq_dist += diff * diff;
            }
            K(i, j) = signalVariance_ * std::exp(-0.5 * sq_dist / (lengthScale_ * lengthScale_));
        }
    }
    
    return K;
}

double GaussianProcess::logMarginalLikelihood() const {
    if (!fitted_) {
        throw std::runtime_error("GaussianProcess model has not been fitted yet");
    }
    
    /// Compute log marginal likelihood
    /// log p(y|X) = -0.5 * (y^T * K_inv * y + log|K| + n*log(2π))
    
    /// Compute y^T * K_inv * y
    double quad_form = 0.0;
    for (size_t i = 0; i < y_.size(); i++) {
        double sum = 0.0;
        for (size_t j = 0; j < y_.size(); j++) {
            sum += K_inv_(i, j) * y_(j);
        }
        quad_form += y_(i) * sum;
    }
    
    /// For simplicity, we use a simple approximation for log|K|
    /// In a real implementation, you would compute this more accurately
    double log_det = 0.0;
    for (size_t i = 0; i < K_.rows(); i++) {
        log_det += std::log(K_(i, i));
    }
    
    size_t n = y_.size();
    double normalizer = n * std::log(2 * M_PI);
    
    return -0.5 * (quad_form + log_det + normalizer);
}

void GaussianProcess::optimizeHyperparameters() {
    /// Simple grid search for hyperparameter optimization
    /// In a real implementation, you would use a proper optimization algorithm
    
    double bestLikelihood = -std::numeric_limits<double>::infinity();
    double bestLengthScale = lengthScale_;
    double bestSignalVar = signalVariance_;
    double bestNoiseVar = noiseVariance_;
    
    std::vector<double> lengthScales = {0.1, 0.5, 1.0, 2.0, 5.0};
    std::vector<double> signalVariances = {0.1, 0.5, 1.0, 2.0, 5.0};
    std::vector<double> noiseVariances = {0.01, 0.05, 0.1, 0.5, 1.0};
    
    for (double ls : lengthScales) {
        for (double sv : signalVariances) {
            for (double nv : noiseVariances) {
                lengthScale_ = ls;
                signalVariance_ = sv;
                noiseVariance_ = nv;
                
                /// Recompute kernel with new hyperparameters
                K_ = computeKernel(X_, X_);
                for (size_t i = 0; i < K_.rows(); ++i) {
                    K_(i, i) += noiseVariance_ + alpha_;
                }
                K_inv_ = K_.inverse();
                
                double likelihood = logMarginalLikelihood();
                if (likelihood > bestLikelihood) {
                    bestLikelihood = likelihood;
                    bestLengthScale = ls;
                    bestSignalVar = sv;
                    bestNoiseVar = nv;
                }
            }
        }
    }
    
    /// Set the best hyperparameters
    lengthScale_ = bestLengthScale;
    signalVariance_ = bestSignalVar;
    noiseVariance_ = bestNoiseVar;
    
    /// Recompute kernel with best hyperparameters
    K_ = computeKernel(X_, X_);
    for (size_t i = 0; i < K_.rows(); ++i) {
        K_(i, i) += noiseVariance_ + alpha_;
    }
    K_inv_ = K_.inverse();
}

///
/// AcquisitionFunction Implementation
///

Vector AcquisitionFunction::maximize(
    const GaussianProcess& gp,
    const Matrix& bounds,
    std::mt19937& rng) const
{
    /// Use a simple random search to maximize the acquisition function
    /// In a real implementation, you would use a proper optimization algorithm
    
    int dims = bounds.rows();
    int numSamples = 1000;
    
    /// Generate random samples within the bounds
    std::vector<Vector> samples;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (int i = 0; i < numSamples; ++i) {
        Vector sample(dims);
        for (int j = 0; j < dims; ++j) {
            double lower = bounds(j, 0);
            double upper = bounds(j, 1);
            sample(j) = lower + dist(rng) * (upper - lower);
        }
        samples.push_back(sample);
    }
    
    /// Evaluate the acquisition function at all samples
    std::vector<double> values(numSamples);
    
    #pragma omp parallel for
    for (int i = 0; i < numSamples; ++i) {
        values[i] = (*this)(samples[i], gp);
    }
    
    /// Find the sample with the highest acquisition value
    int bestIdx = 0;
    double bestValue = values[0];
    
    for (int i = 1; i < numSamples; ++i) {
        if (values[i] > bestValue) {
            bestValue = values[i];
            bestIdx = i;
        }
    }
    
    return samples[bestIdx];
}

///
/// ExpectedImprovement Implementation
///

ExpectedImprovement::ExpectedImprovement(double xi)
    : xi_(xi), bestValue_(-std::numeric_limits<double>::infinity())
{
}

double ExpectedImprovement::operator()(const Vector& x, const GaussianProcess& gp) const {
    /// Get the prediction from the GP
    auto [mu, var] = gp.predict(x);
    double sigma = std::sqrt(var);
    
    if (sigma < 1e-6) {
        return 0.0;  /// Avoid division by zero
    }
    
    /// Calculate improvement
    double improvement = mu - bestValue_ - xi_;
    
    /// Calculate Z score
    double z = improvement / sigma;
    
    /// Calculate expected improvement
    /// Φ(z) is the CDF of the standard normal distribution
    /// φ(z) is the PDF of the standard normal distribution
    double cdf = 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
    double pdf = std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI);
    
    double ei = improvement * cdf + sigma * pdf;
    
    return ei > 0.0 ? ei : 0.0;
}

void ExpectedImprovement::updateBestValue(double value) {
    bestValue_ = std::max(bestValue_, value);
}

///
/// UpperConfidenceBound Implementation
///

UpperConfidenceBound::UpperConfidenceBound(double kappa)
    : kappa_(kappa)
{
}

double UpperConfidenceBound::operator()(const Vector& x, const GaussianProcess& gp) const {
    /// Get the prediction from the GP
    auto [mu, var] = gp.predict(x);
    double sigma = std::sqrt(var);
    
    /// Calculate UCB
    return mu + kappa_ * sigma;
}

} // namespace hypertune
