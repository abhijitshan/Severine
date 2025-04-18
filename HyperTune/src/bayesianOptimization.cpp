//
//  bayesianOptimization.cpp
//  HyperTune
//
//  Created by Abhijit Shanbhag on 18/04/25.
//

#include "../include/bayesianOptimization.hpp"
#include "../include/hyperparameter.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include <sstream>

namespace hypertune {

///
/// BayesianOptimization Implementation
///

BayesianOptimization::BayesianOptimization(
                                           const SearchSpace& searchSpace,
                                           std::mt19937& rng,
                                           int initialSamples,
                                           double explorationFactor,
                                           KernelType kernelType,
                                           AcquisitionFunctionType acqType,
                                           bool adaptExplorationFactor)
: SearchStrategy(searchSpace, rng),
initialSamples_(initialSamples),
explorationFactor_(explorationFactor),
initialExplorationFactor_(explorationFactor),
initCounter_(0),
kernelType_(kernelType),
acqType_(acqType),
adaptExplorationFactor_(adaptExplorationFactor)
{
    /// Initialize parameter indices and types
    int idx = 0;
    for (const auto& [name, param] : searchSpace_.getHyperparameters()) {
        paramIndices_[name] = idx++;
        paramTypes_[name] = param->getType();
    }
    
    /// Initialize surrogate model with specified kernel
    model_ = std::make_unique<GaussianProcess>(kernelType_);
    
    /// Initialize acquisition function based on specified type
    switch (acqType_) {
        case AcquisitionFunctionType::EI:
            acquisitionFn_ = std::make_unique<ExpectedImprovement>(0.01);
            break;
        case AcquisitionFunctionType::UCB:
            acquisitionFn_ = std::make_unique<UpperConfidenceBound>(explorationFactor_);
            break;
        case AcquisitionFunctionType::PI:
            acquisitionFn_ = std::make_unique<ProbabilityOfImprovement>(0.01);
            break;
        default:
            acquisitionFn_ = std::make_unique<ExpectedImprovement>(0.01);
            break;
    }
}

BayesianOptimization::~BayesianOptimization() = default;

Config BayesianOptimization::nextConfiguration() {
    /// Use random search for initial samples
    if (needsInitialSampling()) {
        initCounter_++;
        
        /// Generate a new random configuration
        Config config = searchSpace_.sampleConfiguration(rng_);
        
        /// Check if we need to adapt exploration factor after initialization phase
        if (initCounter_ == initialSamples_ && adaptExplorationFactor_) {
            updateExplorationFactor();
        }
        
        return config;
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
    
    /// Find the configuration that maximizes the acquisition function (improved method)
    Vector x_next = maximizeAcquisitionFunction(*model_, bounds);
    
    /// Convert the normalized vector back to a configuration
    Config nextConfig = denormalizeConfig(x_next);
    
    /// Update exploration factor if adaptive is enabled
    if (adaptExplorationFactor_) {
        updateExplorationFactor();
    }
    
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
    
    /// Update the best value for acquisition functions that need it
    if (auto* ei = dynamic_cast<ExpectedImprovement*>(acquisitionFn_.get())) {
        ei->updateBestValue(results_.back().score);
    } else if (auto* pi = dynamic_cast<ProbabilityOfImprovement*>(acquisitionFn_.get())) {
        pi->updateBestValue(results_.back().score);
    }
    
    /// Cache the evaluation for future reference
    std::stringstream ss;
    for (const auto& [name, value] : result.configuration) {
        ss << name << ":";
        if (std::holds_alternative<int>(value)) {
            ss << std::get<int>(value);
        } else if (std::holds_alternative<float>(value)) {
            ss << std::get<float>(value);
        } else if (std::holds_alternative<bool>(value)) {
            ss << (std::get<bool>(value) ? "true" : "false");
        } else if (std::holds_alternative<std::string>(value)) {
            ss << std::get<std::string>(value);
        }
        ss << ";";
    }
    evaluationCache_[ss.str()] = result.score;
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
                    
                    /// Apply log-transform if using log-uniform distribution
                    if (param->getDistribution() == FloatHyperparameter::Distribution::LOG_UNIFORM) {
                        // Log transform for better optimization
                        double logLower = std::log(lower);
                        double logUpper = std::log(upper);
                        double logValue = std::log(floatValue);
                        x(idx) = (logValue - logLower) / (logUpper - logLower);
                    } else {
                        x(idx) = static_cast<double>(floatValue - lower) / (upper - lower);
                    }
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
                    // For categorical values, use one-hot encoding internally
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
                
                float value;
                if (param->getDistribution() == FloatHyperparameter::Distribution::LOG_UNIFORM) {
                    // Log transform for better optimization
                    double logLower = std::log(lower);
                    double logUpper = std::log(upper);
                    value = static_cast<float>(std::exp(x(idx) * (logUpper - logLower) + logLower));
                } else {
                    value = static_cast<float>(x(idx) * (upper - lower) + lower);
                }
                
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

Vector BayesianOptimization::maximizeAcquisitionFunction(
                                                         const GaussianProcess& gp, const Matrix& bounds) const
{
    /// Improved hybrid approach:
    /// 1. First generate a grid of points across the space
    /// 2. Evaluate the acquisition function at these points
    /// 3. Pick the top k points and run local optimization from each
    /// 4. Return the best overall solution
    
    const int dims = bounds.rows();
    const int pointsPerDim = std::max(3, static_cast<int>(std::pow(1000.0, 1.0/dims)));
    const int totalPoints = std::min(1000, static_cast<int>(std::pow(pointsPerDim, dims)));
    const int localOptimizationCandidates = std::min(5, totalPoints);
    
    /// Generate candidates using Latin Hypercube Sampling (more efficient than grid)
    std::vector<Vector> candidates;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    /// Generate LHS samples
    std::vector<std::vector<double>> lhsSamples(dims);
    for (int d = 0; d < dims; d++) {
        lhsSamples[d].resize(totalPoints);
        for (int i = 0; i < totalPoints; i++) {
            lhsSamples[d][i] = (i + dist(rng_)) / totalPoints;
        }
        std::shuffle(lhsSamples[d].begin(), lhsSamples[d].end(), rng_);
    }
    
    for (int i = 0; i < totalPoints; i++) {
        Vector sample(dims);
        for (int d = 0; d < dims; d++) {
            double lower = bounds(d, 0);
            double upper = bounds(d, 1);
            sample(d) = lower + lhsSamples[d][i] * (upper - lower);
        }
        candidates.push_back(sample);
    }
    
    /// Evaluate acquisition function at all candidates
    std::vector<std::pair<double, int>> scores;
    for (int i = 0; i < candidates.size(); i++) {
        double score = (*acquisitionFn_)(candidates[i], gp);
        scores.push_back({score, i});
    }
    
    /// Sort by score (descending)
    std::sort(scores.begin(), scores.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    /// Perform local optimization from top candidates
    std::vector<std::pair<double, Vector>> optimizedResults;
    for (int i = 0; i < localOptimizationCandidates; i++) {
        int idx = scores[i].second;
        Vector localResult = localOptimize(candidates[idx], gp, bounds);
        double localScore = (*acquisitionFn_)(localResult, gp);
        optimizedResults.push_back({localScore, localResult});
    }
    
    /// Find the best result
    auto bestResult = std::max_element(optimizedResults.begin(), optimizedResults.end(),
                                       [](const auto& a, const auto& b) { return a.first < b.first; });
    
    return bestResult->second;
}

Vector BayesianOptimization::localOptimize(
                                           const Vector& startPoint, const GaussianProcess& gp, const Matrix& bounds) const
{
    const int maxIter = 50;
    double stepSize = 0.01;
    const double tolerance = 1e-5;
    
    Vector currentPoint = startPoint;
    double currentValue = (*acquisitionFn_)(currentPoint, gp);
    
    for (int iter = 0; iter < maxIter; iter++) {
        /// Compute numerical gradient
        Vector gradient(currentPoint.size(), 0.0);
        for (size_t i = 0; i < currentPoint.size(); i++) {
            Vector perturbed = currentPoint;
            double h = 1e-5; /// Small perturbation
            
            perturbed(i) += h;
            double valuePlus = (*acquisitionFn_)(perturbed, gp);
            
            perturbed = currentPoint;
            perturbed(i) -= h;
            double valueMinus = (*acquisitionFn_)(perturbed, gp);
            
            gradient(i) = (valuePlus - valueMinus) / (2 * h);
        }
        
        /// Update point using gradient ascent
        Vector newPoint = currentPoint;
        for (size_t i = 0; i < currentPoint.size(); i++) {
            newPoint(i) += stepSize * gradient(i);
            
            /// Clamp to bounds
            newPoint(i) = std::max(bounds(i, 0), std::min(bounds(i, 1), newPoint(i)));
        }
        
        /// Check if we've improved
        double newValue = (*acquisitionFn_)(newPoint, gp);
        if (newValue > currentValue) {
            currentPoint = newPoint;
            currentValue = newValue;
        } else {
            /// No improvement - try with smaller step size or terminate
            if (stepSize < tolerance) {
                break;
            }
            /// Reduce step size
            stepSize *= 0.5;
        }
        
        /// Check for convergence
        double gradNorm = 0.0;
        for (size_t i = 0; i < gradient.size(); i++) {
            gradNorm += gradient(i) * gradient(i);
        }
        gradNorm = std::sqrt(gradNorm);
        
        if (gradNorm < tolerance) {
            break;
        }
    }
    
    return currentPoint;
}

void BayesianOptimization::updateExplorationFactor() {
    /// Decrease exploration over time (linear decay)
    if (auto* ucb = dynamic_cast<UpperConfidenceBound*>(acquisitionFn_.get())) {
        // Instead of using config_.maxIterations, we'll use an estimated total iterations
        // or simply base decay on current iteration count
        
        // Assuming we'll run for about 4x the number of initial samples
        int estimatedTotalIterations = initialSamples_ * 4;
        double progress = static_cast<double>(iteration_) / estimatedTotalIterations;
        progress = std::min(1.0, progress); // Cap at 1.0
        
        double newKappa = initialExplorationFactor_ * (1.0 - 0.8 * progress);
        ucb->setKappa(newKappa);
    }
}

///
/// GaussianProcess Implementation
///

GaussianProcess::GaussianProcess(KernelType kernelType, double alpha)
: lengthScale_(1.0),
signalVariance_(1.0),
noiseVariance_(0.1),
alpha_(alpha),
fitted_(false),
kernelType_(kernelType)
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

Matrix GaussianProcess::computeKernel(const Matrix& X1, const Matrix& X2) const {
    Matrix K(X1.rows(), X2.rows());
    
    for (size_t i = 0; i < X1.rows(); ++i) {
        Vector x1(X1.cols());
        for (size_t k = 0; k < X1.cols(); ++k) {
            x1(k) = X1(i, k);
        }
        
        for (size_t j = 0; j < X2.rows(); ++j) {
            Vector x2(X2.cols());
            for (size_t k = 0; k < X2.cols(); ++k) {
                x2(k) = X2(j, k);
            }
            
            /// Use the appropriate kernel function
            switch (kernelType_) {
                case KernelType::RBF:
                    K(i, j) = rbfKernel(x1, x2);
                    break;
                case KernelType::MATERN:
                    K(i, j) = maternKernel(x1, x2);
                    break;
                case KernelType::LINEAR:
                    K(i, j) = linearKernel(x1, x2);
                    break;
                default:
                    K(i, j) = rbfKernel(x1, x2);
                    break;
            }
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

std::vector<double> GaussianProcess::gradientLogLikelihood() const {
    /// Approximate gradient of log likelihood w.r.t. hyperparameters
    std::vector<double> grad(3, 0.0); // [length_scale, signal_var, noise_var]
    
    /// Use numerical differentiation
    const double epsilon = 1e-6;
    double base_ll = logMarginalLikelihood();
    
    /// Gradient for length scale
    {
        double saved = lengthScale_;
        const_cast<GaussianProcess*>(this)->lengthScale_ += epsilon;
        
        /// Recompute kernel and inverse with perturbed parameter
        Matrix K_perturbed = computeKernel(X_, X_);
        for (size_t i = 0; i < K_perturbed.rows(); ++i) {
            K_perturbed(i, i) += noiseVariance_ + alpha_;
        }
        Matrix K_inv_perturbed = K_perturbed.inverse();
        
        /// Swap in perturbed matrices temporarily
        Matrix K_saved = K_;
        Matrix K_inv_saved = K_inv_;
        const_cast<GaussianProcess*>(this)->K_ = K_perturbed;
        const_cast<GaussianProcess*>(this)->K_inv_ = K_inv_perturbed;
        
        double perturbed_ll = logMarginalLikelihood();
        grad[0] = (perturbed_ll - base_ll) / epsilon;
        
        /// Restore original matrices and parameter
        const_cast<GaussianProcess*>(this)->K_ = K_saved;
        const_cast<GaussianProcess*>(this)->K_inv_ = K_inv_saved;
        const_cast<GaussianProcess*>(this)->lengthScale_ = saved;
    }
    
    /// Gradient for signal variance
    {
        double saved = signalVariance_;
        const_cast<GaussianProcess*>(this)->signalVariance_ += epsilon;
        
        /// Recompute kernel and inverse
        Matrix K_perturbed = computeKernel(X_, X_);
        for (size_t i = 0; i < K_perturbed.rows(); ++i) {
            K_perturbed(i, i) += noiseVariance_ + alpha_;
        }
        Matrix K_inv_perturbed = K_perturbed.inverse();
        
        /// Swap in perturbed matrices
        Matrix K_saved = K_;
        Matrix K_inv_saved = K_inv_;
        const_cast<GaussianProcess*>(this)->K_ = K_perturbed;
        const_cast<GaussianProcess*>(this)->K_inv_ = K_inv_perturbed;
        
        double perturbed_ll = logMarginalLikelihood();
        grad[1] = (perturbed_ll - base_ll) / epsilon;
        
        /// Restore original matrices and parameter
        const_cast<GaussianProcess*>(this)->K_ = K_saved;
        const_cast<GaussianProcess*>(this)->K_inv_ = K_inv_saved;
        const_cast<GaussianProcess*>(this)->signalVariance_ = saved;
    }
    
    /// Gradient for noise variance
    {
        double saved = noiseVariance_;
        const_cast<GaussianProcess*>(this)->noiseVariance_ += epsilon;
        
        /// Recompute kernel and inverse (only diagonal changes)
        Matrix K_perturbed = K_;
        for (size_t i = 0; i < K_perturbed.rows(); ++i) {
            K_perturbed(i, i) += epsilon;
        }
        Matrix K_inv_perturbed = K_perturbed.inverse();
        
        /// Swap in perturbed matrices
        Matrix K_saved = K_;
        Matrix K_inv_saved = K_inv_;
        const_cast<GaussianProcess*>(this)->K_ = K_perturbed;
        const_cast<GaussianProcess*>(this)->K_inv_ = K_inv_perturbed;
        
        double perturbed_ll = logMarginalLikelihood();
        grad[2] = (perturbed_ll - base_ll) / epsilon;
        
        /// Restore original matrices and parameter
        const_cast<GaussianProcess*>(this)->K_ = K_saved;
        const_cast<GaussianProcess*>(this)->K_inv_ = K_inv_saved;
        const_cast<GaussianProcess*>(this)->noiseVariance_ = saved;
    }
    
    return grad;
}

/// Implement the kernel functions
double GaussianProcess::rbfKernel(const Vector& x1, const Vector& x2) const {
    double sq_dist = 0.0;
    for (size_t i = 0; i < x1.size(); ++i) {
        double diff = x1(i) - x2(i);
        sq_dist += diff * diff;
    }
    return signalVariance_ * std::exp(-0.5 * sq_dist / (lengthScale_ * lengthScale_));
}

double GaussianProcess::maternKernel(const Vector& x1, const Vector& x2) const {
    /// Matérn kernel with nu=5/2
    double sq_dist = 0.0;
    for (size_t i = 0; i < x1.size(); ++i) {
        double diff = x1(i) - x2(i);
        sq_dist += diff * diff;
    }
    
    double d = std::sqrt(5.0 * sq_dist) / lengthScale_;
    return signalVariance_ * (1.0 + d + d*d/3.0) * std::exp(-d);
}

double GaussianProcess::linearKernel(const Vector& x1, const Vector& x2) const {
    double dot_product = 0.0;
    for (size_t i = 0; i < x1.size(); ++i) {
        dot_product += x1(i) * x2(i);
    }
    return signalVariance_ * dot_product;
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

void GaussianProcess::optimizeHyperparameters() {
    /// Enhanced optimization using gradient ascent
    const int maxIter = 50;
    const double initialStepSize = 0.01;
    const double minStepSize = 1e-6;
    const double gradientTolerance = 1e-4;
    
    double stepSize = initialStepSize;
    
    for (int iter = 0; iter < maxIter; iter++) {
        /// Compute gradient of log likelihood
        std::vector<double> grad = gradientLogLikelihood();
        
        /// Apply gradient ascent with adaptive step size
        double lengthScale_new = lengthScale_ + stepSize * grad[0];
        double signalVariance_new = signalVariance_ + stepSize * grad[1];
        double noiseVariance_new = noiseVariance_ + stepSize * grad[2];
        
        /// Ensure parameters remain positive
        lengthScale_new = std::max(1e-3, lengthScale_new);
        signalVariance_new = std::max(1e-3, signalVariance_new);
        noiseVariance_new = std::max(1e-6, noiseVariance_new);
        
        /// Save old parameters and log likelihood
        double oldLengthScale = lengthScale_;
        double oldSignalVariance = signalVariance_;
        double oldNoiseVariance = noiseVariance_;
        double oldLogLikelihood = logMarginalLikelihood();
        
        /// Try new parameters
        lengthScale_ = lengthScale_new;
        signalVariance_ = signalVariance_new;
        noiseVariance_ = noiseVariance_new;
        
        /// Recompute kernel matrix with new parameters
        K_ = computeKernel(X_, X_);
        for (size_t i = 0; i < K_.rows(); ++i) {
            K_(i, i) += noiseVariance_ + alpha_;
        }
        K_inv_ = K_.inverse();
        
        /// Check if log likelihood improved
        double newLogLikelihood = logMarginalLikelihood();
        
        if (newLogLikelihood > oldLogLikelihood) {
            /// Parameters improved the likelihood, continue
        } else {
            /// Revert to old parameters and reduce step size
            lengthScale_ = oldLengthScale;
            signalVariance_ = oldSignalVariance;
            noiseVariance_ = oldNoiseVariance;
            
            /// Recompute kernel with old parameters
            K_ = computeKernel(X_, X_);
            for (size_t i = 0; i < K_.rows(); ++i) {
                K_(i, i) += noiseVariance_ + alpha_;
            }
            K_inv_ = K_.inverse();
            
            /// Reduce step size
            stepSize *= 0.5;
            
            /// If step size gets too small, terminate
            if (stepSize < minStepSize) {
                break;
            }
        }
        
        /// Check for convergence by looking at gradient norm
        double gradNorm = std::sqrt(grad[0]*grad[0] + grad[1]*grad[1] + grad[2]*grad[2]);
        if (gradNorm < gradientTolerance) {
            break;
        }
    }
    
    /// If we don't have enough data for reliable hyperparameter optimization,
    /// use a more robust grid search approach as fallback
    if (X_.rows() < 10) {
        /// Simple grid search for hyperparameter optimization
        /// This is more robust when we have limited data
        
        double bestLikelihood = logMarginalLikelihood();
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
}

///
/// AcquisitionFunction Implementation
///

Vector AcquisitionFunction::maximize(
                                     const GaussianProcess& gp,
                                     const Matrix& bounds,
                                     std::mt19937& rng) const
{
    /// Use a multi-start optimization approach
    const int dims = bounds.rows();
    const int numStarts = 10;
    
    /// Generate initial points using sobol sequence (for better coverage than random)
    std::vector<Vector> startPoints;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (int i = 0; i < numStarts; i++) {
        Vector point(dims);
        for (int j = 0; j < dims; j++) {
            double lower = bounds(j, 0);
            double upper = bounds(j, 1);
            point(j) = lower + dist(rng) * (upper - lower);
        }
        startPoints.push_back(point);
    }
    
    /// Run local optimization from each start point
    std::vector<std::pair<double, Vector>> optimizationResults;
    for (const auto& startPoint : startPoints) {
        Vector optimizedPoint = localOptimize(startPoint, gp, bounds);
        double value = (*this)(optimizedPoint, gp);
        optimizationResults.push_back({value, optimizedPoint});
    }
    
    /// Find the best result
    auto bestResult = std::max_element(optimizationResults.begin(), optimizationResults.end(),
                                       [](const auto& a, const auto& b) { return a.first < b.first; });
    
    return bestResult->second;
}

Vector AcquisitionFunction::localOptimize(
                                          const Vector& startPoint,
                                          const GaussianProcess& gp,
                                          const Matrix& bounds) const
{
    const int maxIter = 100;
    double stepSize = 0.01;
    const double minStepSize = 1e-6;
    const double gradTolerance = 1e-5;
    
    Vector currentPoint = startPoint;
    double currentValue = (*this)(currentPoint, gp);
    
    for (int iter = 0; iter < maxIter; iter++) {
        /// Compute gradient using numerical differentiation
        Vector gradient = computeGradient(currentPoint, gp);
        
        /// Apply gradient ascent step
        Vector newPoint = currentPoint;
        for (size_t i = 0; i < currentPoint.size(); i++) {
            newPoint(i) += stepSize * gradient(i);
            
            /// Clamp to bounds
            newPoint(i) = std::max(bounds(i, 0), std::min(bounds(i, 1), newPoint(i)));
        }
        
        /// Evaluate new point
        double newValue = (*this)(newPoint, gp);
        
        /// Check if improved
        if (newValue > currentValue) {
            currentPoint = newPoint;
            currentValue = newValue;
        } else {
            /// Reduce step size
            stepSize *= 0.5;
            
            if (stepSize < minStepSize) {
                break;
            }
        }
        
        /// Check for convergence
        double gradNorm = 0.0;
        for (size_t i = 0; i < gradient.size(); i++) {
            gradNorm += gradient(i) * gradient(i);
        }
        gradNorm = std::sqrt(gradNorm);
        
        if (gradNorm < gradTolerance) {
            break;
        }
    }
    
    return currentPoint;
}

Vector AcquisitionFunction::computeGradient(
                                            const Vector& x,
                                            const GaussianProcess& gp,
                                            double epsilon) const
{
    Vector gradient(x.size(), 0.0);
    
    /// Base function value
    double baseValue = (*this)(x, gp);
    
    /// Compute gradient using central difference
    for (size_t i = 0; i < x.size(); i++) {
        Vector xPlus = x;
        xPlus(i) += epsilon;
        
        Vector xMinus = x;
        xMinus(i) -= epsilon;
        
        double valuePlus = (*this)(xPlus, gp);
        double valueMinus = (*this)(xMinus, gp);
        
        gradient(i) = (valuePlus - valueMinus) / (2.0 * epsilon);
    }
    
    return gradient;
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
    
    /// Calculate expected improvement using the analytical formula
    /// EI(x) = (μ(x) - f(x+) - ξ) * Φ(z) + σ(x) * φ(z)
    /// where Φ is the CDF and φ is the PDF of the standard normal distribution
    
    /// Calculate CDF and PDF
    double cdf = 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
    double pdf = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * z * z);
    
    double ei = improvement * cdf + sigma * pdf;
    
    return std::max(0.0, ei);
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
    /// UCB(x) = μ(x) + κ * σ(x)
    return mu + kappa_ * sigma;
}

///
/// ProbabilityOfImprovement Implementation
///

ProbabilityOfImprovement::ProbabilityOfImprovement(double xi)
: xi_(xi), bestValue_(-std::numeric_limits<double>::infinity())
{
}

double ProbabilityOfImprovement::operator()(const Vector& x, const GaussianProcess& gp) const {
    /// Get the prediction from the GP
    auto [mu, var] = gp.predict(x);
    double sigma = std::sqrt(var);
    
    if (sigma < 1e-6) {
        return (mu > bestValue_ + xi_) ? 1.0 : 0.0;
    }
    
    /// Calculate improvement
    double improvement = mu - bestValue_ - xi_;
    
    /// Calculate Z score
    double z = improvement / sigma;
    
    /// Calculate probability of improvement
    /// PI(x) = Φ(z)
    /// where Φ is the CDF of the standard normal distribution
    double pi = 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
    
    return pi;
}

void ProbabilityOfImprovement::updateBestValue(double value) {
    bestValue_ = std::max(bestValue_, value);
}

}
