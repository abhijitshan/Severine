//
//  bayesianOptimization.hpp
//  HyperTune
//
//  Created by Abhijit Shanbhag on 15/04/25.
//

#ifndef HYPERTUNE_BAYESIAN_OPTIMIZATION_HPP
#define HYPERTUNE_BAYESIAN_OPTIMIZATION_HPP
#include "searchStrategy.hpp"
#include "hyperparameter.hpp"
#include "vector"
#include "cmath"
#include "limits"
#include "memory"
#include "random"
#include <stdexcept>

namespace hypertune {

/// Note
/// I am including implementation of certain mathematical structures from ChatGPT and other websites, tweaking around using ChatGPT to match our requirements if from websites
class Matrix {
private:
    /// Stores the count of number of rows
    size_t rows_;
    
    /// Stores the count of number of columns
    size_t cols_;
    
    /// Simple 1D Array, flattened out represnetation of the Matrix
    std::vector<double> data_;
    
public:
    /// Initialisers
    Matrix() : rows_(0), cols_(0) {}
    Matrix(size_t rows, size_t cols, double value = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, value) {}
    
    /// The matrix is stored in a 1D vector (`data_`), so the element at row `i` and column `j` is accessed via the formula `i * cols_ + j`, where `i` is the row index and `   j` is the column index.
    double& operator()(size_t i, size_t j) {
        return data_[i * cols_ + j];
    }
    
    /// Simple Access. The matrix is stored in a 1D vector (`data_`), so the element at row `i` and column `j` is accessed via the formula `i * cols_ + j`, where `i` is the row index and `   j` is the column index.
    const double& operator()(size_t i, size_t j) const {
        return data_[i * cols_ + j];
    }
    
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    
    /// Adds a row to the matrix
    /// - Parameter row: The row value to add
    void addRow(const std::vector<double>& row) {
        if (cols_ == 0) {
            cols_ = row.size();
        } else if (row.size() != cols_) {
            throw std::runtime_error("Row size doesn't match matrix columns");
        }
        
        for (const double& val : row) {
            data_.push_back(val);
        }
        rows_++;
    }
    
    /// Get a row as a vector
    std::vector<double> getRow(size_t i) const {
        std::vector<double> row(cols_);
        for (size_t j = 0; j < cols_; j++) {
            row[j] = data_[i * cols_ + j];
        }
        return row;
    }
    
    /// Create identity matrix
    static Matrix identity(size_t size) {
        Matrix m(size, size, 0.0);
        for (size_t i = 0; i < size; i++) {
            m(i, i) = 1.0;
        }
        return m;
    }
    
    /// Matrix determinant (for small matrices only)
    double determinant() const {
        if (rows_ != cols_) {
            throw std::runtime_error("Determinant only defined for square matrices");
        }
        
        if (rows_ == 1) {
            return data_[0];
        } else if (rows_ == 2) {
            return data_[0] * data_[3] - data_[1] * data_[2];
        } else if (rows_ == 3) {
            return data_[0] * (data_[4] * data_[8] - data_[5] * data_[7])
                 - data_[1] * (data_[3] * data_[8] - data_[5] * data_[6])
                 + data_[2] * (data_[3] * data_[7] - data_[4] * data_[6]);
        } else {
            /// For simplicity, we don't implement determinant for larger matrices
            /// In a real implementation, you would use LU decomposition
            throw std::runtime_error("Determinant for matrices larger than 3x3 not implemented");
        }
    }
    
    /// Matrix inverse using Gauss-Jordan elimination. Same as the Linear Algrebra crap we studied in the last semester
    Matrix inverse() const {
        if (rows_ != cols_) {
            throw std::runtime_error("Inverse only defined for square matrices");
        }
        
        size_t n = rows_;
        Matrix result = identity(n);
        Matrix temp = *this;
        /// Make a copy to avoid modifying the original
        
        /// Gaussian elimination
        for (size_t i = 0; i < n; i++) {
            /// Find pivot
            double max_val = std::abs(temp(i, i));
            size_t max_row = i;
            for (size_t j = i + 1; j < n; j++) {
                if (std::abs(temp(j, i)) > max_val) {
                    max_val = std::abs(temp(j, i));
                    max_row = j;
                }
            }
            
            /// Swap rows if necessary
            if (max_row != i) {
                for (size_t j = 0; j < n; j++) {
                    std::swap(temp(i, j), temp(max_row, j));
                    std::swap(result(i, j), result(max_row, j));
                }
            }
            
            /// Check for singularity
            if (std::abs(temp(i, i)) < 1e-10) {
                throw std::runtime_error("Matrix is singular, cannot invert");
            }
            
            /// Scale pivot row
            double pivot = temp(i, i);
            for (size_t j = 0; j < n; j++) {
                temp(i, j) /= pivot;
                result(i, j) /= pivot;
            }
            
            /// Eliminate other rows
            for (size_t j = 0; j < n; j++) {
                if (j != i) {
                    double factor = temp(j, i);
                    for (size_t k = 0; k < n; k++) {
                        temp(j, k) -= factor * temp(i, k);
                        result(j, k) -= factor * result(i, k);
                    }
                }
            }
        }
        return result;
    }
};

/// Matrix multiplication
inline Matrix operator*(const Matrix& a, const Matrix& b) {
    if (a.cols() != b.rows()) {
        throw std::runtime_error("Matrix dimensions don't match for multiplication");
    }
    
    Matrix result(a.rows(), b.cols(), 0.0);
    for (size_t i = 0; i < a.rows(); i++) {
        for (size_t j = 0; j < b.cols(); j++) {
            double sum = 0.0;
            for (size_t k = 0; k < a.cols(); k++) {
                sum += a(i, k) * b(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

/// Vector class (just a wrapper around `std::vector` with some operations)
class Vector {
public:
    Vector() {}
    Vector(size_t size, double value = 0.0) : data_(size, value) {}
    Vector(const std::vector<double>& data) : data_(data) {}
    
    double& operator()(size_t i) { return data_[i]; }
    const double& operator()(size_t i) const { return data_[i]; }
    
    size_t size() const { return data_.size(); }
    
    /// Vector dot product
    double dot(const Vector& other) const {
        if (size() != other.size()) {
            throw std::runtime_error("Vector dimensions don't match for dot product");
        }
        
        double sum = 0.0;
        for (size_t i = 0; i < size(); i++) {
            sum += data_[i] * other.data_[i];
        }
        return sum;
    }
    
    /// Squared Euclidean distance
    double squaredDistance(const Vector& other) const {
        if (size() != other.size()) {
            throw std::runtime_error("Vector dimensions don't match for distance");
        }
        
        double sum = 0.0;
        for (size_t i = 0; i < size(); i++) {
            double diff = data_[i] - other.data_[i];
            sum += diff * diff;
        }
        return sum;
    }
    
    /// Access the underlying data
    const std::vector<double>& data() const { return data_; }
    std::vector<double>& data() { return data_; }
    
private:
    std::vector<double> data_;
};

/// Vector multiplication (matrix-vector)
inline Vector operator*(const Matrix& m, const Vector& v) {
    if (m.cols() != v.size()) {
        throw std::runtime_error("Dimensions don't match for matrix-vector multiplication");
    }
    
    Vector result(m.rows(), 0.0);
    for (size_t i = 0; i < m.rows(); i++) {
        double sum = 0.0;
        for (size_t j = 0; j < m.cols(); j++) {
            sum += m(i, j) * v(j);
        }
        result(i) = sum;
    }
    return result;
}

// Keep your existing Matrix and Vector classes unchanged

/// Forward declarations
class GaussianProcess;
class AcquisitionFunction;

/// Available kernel types for the Gaussian Process
enum class KernelType {
    RBF,          // Radial Basis Function (Gaussian)
    MATERN,       // Matérn kernel with nu=5/2
    LINEAR        // Linear kernel
};

/// Available acquisition function types
enum class AcquisitionFunctionType {
    EI,           // Expected Improvement
    UCB,          // Upper Confidence Bound
    PI            // Probability of Improvement
};

/// Bayesian Optimization search strategy.
class BayesianOptimization: public SearchStrategy {
public:
    /// Constructor with enhanced parameters
    BayesianOptimization(
        const SearchSpace& searchSpace,
        std::mt19937& rng,
        int initialSamples = 10,                               // Increased from 5 to 10
        double explorationFactor = 2.0,
        KernelType kernelType = KernelType::RBF,               // Default to RBF kernel
        AcquisitionFunctionType acqType = AcquisitionFunctionType::EI,  // Default to EI
        bool adaptExplorationFactor = true                     // Adapt kappa over time
    );
    
    ~BayesianOptimization() override;
    
    /// Generate next configuration to evaluate using Bayesian Optimization
    Config nextConfiguration() override;
    
    /// Update the strategy with a new evaluation result
    void update(const EvaluationResult& result) override;
    
private:
    /// Normalize a configuration to the [0,1] range for each dimension
    Vector normalizeConfig(const Config& config) const;
    
    /// Denormalize from [0,1] range back to the original hyperparameter space
    Config denormalizeConfig(const Vector& x) const;
    
    /// Use random search for the initial samples
    bool needsInitialSampling() const;
    
    /// Enhanced acquisition function maximization (replaces random search)
    Vector maximizeAcquisitionFunction(const GaussianProcess& gp, const Matrix& bounds) const;
    
    /// Local optimization to refine a promising point
    Vector localOptimize(const Vector& startPoint, const GaussianProcess& gp, const Matrix& bounds) const;
    
    /// Adapt exploration-exploitation trade-off based on iteration
    void updateExplorationFactor();
    
    /// Map from hyperparameter names to their indices in the vector representation
    std::unordered_map<std::string, int> paramIndices_;
    
    /// Store the hyperparameter types for denormalization
    std::unordered_map<std::string, Hyperparameter::Type> paramTypes_;
    
    /// Surrogate model and acquisition function
    std::unique_ptr<GaussianProcess> model_;
    std::unique_ptr<AcquisitionFunction> acquisitionFn_;
    
    /// Number of random samples to use for initialization
    int initialSamples_;
    
    /// Exploration-exploitation trade-off parameter
    double explorationFactor_;
    double initialExplorationFactor_;
    
    /// Matrices to store the training data for the surrogate model
    Matrix X_; /// Normalized configurations
    Vector y_; /// Objective values
    
    /// Counter for initialization phase
    int initCounter_;
    
    /// Settings for the Bayesian optimization
    KernelType kernelType_;
    AcquisitionFunctionType acqType_;
    bool adaptExplorationFactor_;
    
    /// Cache of previous evaluations to avoid redundant ones
    std::unordered_map<std::string, double> evaluationCache_;
};

/// Enhanced Gaussian Process model for Bayesian Optimization
class GaussianProcess {
public:
    GaussianProcess(KernelType kernelType = KernelType::RBF, double alpha = 1e-6);
    
    /// Fit the GP model to the training data
    void fit(const Matrix& X, const Vector& y);
    
    /// Predict mean and variance at a given point
    std::pair<double, double> predict(const Vector& x) const;
    
    /// Get the log marginal likelihood of the model
    double logMarginalLikelihood() const;
    
    /// Enhanced optimization of hyperparameters using gradient-based method
    void optimizeHyperparameters();
    
    /// Getter for the current kernel type
    KernelType getKernelType() const { return kernelType_; }
    
private:
    /// Compute kernel matrix based on the selected kernel type
    Matrix computeKernel(const Matrix& X1, const Matrix& X2) const;
    
    /// RBF (Gaussian) kernel
    double rbfKernel(const Vector& x1, const Vector& x2) const;
    
    /// Matérn kernel with nu=5/2
    double maternKernel(const Vector& x1, const Vector& x2) const;
    
    /// Linear kernel
    double linearKernel(const Vector& x1, const Vector& x2) const;
    
    /// Approximate gradient of log likelihood w.r.t. kernel hyperparameters
    std::vector<double> gradientLogLikelihood() const;
    
    /// Kernel parameters
    double lengthScale_;
    double signalVariance_;
    double noiseVariance_;
    
    /// Regularization parameter for numerical stability
    double alpha_;
    
    /// Training data
    Matrix X_;
    Vector y_;
    
    /// Precomputed kernel matrix and its inverse
    Matrix K_;
    Matrix K_inv_;
    
    /// Flag to check if the model has been fitted
    bool fitted_;
    
    /// Type of kernel to use
    KernelType kernelType_;
};

/// Base class for acquisition functions
class AcquisitionFunction {
public:
    virtual ~AcquisitionFunction() = default;
    
    /// Calculate the acquisition value at a given point
    virtual double operator()(const Vector& x, const GaussianProcess& gp) const = 0;
    
    /// Improved optimization of acquisition function (hybrid global-local)
    Vector maximize(const GaussianProcess& gp, const Matrix& bounds, std::mt19937& rng) const;
    
protected:
    /// Helper function to perform local optimization with numerical gradient
    Vector localOptimize(const Vector& startPoint, const GaussianProcess& gp, const Matrix& bounds) const;
    
    /// Compute numerical gradient of acquisition function
    Vector computeGradient(const Vector& x, const GaussianProcess& gp, double epsilon = 1e-5) const;
};

/// Expected Improvement acquisition function
class ExpectedImprovement : public AcquisitionFunction {
public:
    ExpectedImprovement(double xi = 0.01);
    
    double operator()(const Vector& x, const GaussianProcess& gp) const override;
    
    /// Set the current best objective value
    void updateBestValue(double value);
    
private:
    /// Trade-off parameter between exploration and exploitation
    double xi_;
    
    /// Current best observed value
    double bestValue_;
};

/// Upper Confidence Bound acquisition function
class UpperConfidenceBound : public AcquisitionFunction {
public:
    UpperConfidenceBound(double kappa = 2.0);
    
    double operator()(const Vector& x, const GaussianProcess& gp) const override;
    
    /// Update the kappa value (for adaptive exploration)
    void setKappa(double kappa) { kappa_ = kappa; }
    
private:
    /// Trade-off parameter between exploration and exploitation
    double kappa_;
};

/// Probability of Improvement acquisition function (new)
class ProbabilityOfImprovement : public AcquisitionFunction {
public:
    ProbabilityOfImprovement(double xi = 0.01);
    
    double operator()(const Vector& x, const GaussianProcess& gp) const override;
    
    /// Set the current best objective value
    void updateBestValue(double value);
    
private:
    /// Trade-off parameter between exploration and exploitation
    double xi_;
    
    /// Current best observed value
    double bestValue_;
};

} // namespace hypertune

#endif // HYPERTUNE_BAYESIAN_OPTIMIZATION_HPP
