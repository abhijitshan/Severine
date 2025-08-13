#ifndef SEVERINE_BAYESIAN_OPTIMIZATION_HPP
#define SEVERINE_BAYESIAN_OPTIMIZATION_HPP
#include "searchStrategy.hpp"
#include "hyperparameter.hpp"
#include "vector"
#include "cmath"
#include "limits"
#include "memory"
#include "random"
namespace severine {
class Matrix {
private:
    size_t rows_;
    size_t cols_;
    std::vector<double> data_;
public:
    Matrix() : rows_(0), cols_(0) {}
    Matrix(size_t rows, size_t cols, double value = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, value) {}
    double& operator()(size_t i, size_t j) {
        return data_[i * cols_ + j];
    }
    const double& operator()(size_t i, size_t j) const {
        return data_[i * cols_ + j];
    }
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
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
    std::vector<double> getRow(size_t i) const {
        std::vector<double> row(cols_);
        for (size_t j = 0; j < cols_; j++) {
            row[j] = data_[i * cols_ + j];
        }
        return row;
    }
    static Matrix identity(size_t size) {
        Matrix m(size, size, 0.0);
        for (size_t i = 0; i < size; i++) {
            m(i, i) = 1.0;
        }
        return m;
    }
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
            throw std::runtime_error("Determinant for matrices larger than 3x3 not implemented");
        }
    }
    Matrix inverse() const {
        if (rows_ != cols_) {
            throw std::runtime_error("Inverse only defined for square matrices");
        }
        size_t n = rows_;
        Matrix result = identity(n);
        Matrix temp = *this;
        for (size_t i = 0; i < n; i++) {
            double max_val = std::abs(temp(i, i));
            size_t max_row = i;
            for (size_t j = i + 1; j < n; j++) {
                if (std::abs(temp(j, i)) > max_val) {
                    max_val = std::abs(temp(j, i));
                    max_row = j;
                }
            }
            if (max_row != i) {
                for (size_t j = 0; j < n; j++) {
                    std::swap(temp(i, j), temp(max_row, j));
                    std::swap(result(i, j), result(max_row, j));
                }
            }
            if (std::abs(temp(i, i)) < 1e-10) {
                throw std::runtime_error("Matrix is singular, cannot invert");
            }
            double pivot = temp(i, i);
            for (size_t j = 0; j < n; j++) {
                temp(i, j) /= pivot;
                result(i, j) /= pivot;
            }
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
class Vector {
public:
    Vector() {}
    Vector(size_t size, double value = 0.0) : data_(size, value) {}
    Vector(const std::vector<double>& data) : data_(data) {}
    double& operator()(size_t i) { return data_[i]; }
    const double& operator()(size_t i) const { return data_[i]; }
    size_t size() const { return data_.size(); }
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
    const std::vector<double>& data() const { return data_; }
    std::vector<double>& data() { return data_; }
private:
    std::vector<double> data_;
};
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
class GaussianProcess;
class AcquisitionFunction;
enum class KernelType {
    RBF,          
    MATERN,       
    LINEAR        
};
enum class AcquisitionFunctionType {
    EI,           
    UCB,          
    PI            
};
class BayesianOptimization: public SearchStrategy {
public:
    BayesianOptimization(
        const SearchSpace& searchSpace,
        std::mt19937& rng,
        int initialSamples = 10,                               
        double explorationFactor = 2.0,
        KernelType kernelType = KernelType::RBF,               
        AcquisitionFunctionType acqType = AcquisitionFunctionType::EI,  
        bool adaptExplorationFactor = true                     
    );
    ~BayesianOptimization() override;
    Config nextConfiguration() override;
    void update(const EvaluationResult& result) override;
private:
    Vector normalizeConfig(const Config& config) const;
    Config denormalizeConfig(const Vector& x) const;
    bool needsInitialSampling() const;
    Vector maximizeAcquisitionFunction(const GaussianProcess& gp, const Matrix& bounds) const;
    Vector localOptimize(const Vector& startPoint, const GaussianProcess& gp, const Matrix& bounds) const;
    void updateExplorationFactor();
    std::unordered_map<std::string, int> paramIndices_;
    std::unordered_map<std::string, Hyperparameter::Type> paramTypes_;
    std::unique_ptr<GaussianProcess> model_;
    std::unique_ptr<AcquisitionFunction> acquisitionFn_;
    int initialSamples_;
    double explorationFactor_;
    double initialExplorationFactor_;
    Matrix X_; 
    Vector y_; 
    int initCounter_;
    KernelType kernelType_;
    AcquisitionFunctionType acqType_;
    bool adaptExplorationFactor_;
    std::unordered_map<std::string, double> evaluationCache_;
};
class GaussianProcess {
public:
    GaussianProcess(KernelType kernelType = KernelType::RBF, double alpha = 1e-6);
    void fit(const Matrix& X, const Vector& y);
    std::pair<double, double> predict(const Vector& x) const;
    double logMarginalLikelihood() const;
    void optimizeHyperparameters();
    KernelType getKernelType() const { return kernelType_; }
private:
    Matrix computeKernel(const Matrix& X1, const Matrix& X2) const;
    double rbfKernel(const Vector& x1, const Vector& x2) const;
    double maternKernel(const Vector& x1, const Vector& x2) const;
    double linearKernel(const Vector& x1, const Vector& x2) const;
    std::vector<double> gradientLogLikelihood() const;
    double lengthScale_;
    double signalVariance_;
    double noiseVariance_;
    double alpha_;
    Matrix X_;
    Vector y_;
    Matrix K_;
    Matrix K_inv_;
    bool fitted_;
    KernelType kernelType_;
};
class AcquisitionFunction {
public:
    virtual ~AcquisitionFunction() = default;
    virtual double operator()(const Vector& x, const GaussianProcess& gp) const = 0;
    Vector maximize(const GaussianProcess& gp, const Matrix& bounds, std::mt19937& rng) const;
protected:
    Vector localOptimize(const Vector& startPoint, const GaussianProcess& gp, const Matrix& bounds) const;
    Vector computeGradient(const Vector& x, const GaussianProcess& gp, double epsilon = 1e-5) const;
};
class ExpectedImprovement : public AcquisitionFunction {
public:
    ExpectedImprovement(double xi = 0.01);
    double operator()(const Vector& x, const GaussianProcess& gp) const override;
    void updateBestValue(double value);
private:
    double xi_;
    double bestValue_;
};
class UpperConfidenceBound : public AcquisitionFunction {
public:
    UpperConfidenceBound(double kappa = 2.0);
    double operator()(const Vector& x, const GaussianProcess& gp) const override;
    void setKappa(double kappa) { kappa_ = kappa; }
private:
    double kappa_;
};
class ProbabilityOfImprovement : public AcquisitionFunction {
public:
    ProbabilityOfImprovement(double xi = 0.01);
    double operator()(const Vector& x, const GaussianProcess& gp) const override;
    void updateBestValue(double value);
private:
    double xi_;
    double bestValue_;
};
} 
#endif 