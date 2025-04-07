//
//  modelAdapters.hpp
//  HyperTune
//
//  Created by Abhijit Shanbhag on 07/04/25.
//
#ifndef HYPERTUNE_MODEL_ADAPTERS_HPP
#define HYPERTUNE_MODEL_ADAPTERS_HPP
#include "modelInterface.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <random>

namespace hypertune {
class DataMatrixModelAdapter : public MLFmkModelAdapter {
public:
    DataMatrixModelAdapter();
    
    /// Data loading methods
    void loadTrainingData(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    void loadTestData(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    
    /// Evaluation metrics
    double calculateMSE() const;
    double calculateRMSE() const;
    double calculateMAE() const;
    double calculateR2() const;
    
    /// Public method inherited from ModelInterface via MLFmkModelAdapter
    void setVerbose(bool verbose) override {
        verbose_ = verbose;
    }
    
protected:
    /// Abstract methods from MLFmkModelAdapter
    void applyHyperparameters(const Config& hyperparameters) override;
    void trainModel() override;
    double evaluateModel() override;
    
    /// Data storage
    std::vector<std::vector<double>> xTrain_;
    std::vector<double> yTrain_;
    std::vector<std::vector<double>> xTest_;
    std::vector<double> yTest_;
    std::vector<double> yPred_;
    
    bool dataLoaded_;
};

class LinearRegressionAdapter : public DataMatrixModelAdapter {
public:
    LinearRegressionAdapter();
    ~LinearRegressionAdapter() = default;
    std::string toString() const override;
    
protected:
    void applyHyperparameters(const Config& hyperparameters) override;
    void trainModel() override;
    double evaluateModel() override;
    
private:
    std::vector<double> coefficients_;
    double intercept_;
    
    /// Hyperparameters
    bool fitIntercept_;
    double alpha_;
    std::string solver_;
    int maxIter_;
};

/// Random Forest Model Adapter
class RandomForestAdapter : public DataMatrixModelAdapter {
public:
    RandomForestAdapter();
    ~RandomForestAdapter() = default;
    
    std::string toString() const override;
    
protected:
    void applyHyperparameters(const Config& hyperparameters) override;
    void trainModel() override;
    double evaluateModel() override;
    
private:
    struct DecisionTree {
        int maxDepth;
        std::vector<int> featureIndices;
        std::vector<double> thresholds;
        std::vector<double> leafValues;
        
        double predict(const std::vector<double>& features) const;
    };
    std::vector<DecisionTree> trees_;
    std::mt19937 rng_;
    int nEstimators_;
    int maxDepth_;
    int minSamplesSplit_;
    int maxFeatures_;
    std::string criterion_;
};

/// Support Vector Machine Adapter
class SVMAdapter : public DataMatrixModelAdapter {
public:
    SVMAdapter();
    ~SVMAdapter() = default;
    std::string toString() const override;
    
protected:
    void applyHyperparameters(const Config& hyperparameters) override;
    void trainModel() override;
    double evaluateModel() override;
    
private:
    /// Model parameters
    std::vector<double> supportVectors_;
    std::vector<double> dualCoeff;
    double intercept_;
    /// Hyperparameters
    double C_; /// Regularization parameter
    std::string kernel_; /// Kernel type
    double gamma_; /// Kernel coefficient
    double epsilon_; /// Epsilon in the epsilon-SVR model
    int maxIter_; /// Maximum number of iterations
    double tol_; /// Tolerance for stopping criterion
};

/// Neural Network Adapter
class NeuralNetworkAdapter : public DataMatrixModelAdapter {
public:
    NeuralNetworkAdapter();
    ~NeuralNetworkAdapter() = default;
    
    std::string toString() const override;
    
protected:
    void applyHyperparameters(const Config& hyperparameters) override;
    void trainModel() override;
    double evaluateModel() override;
    
private:
    struct Layer {
        int inputSize;
        int outputSize;
        std::vector<std::vector<double>> weights;
        std::vector<double> biases;
        std::string activation;
        bool useBatchNorm;
        
        std::vector<double> forward(const std::vector<double>& input) const;
    };
    std::vector<Layer> layers_;
    int numLayers_;
    std::vector<int> hiddenLayerSizes_;
    std::string activation_;
    std::string optimizer_;
    double learningRate_;
    int batchSize_;
    int epochs_;
    double dropoutRate_;
    bool useBatchNorm_;
};
}

#endif // HYPERTUNE_MODEL_ADAPTERS_HPP
