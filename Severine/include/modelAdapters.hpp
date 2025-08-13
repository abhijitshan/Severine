#ifndef SEVERINE_MODEL_ADAPTERS_HPP
#define SEVERINE_MODEL_ADAPTERS_HPP
#include "modelInterface.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <random>
#include <mutex>
namespace severine {
class DataMatrixModelAdapter : public MLFmkModelAdapter {
public:
    DataMatrixModelAdapter();
    void loadTrainingData(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    void loadTestData(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    double calculateMSE() const;
    double calculateRMSE() const;
    double calculateMAE() const;
    double calculateR2() const;
    void setVerbose(bool verbose) override {
        std::lock_guard<std::mutex> lock(mutex_);
        verbose_ = verbose;
    }
protected:
    void applyHyperparameters(const Config& hyperparameters) override;
    void trainModel() override;
    double evaluateModel() override;
    virtual std::shared_ptr<MLFmkModelAdapter> cloneDataMatrixImpl() const = 0;
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
    std::shared_ptr<MLFmkModelAdapter> cloneImpl() const override;
    std::shared_ptr<MLFmkModelAdapter> cloneDataMatrixImpl() const override;
private:
    std::vector<double> coefficients_;
    double intercept_;
    bool fitIntercept_;
    double alpha_;
    std::string solver_;
    int maxIter_;
};
class RandomForestAdapter : public DataMatrixModelAdapter {
public:
    RandomForestAdapter();
    ~RandomForestAdapter() = default;
    std::string toString() const override;
protected:
    void applyHyperparameters(const Config& hyperparameters) override;
    void trainModel() override;
    double evaluateModel() override;
    std::shared_ptr<MLFmkModelAdapter> cloneImpl() const override;
    std::shared_ptr<MLFmkModelAdapter> cloneDataMatrixImpl() const override;
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
class SVMAdapter : public DataMatrixModelAdapter {
public:
    SVMAdapter();
    ~SVMAdapter() = default;
    std::string toString() const override;
protected:
    void applyHyperparameters(const Config& hyperparameters) override;
    void trainModel() override;
    double evaluateModel() override;
    std::shared_ptr<MLFmkModelAdapter> cloneImpl() const override;
    std::shared_ptr<MLFmkModelAdapter> cloneDataMatrixImpl() const override;
private:
    std::vector<double> supportVectors_;
    std::vector<double> dualCoeff;
    double intercept_;
    double C_; 
    std::string kernel_; 
    double gamma_; 
    double epsilon_; 
    int maxIter_; 
    double tol_; 
};
class NeuralNetworkAdapter : public DataMatrixModelAdapter {
public:
    NeuralNetworkAdapter();
    ~NeuralNetworkAdapter() = default;
    std::string toString() const override;
protected:
    void applyHyperparameters(const Config& hyperparameters) override;
    void trainModel() override;
    double evaluateModel() override;
    std::shared_ptr<MLFmkModelAdapter> cloneImpl() const override;
    std::shared_ptr<MLFmkModelAdapter> cloneDataMatrixImpl() const override;
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
#endif 