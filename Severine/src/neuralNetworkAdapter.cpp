#include "../include/modelAdapters.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include <stdexcept>
#include <numeric>
namespace severine {
double activate(double x, const std::string& activation) {
    if (activation == "relu") {
        return std::max(0.0, x);
    } else if (activation == "sigmoid") {
        return 1.0 / (1.0 + std::exp(-x));
    } else if (activation == "tanh") {
        return std::tanh(x);
    } else {
        return x;
    }
}
std::vector<double> NeuralNetworkAdapter::Layer::forward(const std::vector<double>& input) const {
    std::vector<double> output(outputSize, 0.0);
    for (int i = 0; i < outputSize; i++) {
        output[i] = biases[i];
        for (int j = 0; j < inputSize; j++) {
            output[i] += weights[i][j] * input[j];
        }
        output[i] = activate(output[i], activation);
    }
    return output;
}
NeuralNetworkAdapter::NeuralNetworkAdapter()
    : DataMatrixModelAdapter(),
      numLayers_(3),
      hiddenLayerSizes_({64, 32}),
      activation_("relu"),
      optimizer_("adam"),
      learningRate_(0.001),
      batchSize_(32),
      epochs_(100),
      dropoutRate_(0.0),
      useBatchNorm_(false) {}
std::string NeuralNetworkAdapter::toString() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string result = "Neural Network Model with hyperparameters:\n";
    result += "  num_layers: " + std::to_string(numLayers_) + "\n";
    result += "  hidden_layer_sizes: [";
    for (size_t i = 0; i < hiddenLayerSizes_.size(); i++) {
        result += std::to_string(hiddenLayerSizes_[i]);
        if (i < hiddenLayerSizes_.size() - 1) {
            result += ", ";
        }
    }
    result += "]\n";
    result += "  activation: " + activation_ + "\n";
    result += "  optimizer: " + optimizer_ + "\n";
    result += "  learning_rate: " + std::to_string(learningRate_) + "\n";
    result += "  batch_size: " + std::to_string(batchSize_) + "\n";
    result += "  epochs: " + std::to_string(epochs_) + "\n";
    result += "  dropout_rate: " + std::to_string(dropoutRate_) + "\n";
    result += "  use_batch_norm: " + std::string(useBatchNorm_ ? "true" : "false") + "\n";
    if (!layers_.empty()) {
        result += "Model trained with " + std::to_string(layers_.size()) + " layers\n";
    } else {
        result += "Model not yet trained\n";
    }
    return result;
}
void NeuralNetworkAdapter::applyHyperparameters(const Config& hyperparameters) {
    std::lock_guard<std::mutex> lock(mutex_);
    numLayers_ = 3;
    hiddenLayerSizes_ = {64, 32};
    activation_ = "relu";
    optimizer_ = "adam";
    learningRate_ = 0.001;
    batchSize_ = 32;
    epochs_ = 100;
    dropoutRate_ = 0.0;
    useBatchNorm_ = false;
    for (const auto& [name, value] : hyperparameters) {
        if (name == "num_layers" && std::holds_alternative<int>(value)) {
            numLayers_ = std::get<int>(value);
        } else if (name == "activation" && std::holds_alternative<std::string>(value)) {
            activation_ = std::get<std::string>(value);
        } else if (name == "optimizer" && std::holds_alternative<std::string>(value)) {
            optimizer_ = std::get<std::string>(value);
        } else if (name == "learning_rate" && std::holds_alternative<float>(value)) {
            learningRate_ = std::get<float>(value);
        } else if (name == "batch_size" && std::holds_alternative<int>(value)) {
            batchSize_ = std::get<int>(value);
        } else if (name == "epochs" && std::holds_alternative<int>(value)) {
            epochs_ = std::get<int>(value);
        } else if (name == "dropout_rate" && std::holds_alternative<float>(value)) {
            dropoutRate_ = std::get<float>(value);
        } else if (name == "use_batch_norm" && std::holds_alternative<bool>(value)) {
            useBatchNorm_ = std::get<bool>(value);
        }
    }
    if (verbose_) {
        std::cout << "Neural Network configured with:\n"
                  << "  num_layers: " << numLayers_ << "\n"
                  << "  activation: " << activation_ << "\n"
                  << "  optimizer: " << optimizer_ << "\n"
                  << "  learning_rate: " << learningRate_ << "\n"
                  << "  batch_size: " << batchSize_ << "\n"
                  << "  epochs: " << epochs_ << "\n"
                  << "  dropout_rate: " << dropoutRate_ << "\n"
                  << "  use_batch_norm: " << (useBatchNorm_ ? "true" : "false") << std::endl;
    }
}
void NeuralNetworkAdapter::trainModel() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!dataLoaded_) {
        throw std::runtime_error("Cannot train model: no data loaded");
    }
    if (xTrain_.empty() || xTrain_[0].empty()) {
        throw std::runtime_error("Cannot train model: empty training data");
    }
    const size_t n_samples = xTrain_.size();
    const size_t n_features = xTrain_[0].size();
    const size_t n_outputs = 1; /// For regression
    layers_.clear();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 0.1);
    Layer inputLayer;
    inputLayer.inputSize = n_features;
    inputLayer.outputSize = hiddenLayerSizes_[0];
    inputLayer.activation = activation_;
    inputLayer.useBatchNorm = useBatchNorm_;
    inputLayer.weights.resize(inputLayer.outputSize, std::vector<double>(inputLayer.inputSize));
    inputLayer.biases.resize(inputLayer.outputSize);
    for (int i = 0; i < inputLayer.outputSize; i++) {
        inputLayer.biases[i] = dist(gen);
        for (int j = 0; j < inputLayer.inputSize; j++) {
            inputLayer.weights[i][j] = dist(gen);
        }
    }
    layers_.push_back(inputLayer);
    for (size_t i = 1; i < hiddenLayerSizes_.size(); i++) {
        Layer hiddenLayer;
        hiddenLayer.inputSize = hiddenLayerSizes_[i-1];
        hiddenLayer.outputSize = hiddenLayerSizes_[i];
        hiddenLayer.activation = activation_;
        hiddenLayer.useBatchNorm = useBatchNorm_;
        hiddenLayer.weights.resize(hiddenLayer.outputSize, std::vector<double>(hiddenLayer.inputSize));
        hiddenLayer.biases.resize(hiddenLayer.outputSize);
        for (int j = 0; j < hiddenLayer.outputSize; j++) {
            hiddenLayer.biases[j] = dist(gen);
            for (int k = 0; k < hiddenLayer.inputSize; k++) {
                hiddenLayer.weights[j][k] = dist(gen);
            }
        }
        layers_.push_back(hiddenLayer);
    }
    Layer outputLayer;
    outputLayer.inputSize = hiddenLayerSizes_.back();
    outputLayer.outputSize = n_outputs;
    outputLayer.activation = "linear";
    outputLayer.useBatchNorm = false;
    outputLayer.weights.resize(outputLayer.outputSize, std::vector<double>(outputLayer.inputSize));
    outputLayer.biases.resize(outputLayer.outputSize);
    for (int i = 0; i < outputLayer.outputSize; i++) {
        outputLayer.biases[i] = dist(gen);
        for (int j = 0; j < outputLayer.inputSize; j++) {
            outputLayer.weights[i][j] = dist(gen);
        }
    }
    layers_.push_back(outputLayer);
    if (verbose_) {
        std::cout << "Neural Network initialized with " << layers_.size()
                  << " layers" << std::endl;
    }
}
double NeuralNetworkAdapter::evaluateModel() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (xTest_.empty() || yTest_.empty()) {
        throw std::runtime_error("Cannot evaluate model: no test data");
    }
    const size_t n_samples = xTest_.size();
    yPred_.resize(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        std::vector<double> activation = xTest_[i]; /// Input layer
        for (const auto& layer : layers_) {
            activation = layer.forward(activation);
        }
        yPred_[i] = activation[0];
    }
    double r2 = calculateR2();
    if (verbose_) {
        double mse = calculateMSE();
        double rmse = calculateRMSE();
        double mae = calculateMAE();
        std::cout << "Neural Network evaluation metrics:\n"
                  << "  R^2: " << r2 << "\n"
                  << "  MSE: " << mse << "\n"
                  << "  RMSE: " << rmse << "\n"
                  << "  MAE: " << mae << std::endl;
    }
    return r2;
}
std::shared_ptr<MLFmkModelAdapter> NeuralNetworkAdapter::cloneImpl() const {
    return cloneDataMatrixImpl();
}
std::shared_ptr<MLFmkModelAdapter> NeuralNetworkAdapter::cloneDataMatrixImpl() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto clone = std::make_shared<NeuralNetworkAdapter>();
    clone->config_ = config_;
    clone->verbose_ = verbose_;
    clone->numLayers_ = numLayers_;
    clone->hiddenLayerSizes_ = hiddenLayerSizes_;
    clone->activation_ = activation_;
    clone->optimizer_ = optimizer_;
    clone->learningRate_ = learningRate_;
    clone->batchSize_ = batchSize_;
    clone->epochs_ = epochs_;
    clone->dropoutRate_ = dropoutRate_;
    clone->useBatchNorm_ = useBatchNorm_;
    clone->layers_ = layers_;
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