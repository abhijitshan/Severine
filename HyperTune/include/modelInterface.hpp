//
//  modelInterface.hpp
//  HyperTune
//
//  Created by Abhijit Shanbhag on 26/03/25.
//
#ifndef HYPERTUNE_MODEL_INTERFACE_HPP
#define HYPERTUNE_MODEL_INTERFACE_HPP

#include <string>
#include <variant>
#include <unordered_map>
#include <vector>
#include <memory>

namespace hypertune {
/// Alias for configuration type for passing parameters
using Config = std::unordered_map<std::string, std::variant<int, float, bool, std::string>>;

/// Abstract Base Class for all the models
class ModelInterface {
public:
    virtual ~ModelInterface() = default;
    
    /// Train the model on a given data
    virtual void train() = 0;
    
    /// Evaluation mechanism - higher values indicate better performance
    virtual double evaluate() = 0;
    
    /// Configure model with given hyperparameters
    virtual void configure(const Config& hyperparameters) = 0;
    
    /// Get a string representation of model and its parameters
    virtual std::string toString() const = 0;
    
    /// Set verbose output mode
    virtual void setVerbose(bool verbose) = 0;
};

/// Factory function that creates a specific model implementation from the existing ML library.
/// This function implements the Factory Pattern, creating and returning appropriate concrete model adapters
/// while exposing only the common `ModelInterface` to the caller.
/// - Parameter modelType: Type of model to instantiate ("LinearRegression", "RandomForest", "SVM", etc.)
/// - Returns: ModelInterface object pointer that can be used to train and evaluate models
/// - Example Usage: std::unique_ptr<ModelInterface> model = createMLFmkModel("RandomForest");
std::unique_ptr<ModelInterface> createMLFmkModel(const std::string& modelType);

/// `MLFmkModelAdapter` is an abstract adapter class that bridges between HyperTune's `ModelInterface`
/// and ML library's machine learning models. This allows existing ML library models to work with our
/// HyperTune framework and defines common algorithm structure in methods like `train()` while
/// delegating specific steps to subclasses.
class MLFmkModelAdapter : public ModelInterface {
public:
    MLFmkModelAdapter() : verbose_(false) {}
    virtual ~MLFmkModelAdapter() = default;
    
    void train() override;
    double evaluate() override;
    void configure(const Config& hyperparameters) override;
    std::string toString() const override;
    
protected:
    virtual void applyHyperparameters(const Config& hyperparameters) = 0;
    virtual void trainModel() = 0;
    virtual double evaluateModel() = 0;
    Config config_;
    bool verbose_;
};

/// Forward declarations of concrete model adapters
class LinearRegressionAdapter;
class RandomForestAdapter;
class SVMAdapter;
class NeuralNetworkAdapter;
}
#endif

/**
 
To understand better
 ```
 //┌─────────────┐
 //│   Client    │
 //└─────────────┘
 //       │ calls
 //       ▼
 //┌──────────────┐
 //│ModelInterface│◄──────────┐
 //└──────────────┘           │
 //       ▲                   │
 //       │ inherits          │ returns
 //       │                   │
 //┌─────────────┐    ┌─────────────┐
 //│MLFmkModel-  │    │createMLFmk  │
 //│  Adapter    │◄───│   Model()   │
 //└─────────────┘    └─────────────┘
 //       ▲
 //       │ inherits
 //       │
 //┌─────────────┐
 //│MLFmkRandom- │
 //│ForestAdapter│
 //└─────────────┘
 //       │ uses
 //       ▼
 //┌─────────────┐
 //│  MLFmk      │
 //│RandomForest │
 //└─────────────┘
 ```
 */
