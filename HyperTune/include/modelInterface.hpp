//
//  hyperparameter.hpp
//  HyperTune
//
//  Created by Abhijit Shanbhag on 26/03/25.
//
#ifndef HYPERTUNE_MODEL_INTERFACE_HPP
#define HYPERTUNE_MODEL_INTERFACE_HPP

#include "string"
#include "variant"
#include "unordered_map"
#include "vector"
#include "memory"

namespace hypertune {
/// Configuration type for passing parameters
using Config = std::unordered_map<std::string, std::variant<int, float, bool, std::string>>;

/// Abstract Base Class for all the models
class ModelInterface {
public:
    virtual ~ModelInterface()=default;
    
    /// Train the model on a given data
    virtual void train()=0;
    
    /// Evalutation mechanism. It'd be great if one of you already starts working on it from tomorrow (**27th of March, 2025**
    /// Naturally, in this case over here, higher value indicates better performance
    virtual double evaluate()=0;
    
    /// Configure model with given hyperparameters
    virtual void configure(const Config& hyperparameters)=0;
    
    /// Get a string reperesentation of models, probably like the parameters and shit like that
    virtual std::string toString() const = 0;
    
    
};


/// Factory function that creates a specific model implementation from the MLPack library.
/// This function implements the Factory Pattern which they've been yapping about for a while in our class, creating and returning appropriate concrete model adapters while exposing only the common `ModelInterface` to the caller.
/// - Parameter modelType: Standard String Type talking about what kind of model are we trying to instansiate from MLPack Library
/// - Returns `ModelInterface`: Type `ModelInterface` Object (Typecasted) from the actual model class that inherits from `MLPackModelAdapter`
/// - Example Usage
/// ```cpp
/// std::unique_ptr<ModelInterface> uniqueModel = createMLPackModel("RandomForest");
/// ```
std::unique_ptr<ModelInterface> createMLPackModel(const std::string& modelType);

/// `MLPackModelAdapter` is an abstract adapter class that bridges between HyperTune's `ModelInterface` and MLPack library's machine learning models.
/// - This allows MLPack models to work with our HyperTune framework
/// - Also, defines common algorithm structure in methods like `train()` while delegating specific steps to subclasses
class MLPackModelAdapter : public ModelInterface{
public:
    MLPackModelAdapter()=default;
    virtual ~MLPackModelAdapter()=default;
    
    void train() override;
    double evaluate() override;
    void configure(const Config& hyperparameter) override;
    std::string toString() const override;
protected:
    virtual void applyHyperparamters(const Config& hyperparameters)=0;
    virtual void trainModel()=0;
    virtual void evalutateModel()=0;
};

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
 //│MLPackModel- │    │createMLPack │
 //│  Adapter    │◄───│   Model()   │
 //└─────────────┘    └─────────────┘
 //       ▲
 //       │ inherits
 //       │
 //┌─────────────┐
 //│MLPackRandom-│
 //│ForestAdapter│
 //└─────────────┘
 //       │ uses
 //       ▼
 //┌─────────────┐
 //│  MLPack     │
 //│RandomForest │
 //└─────────────┘
 ```
 */
