# HyperTune

**HyperTune** is a high-performance hyperparameter tuning framework for machine learning models in C++. Developed as part of the UE22CS343BB1 (Heterogeneous Parallelism) course, this library provides efficient hyperparameter optimization for machine learning models with parallel processing capabilities.

## Features

- **Modular Design**: Extensible architecture with clear interfaces for models and search strategies
- **Multiple ML Models**: Support for Linear Regression, Random Forest, SVM, and Neural Network models
- **Hyperparameter Types**: Boolean, Categorical, Integer, and Float (including log-uniform distribution)
- **Search Strategies**: Random search implementation with extensible framework for additional strategies
- **Early Stopping**: Multiple early stopping criteria to optimize search efficiency
- **Parallelization**: Multi-threaded evaluation using OpenMP
- **Checkpointing**: Save and resume tuning sessions

## Architecture

HyperTune follows a modular architecture with several key components:

1. **Hyperparameter Space**: Define and sample from hyperparameter search spaces
2. **Model Interface**: Common interface for different ML models
3. **Model Adapters**: Implementations for various ML algorithms
4. **Search Strategies**: Algorithms to explore the hyperparameter space
5. **Tuner**: Orchestrates the tuning process with parallel evaluation

## Supported Models

- **Linear Regression**: With regularization options
- **Random Forest**: Decision tree ensemble with customizable parameters
- **Support Vector Machine**: With various kernel options
- **Neural Network**: Multi-layer perceptron with configurable architecture

For now, we have just kept just these models and with further releases, we shall further the project with many more models 

## Usage Example

```cpp
#include "hyperparameter.hpp"
#include "randomSearch.hpp"
#include "tuner.hpp"
#include "modelAdapters.hpp"
#include <iostream>
#include <memory>
#include <random>

using namespace hypertune;

int main() {
    // Generate or load your data
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;
    // ... load or generate data

    // Define search space
    SearchSpace searchSpace;
    searchSpace.addHyperparameter(
        std::make_shared<BooleanHyperparameter>("fit_intercept"));
    searchSpace.addHyperparameter(
        std::make_shared<FloatHyperparameter>("alpha", 0.0001f, 10.0f,
                                           FloatHyperparameter::Distribution::LOG_UNIFORM));
    searchSpace.addHyperparameter(
        std::make_shared<CategoricalHyperparameter>(
            "solver", std::vector<std::string>{"sgd", "normal_equation"}));
    searchSpace.addHyperparameter(
        std::make_shared<IntegerHyperparameter>("max_iter", 100, 2000));

    // Setup random generator
    std::random_device rd;
    std::mt19937 rng(rd());

    // Create model and load data
    auto model = std::make_shared<LinearRegressionAdapter>();
    model->loadTrainingData(X_train, y_train);
    model->loadTestData(X_test, y_test);

    // Create search strategy
    auto strategy = std::make_shared<RandomSearch>(searchSpace, rng);

    // Configure tuner
    TunerConfig tunerConfig;
    tunerConfig.maxIterations = 100;
    tunerConfig.numThreads = 4;
    tunerConfig.earlyStopStrategy = EarlyStoppingStrategy::NO_IMPROVEMENT;
    tunerConfig.earlyStoppingPatience = 10;

    // Create and run tuner
    Tuner tuner(model, strategy, tunerConfig);
    tuner.tune();

    // Get best configuration
    const EvaluationResult& best = tuner.getBestResult();
    std::cout << "Best configuration score: " << best.score << std::endl;

    return 0;
}

```

## Implementation Details

### Hyperparameter Types

- **`BooleanHyperparameter`**: Simple true/false parameter
- **`CategoricalHyperparameter`**: Selection from a discrete set of options
- **`IntegerHyperparameter`**: Integer range with uniform sampling
- **`FloatHyperparameter`**: Continuous range with uniform or log-uniform sampling

### Search Strategies

Currently implemented:

- **Random Search**: Uniform random sampling from the hyperparameter space

Future planned:

- Grid Search
- Bayesian Optimization
- Population-based methods

### Parallelization

HyperTune leverages OpenMP for multi-threaded evaluation of hyperparameter configurations, making effective use of multi-core processors to accelerate the search process.

## Building and Dependencies

HyperTune requires:

- C++17 compatible compiler, preferably Clang
- OpenMP support
- CMake 3.10 or higher (for building)

If you are using Xcode, the project is as simple as using Configured XcodeProj Scheme Configurations and running it! 

## Future Enhancements

- Additional search strategies (Bayesian Optimization, Evolution Strategies)
- GPU acceleration for model training
- Distributed evaluation across multiple machines
- Integration with popular ML frameworks
- More sophisticated early stopping techniques

## Academic Context

This project was developed as part of the UE22CS343BB1 (Heterogeneous Parallelism) course, focusing on leveraging parallel computing techniques to accelerate machine learning workflows. The implementation showcases the application of multi-threading concepts to the hyperparameter optimization problem.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
