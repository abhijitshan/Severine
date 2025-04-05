#include "include/hyperparameter.hpp"
#include "include/randomSearch.hpp"
#include "include/tuner.hpp"
#include "include/dummyModel.hpp"
#include <iostream>
#include <memory>
#include <random>
#include <string>

using namespace hypertune;

int main() {
    std::cout << "=== Starting HyperTune ===" << std::endl;
    
    // Create a search space
    std::cout << "Creating search space..." << std::endl;
    SearchSpace searchSpace;
    
    // Add some hyperparameters
    std::cout << "Adding hyperparameters..." << std::endl;
    searchSpace.addHyperparameter(
        std::make_shared<IntegerHyperparameter>("num_layers", 1, 5));
    searchSpace.addHyperparameter(
        std::make_shared<FloatHyperparameter>("learning_rate", 0.0001f, 0.1f,
                                             FloatHyperparameter::Distribution::LOG_UNIFORM));
    searchSpace.addHyperparameter(
        std::make_shared<CategoricalHyperparameter>(
            "activation", std::vector<std::string>{"relu", "tanh", "sigmoid"}));
    searchSpace.addHyperparameter(
        std::make_shared<BooleanHyperparameter>("use_batch_norm"));
    
    // Set up RNG
    std::cout << "Setting up RNG..." << std::endl;
    std::random_device rd;
    std::mt19937 rng(rd());
    
    // Create a model
    std::cout << "Creating model..." << std::endl;
    auto model = std::make_shared<DummyModel>();
    
    // Create random search strategy
    std::cout << "Creating search strategy..." << std::endl;
    auto strategy = std::make_shared<RandomSearch>(searchSpace, rng);
    
    // Configure the tuner
    std::cout << "Configuring tuner..." << std::endl;
    TunerConfig tunerConfig;
    tunerConfig.maxIterations = 20;
    tunerConfig.numThreads = 4;  // Use 4 threads
    tunerConfig.earlyStopStrategy = EarlyStoppingStrategy::NO_IMPROVEMENT;
    tunerConfig.earlyStoppingPatience = 5;
    
    // Create and run the tuner
    std::cout << "Creating tuner..." << std::endl;
    Tuner tuner(model, strategy, tunerConfig);
    
    std::cout << "Running tuner..." << std::endl;
    tuner.tune();
    
    // Get the best result
    std::cout << "Getting best result..." << std::endl;
    const EvaluationResult& best = tuner.getBestResult();
    std::cout << "\nBest configuration found:" << std::endl;
    for (const auto& [name, value] : best.configuration) {
        std::cout << "  " << name << ": ";
        if (std::holds_alternative<int>(value)) {
            std::cout << std::get<int>(value);
        } else if (std::holds_alternative<float>(value)) {
            std::cout << std::get<float>(value);
        } else if (std::holds_alternative<bool>(value)) {
            std::cout << (std::get<bool>(value) ? "true" : "false");
        } else if (std::holds_alternative<std::string>(value)) {
            std::cout << std::get<std::string>(value);
        }
        std::cout << std::endl;
    }
    std::cout << "Score: " << best.score << std::endl;
    
    std::cout << "=== HyperTune completed ===" << std::endl;
    return 0;
}
