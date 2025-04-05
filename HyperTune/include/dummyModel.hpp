//
//  dummyModel.hpp
//  HyperTune
//
//  Created by Abhijit Shanbhag on 05/04/25.
//

#ifndef HYPERTUNE_DUMMY_MODEL_HPP
#define HYPERTUNE_DUMMY_MODEL_HPP

#include "modelInterface.hpp"
#include <iostream>
#include <random>
#include <unordered_map>
#include <cmath>
#include "thread"

namespace hypertune {

// A dummy model for testing purposes
class DummyModel : public ModelInterface {
public:
    DummyModel() : rng_(std::random_device()()) {}
    
    void train() override {
        // Simulate training time
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    double evaluate() override {
        // Create a deterministic score based on hyperparameters
        // This will help us verify that the optimization is working
        double score = 0.5;  // Base score
        
        if (config_.count("num_layers")) {
            auto num_layers = std::get<int>(config_.at("num_layers"));
            // Optimal number of layers is 3
            score += 0.1 * (1.0 - std::abs(num_layers - 3) / 5.0);
        }
        
        if (config_.count("learning_rate")) {
            auto lr = std::get<float>(config_.at("learning_rate"));
            // Optimal learning rate is around 0.01
            score += 0.1 * (1.0 - std::abs(std::log10(lr) - std::log10(0.01)) / 3.0);
        }
        
        if (config_.count("activation")) {
            auto activation = std::get<std::string>(config_.at("activation"));
            // Optimal activation is "relu"
            if (activation == "relu") {
                score += 0.1;
            }
        }
        
        if (config_.count("use_batch_norm")) {
            auto use_bn = std::get<bool>(config_.at("use_batch_norm"));
            // Optimal is true
            if (use_bn) {
                score += 0.1;
            }
        }
        
        // Add some random noise
        std::normal_distribution<double> noise(0.0, 0.02);
        score += noise(rng_);
        
        // Ensure score is in [0, 1] range
        return std::max(0.0, std::min(1.0, score));
    }
    
    void configure(const Config& hyperparameters) override {
        config_ = hyperparameters;
    }
    
    std::string toString() const override {
        return "DummyModel";
    }

private:
    Config config_;
    std::mt19937 rng_;
};

} // namespace hypertune

#endif // HYPERTUNE_DUMMY_MODEL_HPP
