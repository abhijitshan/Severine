//
//  tuner.cpp
//  HyperTune
//
//  Created by Abhijit Shanbhag on 26/03/25.
//
#include "../include/tuner.hpp"
#include "iostream"
#include "iomanip"
#include "algorithm"
#include "omp.h"

namespace hypertune {
Tuner::Tuner(std::shared_ptr<ModelInterface> model, std::shared_ptr<SearchStrategy> strategy, const TunerConfig& config):model_(model), strategy_(strategy), config_(config), rng_(rd_()){
    ///Setting the  number of threads
    omp_set_num_threads(config_.numThreads);
}

void Tuner::tune(){
    startTime_ = std::chrono::high_resolution_clock::now();
    
    /// Try to load checkpoint if specified
    if (!config_.checkpointFile.empty()) {
        loadCheckpoint(config_.checkpointFile);
    }
    std::vector<EvaluationResult> iterationResults;
    for (int index=0; index<config_.maxIterations; index++) {
        if (config_.verbose) {
            std::cout<<"Iteration Count" << (index+1) << "/" << config_.maxIterations <<"\n";
        }
        Config config = strategy_->nextConfiguration();
        
        model_->configure(config);
        
        model_->train();
        double score=model_->evaluate();
        
        EvaluationResult result{config, score, index};
        strategy_->update(result);
        
        if (score > bestScore_) {
            bestScore_ = score;
            iterationsSinceImprovement_=0;
        }else{
            iterationsSinceImprovement_++;
        }
        
        if (config_.verbose) {
            std::cout<<" Score: " << std::fixed << std::setprecision(6)<<score;
            
            /// Caculate the elapsed time and release in output stream
            auto currentTime=std::chrono::high_resolution_clock::now();
            auto elapsed=std::chrono::duration_cast<std::chrono::seconds>(currentTime-startTime_).count();
            std::cout<<"   |  Elapsed Time: " << elapsed <<"s"<<"\n";
        }
        if (!config_.checkpointFile.empty() &&
            (index+1)%config_.checkpointInteral==0){
            saveCheckpoint(config_.checkpointFile);
        }
        
        /// Early Stop Checks
        if (shouldEarlyStop()) {
            if (config_.verbose) {
                std::cout<<"Early Stopping Triggered at Iteration " << index+1 << "\n";
            }
            break;
        }
        if (config_.verbose) {
            const auto& best=getBestResult();
            std::cout <<"\nBest Configration Score: "<<best.score<< "\t Iteration: " << best.iteration <<"\n";
            for (const auto& [name,value] : best.configuration) {
                std::cout<<" "<<name<<": ";
                if (std::holds_alternative<int>(value)) {
                    std::cout<<std::get<int>(value);
                } else if (std::holds_alternative<float>(value)) {
                    std::cout << std::get<float>(value);
                } else if (std::holds_alternative<bool>(value)) {
                    std::cout << (std::get<bool>(value) ? "true" : "false");
                } else if (std::holds_alternative<std::string>(value)) {
                    std::cout << std::get<std::string>(value);
                }
                std::cout << std::endl;
            }
        }
    }
}
const EvaluationResult& Tuner::getBestResult() const {
    return strategy_->getBestResult();
}

const std::vector<EvaluationResult>& Tuner::getAllResults() const {
    return strategy_->getAllResults();
}

bool Tuner::saveCheckpoint(const std::string& filename) const {
    // For the MVP, we'll implement a simple checkpoint format
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }
    
    /// This is a placeholder for actual serialization
    /// In a complete implementation, you would serialize the state of the
    /// search strategy and all evaluation results
    
    return true;
}

bool Tuner::loadCheckpoint(const std::string& filename) {
    // For the MVP, we'll implement a simple checkpoint format
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }
    
    /// This is a placeholder for actual deserialization
    /// In a complete implementation, you would deserialize the state of the
    /// search strategy and all evaluation results
    
    return true;
}

bool Tuner::shouldEarlyStop() const {
    switch (config_.earlyStopStrategy) {
        case EarlyStoppingStrategy::NONE:
            return false;
        
        case EarlyStoppingStrategy::NO_IMPROVEMENT:
            return iterationsSinceImprovement_ >= config_.earlyStoppingPatience;
        
        case EarlyStoppingStrategy::THRESHOLD:
            return bestScore_ >= config_.earlyStoppingThreshold;
        
        default:
            return false;
    }
}
}
