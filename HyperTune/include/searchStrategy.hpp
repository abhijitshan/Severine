//
//  hyperparameter.hpp
//  HyperTune
//
//  Created by Abhijit Shanbhag on 26/03/25.
//
#ifndef HYPERTUNE_SEARCH_STRATEGY_HPP
#define HYPERTUNE_SEARCH_STRATEGY_HPP

#include "hyperparameter.hpp"
#include "modelInterface.hpp"
#include "memory"
#include "random"
#include "vector"

namespace hypertune {
struct EvaluationResult {
    Config configuration;
    double score;
    int iteration;
    
    /// Sorting Results for the evalutation with higher scores as the better ones
    bool operator < (const EvaluationResult& other) const {
        return score < other.score;
    }
};

/// Base Class for Search Strategies
class SearchStrategy{
public:
    SearchStrategy(const SearchSpace& searchSpace, std::mt19937& rng);
    virtual ~SearchStrategy()=default;
    
    /// Generate next configuration to evaluate
    virtual Config nextConfiguration()=0;
    
    /// Update the strategy with a new evaluation result
    virtual void update(const EvaluationResult& result)=0;
    
    /// Get the best configuration so far
    virtual const EvaluationResult& getBestResult() const;
    
    /// Get all results collected so far
    virtual const std::vector<EvaluationResult>& getAllResults() const;
protected:
    const SearchSpace& searchSpace_;
    std::mt19937& rng_;
    std::vector<EvaluationResult> results_;
    int iteration_;
};


}

#endif // HYPERTUNE_SEARCH_STRATEGY_HPP

