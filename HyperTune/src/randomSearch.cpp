//
//  randomSearch.cpp
//  HyperTune
//
//  Created by Abhijit Shanbhag on 26/03/25.
//
#include "../include/randomSearch.hpp"
#include "algorithm"

namespace hypertune {
SearchStrategy::SearchStrategy(const SearchSpace& searchSpace, std::mt19937& rng): searchSpace_(searchSpace), rng_(rng), iteration_(0){}


const EvaluationResult& SearchStrategy::getBestResult() const {
    if (results_.empty()) {
        throw std::runtime_error("No Results Available");
    }
    return *std::max_element(results_.begin(), results_.end());
}

const std::vector<EvaluationResult>& SearchStrategy::getAllResults() const {
    return results_;
}

/// Implementing `RandomSearch` Methods
RandomSearch::RandomSearch(const SearchSpace& searchSpace, std::mt19937& rng):SearchStrategy(searchSpace, rng){}
Config RandomSearch::nextConfiguration(){
    Config config=searchSpace_.sampleConfiguration(rng_);
    iteration_++;
    return config;
}

void RandomSearch::update(const EvaluationResult& result){
    results_.push_back(result);
}
}


