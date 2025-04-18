
//
//  randomSearch.cpp
//  HyperTune
//
//  Created by Abhijit Shanbhag on 26/03/25.
//
#include "../include/randomSearch.hpp"
#include <algorithm>

namespace hypertune {

// SearchStrategy base class implementations are now in searchStrategy.cpp

/// Implementing `RandomSearch` Methods
RandomSearch::RandomSearch(const SearchSpace& searchSpace, std::mt19937& rng)
: SearchStrategy(searchSpace, rng) {}

Config RandomSearch::nextConfiguration() {
    Config config = searchSpace_.sampleConfiguration(rng_);
    iteration_++;
    return config;
}

void RandomSearch::update(const EvaluationResult& result) {
    results_.push_back(result);
}
}
