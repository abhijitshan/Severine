//
//  searchStrategy.cpp
//  HyperTune
//
//  Created by Abhijit Shanbhag on 18/04/25.
//

#include "../include/searchStrategy.hpp"
#include <algorithm>

namespace hypertune {

SearchStrategy::SearchStrategy(const SearchSpace& searchSpace, std::mt19937& rng)
    : searchSpace_(searchSpace), rng_(rng), iteration_(0) {}

const EvaluationResult& SearchStrategy::getBestResult() const {
    if (results_.empty()) {
        throw std::runtime_error("No Results Available");
    }
    return *std::max_element(results_.begin(), results_.end());
}

const std::vector<EvaluationResult>& SearchStrategy::getAllResults() const {
    return results_;
}

} // namespace hypertune
