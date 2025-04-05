//
//
//  hyperparameter.hpp
//  HyperTune
//
//  Created by Abhijit Shanbhag on 26/03/25.
//
#ifndef HYPERTUNE_RANDOM_SEARCH_HPP
#define HYPERTUNE_RANDOM_SEARCH_HPP
#include "searchStrategy.hpp"
namespace hypertune{
class RandomSearch : public SearchStrategy {
public:
    RandomSearch(const SearchSpace& searchSpace, std::mt19937& rng);
    
    /// Gnerate Random Configuration
    Config nextConfiguration() override;
    
    /// Update with a new reuslt
    void update(const EvaluationResult& result) override;
};
}

#endif // HYPERTUNE_RANDOM_SEARCH_HPP

