#include "../include/randomSearch.hpp"
#include <algorithm>
namespace severine {
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