#ifndef SEVERINE_RANDOM_SEARCH_HPP
#define SEVERINE_RANDOM_SEARCH_HPP
#include "searchStrategy.hpp"
namespace severine{
class RandomSearch : public SearchStrategy {
public:
    RandomSearch(const SearchSpace& searchSpace, std::mt19937& rng);
    Config nextConfiguration() override;
    void update(const EvaluationResult& result) override;
};
}
#endif 