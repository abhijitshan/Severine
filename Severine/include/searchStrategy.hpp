#ifndef SEVERINE_SEARCH_STRATEGY_HPP
#define SEVERINE_SEARCH_STRATEGY_HPP
#include "hyperparameter.hpp"
#include "modelInterface.hpp"
#include "memory"
#include "random"
#include "vector"
namespace severine {
struct EvaluationResult {
    Config configuration;
    double score;
    int iteration;
    bool operator < (const EvaluationResult& other) const {
        return score < other.score;
    }
};
class SearchStrategy{
public:
    SearchStrategy(const SearchSpace& searchSpace, std::mt19937& rng);
    virtual ~SearchStrategy()=default;
    virtual Config nextConfiguration()=0;
    virtual void update(const EvaluationResult& result)=0;
    virtual const EvaluationResult& getBestResult() const;
    virtual const std::vector<EvaluationResult>& getAllResults() const;
protected:
    const SearchSpace& searchSpace_;
    std::mt19937& rng_;
    std::vector<EvaluationResult> results_;
    int iteration_;
};
}
#endif 