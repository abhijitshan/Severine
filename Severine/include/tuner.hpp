#ifndef SEVERINE_TUNER_HPP
#define SEVERINE_TUNER_HPP
#include "hyperparameter.hpp"
#include "searchStrategy.hpp"
#include "modelInterface.hpp"
#include <memory>
#include <random>
#include <functional>
#include <chrono>
#include <string>
#include <fstream>
#include <mutex>
#include <vector>
#include <future>
#include <atomic>
#include <queue>
#include <type_traits>
namespace severine {
enum class EarlyStoppingStrategy {
    NONE,           
    NO_IMPROVEMENT, 
    THRESHOLD       
};
enum class ParallelizationStrategy {
    NONE,           
    THREAD_POOL,    
    ASYNC_TASKS     
};
struct TunerConfig {
    int maxIterations = 100;
    int numThreads = 1;
    bool verbose = true;
    std::string checkpointFile = "";
    int checkpointInteral = 10;
    EarlyStoppingStrategy earlyStopStrategy = EarlyStoppingStrategy::NONE;
    int earlyStoppingPatience = 10;
    double earlyStoppingThreshold = 0.95;
    ParallelizationStrategy parallelStrategy = ParallelizationStrategy::NONE;
};
class Tuner {
public:
    Tuner(std::shared_ptr<ModelInterface> model, std::shared_ptr<SearchStrategy> strategy, const TunerConfig& config = TunerConfig{});
    void tune();
    const EvaluationResult& getBestResult() const;
    const std::vector<EvaluationResult>& getAllResults() const;
    bool saveCheckpoint(const std::string& filename) const;
    bool loadCheckpoint(const std::string& filename);
    bool shouldEarlyStop() const;
private:
    void tuneSequential();
    void tuneSequentialOpenMP();
    EvaluationResult evaluateConfiguration(const Config& config, int iterationIndex);
    void updateBestScore(double score);
    std::shared_ptr<ModelInterface> cloneModel() const;
    std::shared_ptr<ModelInterface> model_;
    std::shared_ptr<SearchStrategy> strategy_;
    TunerConfig config_;
    std::random_device rd_;
    std::mt19937 rng_;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime_;
    std::atomic<int> iterationsSinceImprovement_{0};
    std::atomic<double> bestScore_{-std::numeric_limits<double>::infinity()};
    mutable std::mutex strategyMutex_;
    mutable std::mutex resultsMutex_;
    mutable std::mutex checkpointMutex_;
};
}  
#endif 