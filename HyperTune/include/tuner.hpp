//
//
//  hyperparameter.hpp
//  HyperTune
//
//  Created by Abhijit Shanbhag on 26/03/25.
//
#ifndef HYPERTUNE_TUNER_HPP
#define HYPERTUNE_TUNER_HPP


#include "hyperparameter.hpp"
#include "searchStrategy.hpp"
#include "modelInterface.hpp"

#include "memory"
#include "random"
#include "functional"
#include "chrono"
#include "string"
#include "fstream"

namespace hypertune{
/// Early Stop Options
enum class EarlyStoppingStrategy{
    NONE,   /// No Early Stopping
    NO_IMPROVEMENT, /// Doesn't see any form of improvement
    THRESHOLD   ///Stop if iterations exceeds threshold
};

struct TunerConfig{
    int maxIterations=100;
    int numThreads=1;
    bool verbose=true;
    std::string checkpointFile="";
    int checkpointInteral=10;
    EarlyStoppingStrategy earlyStopStrategy=EarlyStoppingStrategy::NONE;
    int earlyStoppingPatience=10;
    double earlyStoppingThreshold=0.95;
};

class Tuner{
public:
    Tuner(std::shared_ptr<ModelInterface> model, std::shared_ptr<SearchStrategy> strategy, const TunerConfig& config = TunerConfig{});
    
    /// Run the Tuner
    void tune();
    
    /// Get Best Configraution
    const EvaluationResult& getBestResult() const;
    
    /// Get all results
    const std::vector <EvaluationResult>& getAllResults() const;
    
    /// Save current sttae to a check point state
    bool saveCheckpoint(const std::string& filename) const;
    
    /// Load from a checkpoint file
    bool loadCheckpoint(const std::string& filename);
    
    /// Early Stop Check
    bool shouldEarlyStop() const;
private:
    std::shared_ptr<ModelInterface> model_;
    std::shared_ptr<SearchStrategy> strategy_;
    TunerConfig config_;
    std::random_device rd_;
    std::mt19937 rng_;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime_;
    int iterationsSinceImprovement_ = 0;
    double bestScore_ = -std::numeric_limits<double>::infinity();
};

}



#endif // HYPERTUNE_TUNER_HPP
