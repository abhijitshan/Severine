#include "../include/tuner.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <thread>
namespace severine {
Tuner::Tuner(std::shared_ptr<ModelInterface> model, std::shared_ptr<SearchStrategy> strategy, const TunerConfig& config)
    : model_(model), strategy_(strategy), config_(config), rng_(rd_()) {
    // Removed omp_set_num_threads call
}
void Tuner::tune() {
    startTime_ = std::chrono::high_resolution_clock::now();
    if (!config_.checkpointFile.empty()) {
        loadCheckpoint(config_.checkpointFile);
    }
    switch (config_.parallelStrategy) {
        case ParallelizationStrategy::THREAD_POOL:
        case ParallelizationStrategy::ASYNC_TASKS:
            tuneSequentialOpenMP();
            break;
        case ParallelizationStrategy::NONE:
        default:
            tuneSequential();
            break;
    }
}
void Tuner::tuneSequential() {
    for (int index = 0; index < config_.maxIterations; index++) {
        if (config_.verbose) {
            std::cout << "Iteration " << (index + 1) << "/" << config_.maxIterations << "\n";
        }
        Config config;
        {
            std::lock_guard<std::mutex> lock(strategyMutex_);
            config = strategy_->nextConfiguration();
        }
        EvaluationResult result = evaluateConfiguration(config, index);
        {
            std::lock_guard<std::mutex> lock(strategyMutex_);
            strategy_->update(result);
        }
        updateBestScore(result.score);
        if (config_.verbose) {
            std::cout << " Score: " << std::fixed << std::setprecision(6) << result.score;
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime_).count();
            std::cout << "   |  Elapsed Time: " << elapsed << "s" << "\n";
        }
        if (!config_.checkpointFile.empty() && (index + 1) % config_.checkpointInteral == 0) {
            std::lock_guard<std::mutex> lock(checkpointMutex_);
            saveCheckpoint(config_.checkpointFile);
        }
        if (shouldEarlyStop()) {
            if (config_.verbose) {
                std::cout << "Early Stopping Triggered at Iteration " << index + 1 << "\n";
            }
            break;
        }
        if (config_.verbose) {
            const auto& best = getBestResult();
            std::cout << "\nBest Configuration Score: " << best.score << "\t Iteration: " << best.iteration << "\n";
            for (const auto& [name, value] : best.configuration) {
                std::cout << " " << name << ": ";
                if (std::holds_alternative<int>(value)) {
                    std::cout << std::get<int>(value);
                } else if (std::holds_alternative<float>(value)) {
                    std::cout << std::get<float>(value);
                } else if (std::holds_alternative<bool>(value)) {
                    std::cout << (std::get<bool>(value) ? "true" : "false");
                } else if (std::holds_alternative<std::string>(value)) {
                    std::cout << std::get<std::string>(value);
                }
                std::cout << std::endl;
            }
        }
    }
}
void Tuner::tuneSequentialOpenMP() {
    int numIterations = config_.maxIterations;
    int completedIterations = 0;
    int currentIteration = 0;
    if (config_.verbose) {
        std::cout << "Starting parallel tuning with " << config_.numThreads << " threads\n";
    }
    std::vector<Config> configBatch;
    std::vector<EvaluationResult> resultBatch;
    std::vector<int> iterationIndices;
    {
        std::lock_guard<std::mutex> lock(strategyMutex_);
        configBatch.clear();
        iterationIndices.clear();
        int batchSize = std::min(config_.numThreads, numIterations);
        configBatch.resize(batchSize);
        iterationIndices.resize(batchSize);
        for (int i = 0; i < batchSize; i++) {
            configBatch[i] = strategy_->nextConfiguration();
            iterationIndices[i] = currentIteration++;
        }
    }
    while (completedIterations < numIterations) {
        int batchSize = configBatch.size();
        resultBatch.resize(batchSize);
        if (config_.verbose) {
            std::cout << "Processing batch of " << batchSize << " configurations\n";
        }
        // Serial loop replacing OpenMP parallel for
        for (int i = 0; i < batchSize; i++) {
            if (config_.verbose) {
                std::cout << "Thread 0 evaluating configuration " << (iterationIndices[i] + 1) << "\n";
            }
            try {
                std::shared_ptr<ModelInterface> modelClone = model_->clone();
                resultBatch[i] = evaluateConfiguration(configBatch[i], iterationIndices[i]);
                if (config_.verbose) {
                    std::cout << "Thread 0 completed evaluation " << (iterationIndices[i] + 1)
                              << " with score: " << resultBatch[i].score << "\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "Exception in thread 0 evaluating configuration " << (iterationIndices[i] + 1)
                          << ": " << e.what() << std::endl;
                resultBatch[i] = EvaluationResult{
                    configBatch[i],
                    -std::numeric_limits<double>::infinity(),
                    iterationIndices[i]
                };
            }
        }
        {
            std::lock_guard<std::mutex> lock(strategyMutex_);
            for (const auto& result : resultBatch) {
                strategy_->update(result);
                updateBestScore(result.score);
                completedIterations++;
                if (config_.verbose) {
                    std::cout << "Updated strategy with result from iteration "
                              << (result.iteration + 1) << ", score: " << result.score << "\n";
                }
            }
            if (!config_.checkpointFile.empty() && completedIterations % config_.checkpointInteral == 0) {
                saveCheckpoint(config_.checkpointFile);
            }
            if (shouldEarlyStop()) {
                if (config_.verbose) {
                    std::cout << "Early Stopping Triggered after " << completedIterations << " iterations\n";
                }
                break;
            }
            int remainingIterations = numIterations - currentIteration;
            if (remainingIterations > 0) {
                int nextBatchSize = std::min(config_.numThreads, remainingIterations);
                configBatch.resize(nextBatchSize);
                iterationIndices.resize(nextBatchSize);
                for (int i = 0; i < nextBatchSize; i++) {
                    configBatch[i] = strategy_->nextConfiguration();
                    iterationIndices[i] = currentIteration++;
                }
            } else {
                configBatch.clear();
                iterationIndices.clear();
            }
        }
        if (config_.verbose) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime_).count();
            std::cout << "Progress: " << completedIterations << "/" << numIterations
                      << " | Elapsed Time: " << elapsed << "s" << std::endl;
            const auto& best = getBestResult();
            std::cout << "Current Best: Score=" << best.score
                      << " Iteration=" << best.iteration << std::endl;
        }
    }
}
EvaluationResult Tuner::evaluateConfiguration(const Config& config, int iterationIndex) {
    if (config_.verbose) {
        std::cout << "Evaluating configuration " << iterationIndex + 1 << std::endl;
    }
    try {
        std::shared_ptr<ModelInterface> modelClone;
        try {
            modelClone = model_->clone();
        } catch (const std::exception& e) {
            std::cerr << "Exception in model cloning: " << e.what() << std::endl;
            modelClone = model_;
        }
        try {
            if (config_.verbose) {
                std::cout << "Thread " << std::this_thread::get_id()
                        << " configuring model for iteration " << iterationIndex + 1 << std::endl;
            }
            modelClone->configure(config);
        } catch (const std::exception& e) {
            std::cerr << "Exception in configure: " << e.what() << std::endl;
            return EvaluationResult{config, -std::numeric_limits<double>::infinity(), iterationIndex};
        }
        try {
            if (config_.verbose) {
                std::cout << "Thread " << std::this_thread::get_id()
                        << " training model for iteration " << iterationIndex + 1 << std::endl;
            }
            modelClone->train();
        } catch (const std::exception& e) {
            std::cerr << "Exception in training: " << e.what() << std::endl;
            return EvaluationResult{config, -std::numeric_limits<double>::infinity(), iterationIndex};
        }
        double score = -std::numeric_limits<double>::infinity();
        try {
            if (config_.verbose) {
                std::cout << "Thread " << std::this_thread::get_id()
                        << " evaluating model for iteration " << iterationIndex + 1 << std::endl;
            }
            score = modelClone->evaluate();
        } catch (const std::exception& e) {
            std::cerr << "Exception in evaluation: " << e.what() << std::endl;
        }
        EvaluationResult result{config, score, iterationIndex};
        if (config_.verbose) {
            std::cout << "Thread " << std::this_thread::get_id()
                    << " completed evaluation " << iterationIndex + 1
                    << " with score: " << score << std::endl;
        }
        return result;
    } catch (const std::exception& e) {
        std::cerr << "Exception in evaluateConfiguration: " << e.what() << std::endl;
        return EvaluationResult{config, -std::numeric_limits<double>::infinity(), iterationIndex};
    }
}
std::shared_ptr<ModelInterface> Tuner::cloneModel() const {
    return model_->clone();
}
void Tuner::updateBestScore(double score) {
    if (score > bestScore_) {
        bestScore_ = score;
        iterationsSinceImprovement_ = 0;
    } else {
        iterationsSinceImprovement_++;
    }
}
const EvaluationResult& Tuner::getBestResult() const {
    std::lock_guard<std::mutex> lock(strategyMutex_);
    return strategy_->getBestResult();
}
const std::vector<EvaluationResult>& Tuner::getAllResults() const {
    std::lock_guard<std::mutex> lock(strategyMutex_);
    return strategy_->getAllResults();
}
bool Tuner::saveCheckpoint(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }
    return true;
}
bool Tuner::loadCheckpoint(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }
    return true;
}
bool Tuner::shouldEarlyStop() const {
    switch (config_.earlyStopStrategy) {
        case EarlyStoppingStrategy::NONE:
            return false;
        case EarlyStoppingStrategy::NO_IMPROVEMENT:
            return iterationsSinceImprovement_ >= config_.earlyStoppingPatience;
        case EarlyStoppingStrategy::THRESHOLD:
            return bestScore_ >= config_.earlyStoppingThreshold;
        default:
            return false;
    }
}
}  // namespace severine
