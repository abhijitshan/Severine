#include "../include/modelAdapters.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <omp.h>
namespace severine {
double RandomForestAdapter::DecisionTree::predict(const std::vector<double>& features) const {
    size_t node_idx = 0;
    while (node_idx < featureIndices.size() && featureIndices[node_idx] >= 0) {
        int feature_idx = featureIndices[node_idx];
        double threshold = thresholds[node_idx];
        if (features[feature_idx] <= threshold) {
            node_idx = 2 * node_idx + 1;
        } else {
            node_idx = 2 * node_idx + 2;
        }
        if (node_idx >= featureIndices.size()) {
            break;
        }
    }
    return leafValues[node_idx % leafValues.size()];
}
RandomForestAdapter::RandomForestAdapter()
    : DataMatrixModelAdapter(),
      rng_(std::random_device()()),
      nEstimators_(100),
      maxDepth_(10),
      minSamplesSplit_(2),
      maxFeatures_(-1),
      criterion_("mse") {}
std::string RandomForestAdapter::toString() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string result = "RandomForest Model with hyperparameters:\n";
    result += "  n_estimators: " + std::to_string(nEstimators_) + "\n";
    result += "  max_depth: " + std::to_string(maxDepth_) + "\n";
    result += "  min_samples_split: " + std::to_string(minSamplesSplit_) + "\n";
    result += "  max_features: " + std::to_string(maxFeatures_) + "\n";
    result += "  criterion: " + criterion_ + "\n";
    if (!trees_.empty()) {
        result += "Model trained with " + std::to_string(trees_.size()) + " trees\n";
    } else {
        result += "Model not yet trained\n";
    }
    return result;
}
void RandomForestAdapter::applyHyperparameters(const Config& hyperparameters) {
    nEstimators_ = 100;
    maxDepth_ = 10;
    minSamplesSplit_ = 2;
    maxFeatures_ = -1;
    criterion_ = "mse";
    for (const auto& [name, value] : hyperparameters) {
        if (name == "n_estimators" && std::holds_alternative<int>(value)) {
            nEstimators_ = std::get<int>(value);
        } else if (name == "max_depth" && std::holds_alternative<int>(value)) {
            maxDepth_ = std::get<int>(value);
        } else if (name == "min_samples_split" && std::holds_alternative<int>(value)) {
            minSamplesSplit_ = std::get<int>(value);
        } else if (name == "max_features" && std::holds_alternative<int>(value)) {
            maxFeatures_ = std::get<int>(value);
        } else if (name == "criterion" && std::holds_alternative<std::string>(value)) {
            criterion_ = std::get<std::string>(value);
        }
    }
    if (verbose_) {
        std::cout << "Random Forest configured with:\n"
                  << "  n_estimators: " << nEstimators_ << "\n"
                  << "  max_depth: " << maxDepth_ << "\n"
                  << "  min_samples_split: " << minSamplesSplit_ << "\n"
                  << "  max_features: " << maxFeatures_ << "\n"
                  << "  criterion: " << criterion_ << std::endl;
    }
}
void RandomForestAdapter::trainModel() {
    if (!dataLoaded_) {
        throw std::runtime_error("Cannot train model: no data loaded");
    }
    if (xTrain_.empty() || xTrain_[0].empty()) {
        throw std::runtime_error("Cannot train model: empty training data");
    }
    const size_t n_samples = xTrain_.size();
    const size_t n_features = xTrain_[0].size();
    int actual_max_features = maxFeatures_;
    if (actual_max_features <= 0) {
        actual_max_features = static_cast<int>(std::sqrt(n_features));
    }
    actual_max_features = std::min(actual_max_features, static_cast<int>(n_features));
    trees_.clear();
    trees_.resize(nEstimators_);
    #pragma omp parallel for default(none) \
        shared(trees_, n_samples, n_features, actual_max_features, xTrain_, yTrain_, std::cout) \
        firstprivate(maxDepth_, nEstimators_, verbose_)
    for (int i = 0; i < nEstimators_; ++i) {
        unsigned int tree_seed = omp_get_thread_num() * 1000 + i;
        std::mt19937 thread_rng(tree_seed);
        if (verbose_) {
            #pragma omp critical
            {
                std::cout << "Thread " << omp_get_thread_num()
                          << " training tree " << (i + 1) << " of " << nEstimators_ << std::endl;
            }
        }
        std::vector<size_t> bootstrap_indices(n_samples);
        std::uniform_int_distribution<size_t> sample_dist(0, n_samples - 1);
        for (size_t j = 0; j < n_samples; ++j) {
            bootstrap_indices[j] = sample_dist(thread_rng);
        }
        DecisionTree& tree = trees_[i];
        const int treeSize = std::min(15, (1 << maxDepth_) - 1);
        tree.featureIndices.resize(treeSize, -1);
        tree.thresholds.resize(treeSize, 0.0);
        tree.leafValues.resize(treeSize + 1, 0.0);
        std::vector<std::vector<double>> local_x_train;
        std::vector<double> local_y_train;
        local_x_train.reserve(bootstrap_indices.size());
        local_y_train.reserve(bootstrap_indices.size());
        for (size_t idx : bootstrap_indices) {
            local_x_train.push_back(xTrain_[idx]);
            local_y_train.push_back(yTrain_[idx]);
        }
        for (int node_idx = 0; node_idx < treeSize / 2; ++node_idx) {
            std::uniform_int_distribution<int> feature_dist(0, actual_max_features - 1);
            int feature_idx = feature_dist(thread_rng);
            double min_val = std::numeric_limits<double>::max();
            double max_val = std::numeric_limits<double>::lowest();
            for (const auto& features : local_x_train) {
                min_val = std::min(min_val, features[feature_idx]);
                max_val = std::max(max_val, features[feature_idx]);
            }
            std::uniform_real_distribution<double> threshold_dist(min_val, max_val);
            double threshold = threshold_dist(thread_rng);
            tree.featureIndices[node_idx] = feature_idx;
            tree.thresholds[node_idx] = threshold;
        }
        for (int leaf_idx = treeSize / 2; leaf_idx < treeSize; ++leaf_idx) {
            double min_y = *std::min_element(local_y_train.begin(), local_y_train.end());
            double max_y = *std::max_element(local_y_train.begin(), local_y_train.end());
            std::uniform_real_distribution<double> leaf_dist(min_y, max_y);
            tree.leafValues[leaf_idx] = leaf_dist(thread_rng);
        }
        if (verbose_) {
            #pragma omp critical
            {
                std::cout << "Thread " << omp_get_thread_num()
                          << " completed tree " << (i + 1) << std::endl;
            }
        }
    }
    if (verbose_) {
        std::cout << "Random Forest trained with " << nEstimators_
                  << " trees using " << n_features << " features and "
                  << n_samples << " samples" << std::endl;
    }
}
double RandomForestAdapter::evaluateModel() {
    if (xTest_.empty() || yTest_.empty()) {
        throw std::runtime_error("Cannot evaluate model: no test data");
    }
    const size_t n_samples = xTest_.size();
    yPred_.resize(n_samples, 0.0);
    #pragma omp parallel default(none) \
        shared(n_samples, yPred_, xTest_, trees_, std::cout) \
        firstprivate(verbose_)
    {
        std::vector<double> local_predictions(n_samples, 0.0);
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        int trees_per_thread = (trees_.size() + num_threads - 1) / num_threads;
        int start_tree = thread_id * trees_per_thread;
        int end_tree = std::min(start_tree + trees_per_thread, static_cast<int>(trees_.size()));
        if (verbose_) {
            #pragma omp critical
            {
                std::cout << "Thread " << thread_id << " processing trees "
                          << start_tree << " to " << end_tree - 1 << std::endl;
            }
        }
        for (int t = start_tree; t < end_tree; t++) {
            for (size_t i = 0; i < n_samples; ++i) {
                local_predictions[i] += trees_[t].predict(xTest_[i]);
            }
        }
        #pragma omp critical
        {
            for (size_t i = 0; i < n_samples; ++i) {
                yPred_[i] += local_predictions[i];
            }
        }
        #pragma omp barrier
        #pragma omp single
        {
            for (size_t i = 0; i < n_samples; ++i) {
                yPred_[i] /= trees_.size();
            }
        }
    }
    double r2 = calculateR2();
    if (verbose_) {
        double mse = calculateMSE();
        double rmse = calculateRMSE();
        double mae = calculateMAE();
        std::cout << "Random Forest evaluation metrics:\n"
                  << "  R^2: " << r2 << "\n"
                  << "  MSE: " << mse << "\n"
                  << "  RMSE: " << rmse << "\n"
                  << "  MAE: " << mae << std::endl;
    }
    return r2;
}
std::shared_ptr<MLFmkModelAdapter> RandomForestAdapter::cloneImpl() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cloneDataMatrixImpl();
}
std::shared_ptr<MLFmkModelAdapter> RandomForestAdapter::cloneDataMatrixImpl() const {
    auto clone = std::make_shared<RandomForestAdapter>();
    clone->config_ = config_;
    clone->verbose_ = verbose_;
    clone->nEstimators_ = nEstimators_;
    clone->maxDepth_ = maxDepth_;
    clone->minSamplesSplit_ = minSamplesSplit_;
    clone->maxFeatures_ = maxFeatures_;
    clone->criterion_ = criterion_;
    clone->trees_ = trees_;
    clone->rng_ = rng_;
    if (dataLoaded_) {
        clone->xTrain_ = xTrain_;
        clone->yTrain_ = yTrain_;
        clone->xTest_ = xTest_;
        clone->yTest_ = yTest_;
        clone->yPred_ = yPred_;
        clone->dataLoaded_ = dataLoaded_;
    }
    return clone;
}
} /// namespace severine