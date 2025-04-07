//
//  randomForestAdapter.cpp
//  HyperTune
//
//  Created by Abhijit Shanbhag on 07/04/25.
//


#include "../include/modelAdapters.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace hypertune {

/// Implementation of the DecisionTree predict method
double RandomForestAdapter::DecisionTree::predict(const std::vector<double>& features) const {
    /// This is a simplified implementation for demonstration purposes
    /// In a real implementation, this would traverse the tree structure
    
    size_t node_idx = 0;
    while (node_idx < featureIndices.size() && featureIndices[node_idx] >= 0) {
        int feature_idx = featureIndices[node_idx];
        double threshold = thresholds[node_idx];
        
        if (features[feature_idx] <= threshold) {
            /// Go left
            node_idx = 2 * node_idx + 1;
        } else {
            /// Go right
            node_idx = 2 * node_idx + 2;
        }
        
        /// Check if we've reached a leaf or gone out of bounds
        if (node_idx >= featureIndices.size()) {
            break;
        }
    }
    
    /// Find the appropriate leaf value
    return leafValues[node_idx % leafValues.size()];
}

/// RandomForestAdapter implementation
RandomForestAdapter::RandomForestAdapter()
    : DataMatrixModelAdapter(),
      rng_(std::random_device()()),
      nEstimators_(100),
      maxDepth_(10),
      minSamplesSplit_(2),
      maxFeatures_(-1),  /// -1 means sqrt(n_features)
      criterion_("mse") {}

std::string RandomForestAdapter::toString() const {
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
    /// Set default values
    nEstimators_ = 100;
    maxDepth_ = 10;
    minSamplesSplit_ = 2;
    maxFeatures_ = -1;  /// -1 means sqrt(n_features)
    criterion_ = "mse";
    
    /// Apply custom hyperparameters if provided
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
    
    /// Get dimensions
    const size_t n_samples = xTrain_.size();
    const size_t n_features = xTrain_[0].size();
    
    /// Determine max_features if needed
    int actual_max_features = maxFeatures_;
    if (actual_max_features <= 0) {
        /// Use sqrt(n_features) as the default
        actual_max_features = static_cast<int>(std::sqrt(n_features));
    }
    actual_max_features = std::min(actual_max_features, static_cast<int>(n_features));
    
    /// Initialize the forest
    trees_.clear();
    trees_.resize(nEstimators_);
    
    /// Train each tree
    #pragma omp parallel for
    for (int i = 0; i < nEstimators_; ++i) {
        /// Create a seed for this tree
        unsigned int tree_seed = std::random_device()() + i;
        std::mt19937 tree_rng(tree_seed);
        
        /// Bootstrap sample indices (sample with replacement)
        std::vector<size_t> bootstrap_indices(n_samples);
        std::uniform_int_distribution<size_t> sample_dist(0, n_samples - 1);
        for (size_t j = 0; j < n_samples; ++j) {
            bootstrap_indices[j] = sample_dist(tree_rng);
        }
        
        /// Simple implementation creates a mock decision tree
        /// In a real implementation, you would build the tree recursively
        DecisionTree& tree = trees_[i];
        
        /// Set up a simple tree structure (for demonstration)
        const int treeSize = std::min(15, (1 << maxDepth_) - 1); /// Simple binary tree size
        tree.featureIndices.resize(treeSize, -1); /// -1 indicates a leaf
        tree.thresholds.resize(treeSize, 0.0);
        tree.leafValues.resize(treeSize + 1, 0.0); /// +1 for extra leaf nodes
        
        /// Create a simple decision tree (this is highly simplified)
        for (int node_idx = 0; node_idx < treeSize / 2; ++node_idx) {
            /// For non-leaf nodes, select a random feature and threshold
            std::uniform_int_distribution<int> feature_dist(0, actual_max_features - 1);
            int feature_idx = feature_dist(tree_rng);
            
            /// Find min/max values for this feature in the bootstrap sample
            double min_val = std::numeric_limits<double>::max();
            double max_val = std::numeric_limits<double>::lowest();
            for (size_t idx : bootstrap_indices) {
                min_val = std::min(min_val, xTrain_[idx][feature_idx]);
                max_val = std::max(max_val, xTrain_[idx][feature_idx]);
            }
            
            /// Choose a random threshold between min and max
            std::uniform_real_distribution<double> threshold_dist(min_val, max_val);
            double threshold = threshold_dist(tree_rng);
            
            tree.featureIndices[node_idx] = feature_idx;
            tree.thresholds[node_idx] = threshold;
        }
        
        /// Set leaf values (simplified approach)
        for (int leaf_idx = treeSize / 2; leaf_idx < treeSize; ++leaf_idx) {
            /// In a real implementation, this would be the average of y values in this leaf
            std::uniform_real_distribution<double> leaf_dist(
                *std::min_element(yTrain_.begin(), yTrain_.end()),
                *std::max_element(yTrain_.begin(), yTrain_.end())
            );
            tree.leafValues[leaf_idx] = leaf_dist(tree_rng);
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
    
    /// Make predictions by averaging all trees
    for (size_t i = 0; i < n_samples; ++i) {
        double sum_pred = 0.0;
        for (const auto& tree : trees_) {
            sum_pred += tree.predict(xTest_[i]);
        }
        yPred_[i] = sum_pred / trees_.size();
    }
    
    /// Calculate R^2 score (higher is better)
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
    
    /// Return R^2 score as our model performance metric
    return r2;
}

} /// namespace hypertune
