//
//  hyperparameter.cpp
//  HyperTune
//
//  Created by Abhijit Shanbhag on 26/03/25.
//

#include "../include/hyperparameter.hpp"
#include <sstream>
#include <algorithm>
#include <cmath>

namespace hypertune {

/// `Hyperparameter` base class implementation
Hyperparameter::Hyperparameter(const std::string& name, Type type)
    : name_(name), type_(type) {}

const std::string& Hyperparameter::getName() const {
    return name_;
}

Hyperparameter::Type Hyperparameter::getType() const {
    return type_;
}





/// `CategoricalHyperparameter` implementation
CategoricalHyperparameter::CategoricalHyperparameter(
    const std::string& name, const std::vector<std::string>& values)
    : Hyperparameter(name, Type::CATEGORICAL), values_(values) {}

std::variant<int, float, bool, std::string> CategoricalHyperparameter::sample(std::mt19937& rng) const {
    std::uniform_int_distribution<unsigned long> dist(0, values_.size() - 1);
    return values_[dist(rng)];
}

const std::vector<std::string>& CategoricalHyperparameter::getValues() const {
    return values_;
}

std::string CategoricalHyperparameter::toString() const {
    std::stringstream ss;
    ss << name_ << " (categorical): [";
    for (size_t i = 0; i < values_.size(); ++i) {
        ss << values_[i];
        if (i < values_.size() - 1) ss << ", ";
    }
    ss << "]";
    return ss.str();
}





/// `IntegerHyperparameter` implementation
IntegerHyperparameter::IntegerHyperparameter(
    const std::string& name, int lower, int upper)
    : Hyperparameter(name, Type::INTEGER), lower_(lower), upper_(upper) {}

std::variant<int, float, bool, std::string> IntegerHyperparameter::sample(std::mt19937& rng) const {
    std::uniform_int_distribution<int> dist(lower_, upper_);
    return dist(rng);
}

int IntegerHyperparameter::getLower() const {
    return lower_;
}

int IntegerHyperparameter::getUpper() const {
    return upper_;
}

std::string IntegerHyperparameter::toString() const {
    std::stringstream ss;
    ss << name_ << " (integer): [" << lower_ << ", " << upper_ << "]";
    return ss.str();
}





/// `FloatHyperparameter` implementation
FloatHyperparameter::FloatHyperparameter(
    const std::string& name, float lower, float upper, Distribution dist)
    : Hyperparameter(name, Type::FLOAT), lower_(lower), upper_(upper), distribution_(dist) {}

std::variant<int, float, bool, std::string> FloatHyperparameter::sample(std::mt19937& rng) const {
    if (distribution_ == Distribution::UNIFORM) {
        std::uniform_real_distribution<float> dist(lower_, upper_);
        return dist(rng);
    } else { // LOG_UNIFORM
        std::uniform_real_distribution<float> dist(std::log(lower_), std::log(upper_));
        return std::exp(dist(rng));
    }
}

float FloatHyperparameter::getLower() const {
    return lower_;
}

float FloatHyperparameter::getUpper() const {
    return upper_;
}

std::string FloatHyperparameter::toString() const {
    std::stringstream ss;
    ss << name_ << " (float): [" << lower_ << ", " << upper_ << "]";
    if (distribution_ == Distribution::LOG_UNIFORM) {
        ss << " (log-uniform)";
    }
    return ss.str();
}





/// `BooleanHyperparameter` implementation
BooleanHyperparameter::BooleanHyperparameter(const std::string& name)
    : Hyperparameter(name, Type::BOOLEAN) {}

std::variant<int, float, bool, std::string> BooleanHyperparameter::sample(std::mt19937& rng) const {
    std::uniform_int_distribution<int> dist(0, 1);
    return static_cast<bool>(dist(rng));
}

std::string BooleanHyperparameter::toString() const {
    return name_ + " (boolean): [true, false]";
}





/// `SearchSpace` implementation
void SearchSpace::addHyperparameter(std::shared_ptr<Hyperparameter> param) {
    hyperparameters_[param->getName()] = param;
}

const HyperparameterMap& SearchSpace::getHyperparameters() const {
    return hyperparameters_;
}

std::unordered_map<std::string, std::variant<int, float, bool, std::string>>
SearchSpace::sampleConfiguration(std::mt19937& rng) const {
    std::unordered_map<std::string, std::variant<int, float, bool, std::string>> config;
    
    for (const auto& [name, param] : hyperparameters_) {
        config[name] = param->sample(rng);
    }
    
    return config;
}

} 
