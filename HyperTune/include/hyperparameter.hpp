//
//  hyperparameter.hpp
//  HyperTune
//
//  Created by Abhijit Shanbhag on 26/03/25.
//

#ifndef HYPERTUNE_HYPERPARAMETER_HPP
#define HYPERTUNE_HYPERPARAMETER_HPP

#include <string>
#include <vector>
#include <random>
#include <memory>
#include <variant>
#include <unordered_map>

namespace hypertune {

// Forward declaration
class Hyperparameter;

/// Read about `shared_ptr` later. It is apparently a thread safe. Read more later about it
using HyperparameterMap = std::unordered_map<std::string, std::shared_ptr<Hyperparameter>>;

/// Base Class for All Hyperparameters
class Hyperparameter {
public:
    enum class Type {
        CATEGORICAL,
        BOOLEAN,
        INTEGER,
        FLOAT
    };
    
    Hyperparameter(const std::string& name, Type type);
    
    /// For **@AnujPai**
    /// If I attach `const` it means it is not going to you know, modify internal states of the hyperparameter tuning mechanism
    /// For now use the default destructors
    virtual ~Hyperparameter() = default;
    
    /// Get the name of this hyperparameter
    const std::string& getName() const;
    
    /// Get the type of this hyperparameter
    Type getType() const;
    
    /// Sample a value from the hyperparameter's hyperspace using Mersenne Twister
    /// Now, this one basically, is to sample from a given hyperparameter space.
    /// Using `std::variant` that can return any of those Enumerated Types as mentioned above
    virtual std::variant<int, float, bool, std::string> sample(std::mt19937& rng) const = 0;
    
    /// Convert the hyperparameter to a string representation
    virtual std::string toString() const = 0;
    
protected:
    std::string name_;
    Type type_;
};

/// Categorical Hyperparameters
class CategoricalHyperparameter : public Hyperparameter {
public:
    /// Constructor for the `CategoricalHyperparameter` Class
    /// - Parameters:
    ///   - name: A string value that can be used to name the categorical parameter name
    ///   - values: A string vector collection of all possible values that can be associated with the value
    CategoricalHyperparameter(const std::string& name, const std::vector<std::string>& values);
    
    /// Generate a random value from the sample space
    /// - Parameter rng: The random pseudo-random generated value
    /// - Note
    /// Type Flexibility: The `std::variant` return type allows the method to work within a unified interface where different hyperparameter types might return different value types.
    std::variant<int, float, bool, std::string> sample(std::mt19937& rng) const override;
    
    const std::vector<std::string>& getValues() const;
    std::string toString() const override;
    
private:
    std::vector<std::string> values_;
};

/// IntegerHyperparameter
class IntegerHyperparameter : public Hyperparameter {
public:
    IntegerHyperparameter(const std::string& name, int lower, int upper);
    
    /// Sample an integer value
    std::variant<int, float, bool, std::string> sample(std::mt19937& rng) const override;
    
    int getLower() const;
    int getUpper() const;
    
    std::string toString() const override;

private:
    /// The lower and upper range
    int lower_;
    int upper_;
};

/// Float hyperparameter with a range
class FloatHyperparameter : public Hyperparameter {
public:
    enum class Distribution {
        UNIFORM,
        LOG_UNIFORM
    };

    FloatHyperparameter(const std::string& name, float lower, float upper,
                        Distribution dist = Distribution::UNIFORM);
    
    /// Sample a float value
    std::variant<int, float, bool, std::string> sample(std::mt19937& rng) const override;
    
    /// Get the lower and upper bounds
    float getLower() const;
    float getUpper() const;
    
    std::string toString() const override;

private:
    /// The lower and upper range, as well as the set distribution category
    float lower_;
    float upper_;
    Distribution distribution_;
};

/// Boolean hyperparameter
class BooleanHyperparameter : public Hyperparameter {
public:
    BooleanHyperparameter(const std::string& name);
    
    /// Sample a boolean value
    std::variant<int, float, bool, std::string> sample(std::mt19937& rng) const override;
    
    std::string toString() const override;
};

/// `SearchSpace` class to hold all hyperparameters
class SearchSpace {
public:
    SearchSpace() = default;
    
    /// Add a hyperparameter to the search space
    void addHyperparameter(std::shared_ptr<Hyperparameter> param);
    
    /// Get all hyperparameters in the search space
    const HyperparameterMap& getHyperparameters() const;
    
    /// Sample a complete configuration from the search space
    std::unordered_map<std::string, std::variant<int, float, bool, std::string>>
    sampleConfiguration(std::mt19937& rng) const;

private:
    HyperparameterMap hyperparameters_;
};

}

#endif
