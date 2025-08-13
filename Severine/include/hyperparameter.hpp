#ifndef SEVERINE_HYPERPARAMETER_HPP
#define SEVERINE_HYPERPARAMETER_HPP
#include <string>
#include <vector>
#include <random>
#include <memory>
#include <variant>
#include <unordered_map>
namespace severine {
class Hyperparameter;
using HyperparameterMap = std::unordered_map<std::string, std::shared_ptr<Hyperparameter>>;
class Hyperparameter {
public:
    enum class Type {
        CATEGORICAL,
        BOOLEAN,
        INTEGER,
        FLOAT
    };
    Hyperparameter(const std::string& name, Type type);
    virtual ~Hyperparameter() = default;
    const std::string& getName() const;
    Type getType() const;
    virtual std::variant<int, float, bool, std::string> sample(std::mt19937& rng) const = 0;
    virtual std::string toString() const = 0;
protected:
    std::string name_;
    Type type_;
};
class CategoricalHyperparameter : public Hyperparameter {
public:
    CategoricalHyperparameter(const std::string& name, const std::vector<std::string>& values);
    std::variant<int, float, bool, std::string> sample(std::mt19937& rng) const override;
    const std::vector<std::string>& getValues() const;
    std::string toString() const override;
private:
    std::vector<std::string> values_;
};
class IntegerHyperparameter : public Hyperparameter {
public:
    IntegerHyperparameter(const std::string& name, int lower, int upper);
    std::variant<int, float, bool, std::string> sample(std::mt19937& rng) const override;
    int getLower() const;
    int getUpper() const;
    std::string toString() const override;
private:
    int lower_;
    int upper_;
};
class FloatHyperparameter : public Hyperparameter {
public:
    enum class Distribution {
        UNIFORM,
        LOG_UNIFORM
    };
    FloatHyperparameter(const std::string& name, float lower, float upper,
                        Distribution dist = Distribution::UNIFORM);
    std::variant<int, float, bool, std::string> sample(std::mt19937& rng) const override;
    float getLower() const;
    float getUpper() const;
    std::string toString() const override;
    Distribution getDistribution() const {
        return distribution_;
    }
private:
    float lower_;
    float upper_;
    Distribution distribution_;
};
class BooleanHyperparameter : public Hyperparameter {
public:
    BooleanHyperparameter(const std::string& name);
    std::variant<int, float, bool, std::string> sample(std::mt19937& rng) const override;
    std::string toString() const override;
};
class SearchSpace {
public:
    SearchSpace() = default;
    void addHyperparameter(std::shared_ptr<Hyperparameter> param);
    const HyperparameterMap& getHyperparameters() const;
    std::unordered_map<std::string, std::variant<int, float, bool, std::string>>
    sampleConfiguration(std::mt19937& rng) const;
private:
    HyperparameterMap hyperparameters_;
};
}
#endif
