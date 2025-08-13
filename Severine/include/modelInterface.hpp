#ifndef SEVERINE_MODEL_INTERFACE_HPP
#define SEVERINE_MODEL_INTERFACE_HPP
#include <string>
#include <variant>
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>
namespace severine {
using Config = std::unordered_map<std::string, std::variant<int, float, bool, std::string>>;
class ModelInterface {
public:
    virtual ~ModelInterface() = default;
    virtual void train() = 0;
    virtual double evaluate() = 0;
    virtual void configure(const Config& hyperparameters) = 0;
    virtual std::string toString() const = 0;
    virtual void setVerbose(bool verbose) = 0;
    virtual std::shared_ptr<ModelInterface> clone() const = 0;
    void threadSafeTrain() {
        std::lock_guard<std::mutex> lock(mutex_);
        train();
    }
    double threadSafeEvaluate() {
        std::lock_guard<std::mutex> lock(mutex_);
        return evaluate();
    }
    void threadSafeConfigure(const Config& hyperparameters) {
        std::lock_guard<std::mutex> lock(mutex_);
        configure(hyperparameters);
    }
protected:
    mutable std::mutex mutex_; 
};
std::unique_ptr<ModelInterface> createMLFmkModel(const std::string& modelType);
class MLFmkModelAdapter : public ModelInterface {
public:
    MLFmkModelAdapter() : verbose_(false) {}
    virtual ~MLFmkModelAdapter() = default;
    void train() override;
    double evaluate() override;
    void configure(const Config& hyperparameters) override;
    std::string toString() const override;
    std::shared_ptr<ModelInterface> clone() const override;
    void setVerbose(bool verbose) override {
        std::lock_guard<std::mutex> lock(mutex_);
        verbose_ = verbose;
    }
protected:
    virtual void applyHyperparameters(const Config& hyperparameters) = 0;
    virtual void trainModel() = 0;
    virtual double evaluateModel() = 0;
    virtual std::shared_ptr<MLFmkModelAdapter> cloneImpl() const = 0;
    Config config_;
    bool verbose_;
};
class LinearRegressionAdapter;
class RandomForestAdapter;
class SVMAdapter;
class NeuralNetworkAdapter;
}
#endif