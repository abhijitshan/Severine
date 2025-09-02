//
//  MLXMatrix.cpp
//  Severine
//
//  Created by Abhijit Shanbhag on 02/09/25.
//

#include "MLXMatrix.hpp"
namespace severine {

void MLXMatrix::conditionalEval() {
    // TODO: Check flag and eval if needed
    if (immediateEval_) {
        mlx::core::eval(data_);
        evaluatedState_=true;
    }
}

MLXMatrix::MLXMatrix(int rows, int cols, bool immediate)
: data_(mlx::core::zeros({rows, cols})), immediateEval_(immediate){
    conditionalEval();
}


MLXMatrix::MLXMatrix(const mlx::core::array& arr, bool immediate) : data_(arr), immediateEval_(immediate) {
    if (arr.shape().size() != 2) {
        throw std::runtime_error("MLXMatrix requires a 2D array, got " +
                                std::to_string(arr.shape().size()) + "D array");
    }
    conditionalEval();
}

int MLXMatrix::rows() const {
    return data_.shape()[0];
}

int MLXMatrix::cols() const {
    return data_.shape()[1];
    
}

float MLXMatrix::operator()(int row, int col) {
    mlx::core::eval(data_);
    
    // Use strides for correct indexing
    auto strides = data_.strides();
    size_t offset = row * strides[0] + col * strides[1];
    
    auto ptr = data_.data<float>();
    return ptr[offset];
}

void MLXMatrix::set(int row, int col, float value) {
    mlx::core::eval(data_);
    
    // Use strides for correct indexing
    auto strides = data_.strides();
    size_t offset = row * strides[0] + col * strides[1];
    
    auto ptr = data_.data<float>();
    ptr[offset] = value;
    
    evaluatedState_ = false;
    conditionalEval();
}

MLXMatrix MLXMatrix::multiply(const MLXMatrix& other) const {
    auto resultData = mlx::core::matmul(data_, other.data_);
    return MLXMatrix(resultData, immediateEval_);
}

MLXMatrix MLXMatrix::add(const MLXMatrix& other) const {
    auto result = mlx::core::add(data_, other.data_);
    return MLXMatrix(result, immediateEval_);
}

MLXMatrix MLXMatrix::transpose() const {
    auto result = mlx::core::transpose(data_);
    return MLXMatrix(result, immediateEval_);
    // TODO: Support custom axes
}

void MLXMatrix::evaluate() {
    mlx::core::eval(data_);
    evaluatedState_ = true;
}

bool MLXMatrix::isEvaluated() const {
    return evaluatedState_;
    // TODO: Consider smarter tracking vs always true after eval()
}

void MLXMatrix::setEvaluationMode(bool immediate) {
    immediateEval_ = immediate;
    if (immediateEval_) {
        evaluate();
    }
}

void MLXMatrix::print() const {
    if (data_.flags().contiguous) {
        auto ptr = data_.data<float>();
        for (int currentRow = 0; currentRow < rows(); currentRow++) {
            for (int currentCol = 0; currentCol < cols(); currentCol++) {
                std::cout << ptr[currentRow * cols() + currentCol] << " ";
            }
            std::cout << "\n";
        }
    } else {
        auto dense = mlx::core::copy(data_);
        mlx::core::eval(dense);
        auto ptr = dense.data<float>();
        for (int currentRow = 0; currentRow < rows(); currentRow++) {
            for (int currentCol = 0; currentCol < cols(); currentCol++) {
                std::cout << ptr[currentRow * cols() + currentCol] << " ";
            }
            std::cout << "\n";
        }
    }
}
}
