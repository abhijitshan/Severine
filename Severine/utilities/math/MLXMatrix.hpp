//
//  MLXMatrix.hpp
//  Severine
//
//  Created by Abhijit Shanbhag on 02/09/25.
//

#ifndef SEVERINE_MLX_MATRIX_HPP
#define SEVERINE_MLX_MATRIX_HPP

#include <mlx/mlx.h>
#include <stdexcept>
#include <iostream>

namespace severine {

/// A matrix class providing basic matrix operations backed by MLX array storage.
///
/// The `MLXMatrix` class encapsulates a 2D matrix of floating-point values using 
/// the MLX library's array backend for efficient data storage and manipulation.
/// It supports operations like multiplication, addition, and transposition.
///
/// The class supports two evaluation modes:
/// - Immediate evaluation: operations are computed and results stored immediately.
/// - Deferred evaluation: computations are delayed until explicitly evaluated via `evaluate()`.
///
/// This design allows integration of lazy evaluation strategies for performance optimization.
///
/// Typical usage involves constructing matrices, performing operations, and optionally 
/// evaluating results explicitly.
///
/// - Note: MLXMatrix relies on the MLX library's core array capabilities and exposes
/// its data via the `data()` accessor.
///
/// ### Example Usage
/// ```cpp
/// // Example: Constructing and multiplying two matrices
/// severine::MLXMatrix A(2, 3); // 2 rows, 3 columns
/// severine::MLXMatrix B(3, 2); // 3 rows, 2 columns
///
/// // Fill matrices with values
/// for (int i = 0; i < A.rows(); ++i) {
///     for (int j = 0; j < A.cols(); ++j) {
///         A.set(i, j, static_cast<float>(i + j));
///     }
/// }
///
/// for (int i = 0; i < B.rows(); ++i) {
///     for (int j = 0; j < B.cols(); ++j) {
///         B.set(i, j, static_cast<float>(i * j));
///     }
/// }
///
/// // Multiply matrices
/// severine::MLXMatrix C = A.multiply(B);
/// C.print();
/// ```
class MLXMatrix {
private:
    mlx::core::array data_;
    bool immediateEval_;
    mutable bool evaluatedState_ = false; 
    
    void conditionalEval();
    
public:
    /// Constructs a matrix with specified dimensions.
    ///
    /// Initializes the matrix with the given number of rows and columns. 
    /// All elements are initially uninitialized.
    ///
    /// - Parameters:
    ///   - rows: Number of rows in the matrix.
    ///   - cols: Number of columns in the matrix.
    ///   - immediate: If `true`, operations are evaluated immediately (default `true`).
    ///                If `false`, evaluation is deferred until explicitly invoked.
    ///
    /// @code
    /// severine::MLXMatrix mat(3, 4); // 3x4 matrix with immediate evaluation
    /// @endcode
    MLXMatrix(int rows, int cols, bool immediate = true);
    
    /// Constructs a matrix from an existing MLX array.
    ///
    /// Wraps an existing `mlx::core::array` as the underlying data store.
    ///
    /// - Parameters:
    ///   - arr: The MLX array to wrap. Must represent a 2D float array.
    ///   - immediate: If `true`, operations are evaluated immediately (default `true`).
    ///
    /// - Throws: std::invalid_argument if `arr` dimensions are inconsistent or invalid.
    ///
    /// @code
    /// mlx::core::array arr = ...; // some valid MLX array
    /// severine::MLXMatrix mat(arr, false); // construct with deferred evaluation
    /// @endcode
    MLXMatrix(const mlx::core::array& arr, bool immediate = true);
    
    /// Returns the number of rows of the matrix.
    ///
    /// - Returns: The integer count of rows.
    int rows() const;
    
    /// Returns the number of columns of the matrix.
    ///
    /// - Returns: The integer count of columns.
    int cols() const;
    
    /// Accesses the element at the specified row and column.
    ///
    /// - Parameters:
    ///   - row: Zero-based row index.
    ///   - column: Zero-based column index.
    ///
    /// - Returns: The floating-point value stored at the specified position.
    ///
    /// - Throws: std::out_of_range if indices are out of bounds.
    float operator()(int row, int column);
    
    /// Sets the element at the specified row and column to a given value.
    ///
    /// - Parameters:
    ///   - row: Zero-based row index.
    ///   - column: Zero-based column index.
    ///   - value: The floating-point value to assign.
    ///
    /// - Throws: std::out_of_range if indices are out of bounds.
    void set(int row, int column, float value);
    
    /// Returns a new matrix that is the product of this matrix and another.
    ///
    /// Performs matrix multiplication (`this * other`).
    ///
    /// - Parameters:
    ///   - other: The matrix to multiply with.
    ///
    /// - Returns: A new `MLXMatrix` representing the product.
    ///
    /// - Throws: std::invalid_argument if the matrices have incompatible dimensions.
    ///
    /// @code
    /// auto C = A.multiply(B);
    /// @endcode
    MLXMatrix multiply(const MLXMatrix& other) const;
    
    /// Returns a new matrix that is the sum of this matrix and another.
    ///
    /// Performs element-wise addition.
    ///
    /// - Parameters:
    ///   - other: The matrix to add.
    ///
    /// - Returns: A new `MLXMatrix` representing the sum.
    ///
    /// - Throws: std::invalid_argument if matrices have different dimensions.
    MLXMatrix add(const MLXMatrix& other) const;
    
    /// Returns a new matrix that is the transpose of this matrix.
    ///
    /// Transposes rows and columns.
    ///
    /// - Returns: A new `MLXMatrix` representing the transpose.
    MLXMatrix transpose() const;
    
    /// Forces evaluation of any deferred computations.
    ///
    /// If the matrix is in deferred evaluation mode and has pending operations,
    /// this method triggers their completion and stores results.
    void evaluate();
    
    /// Checks whether the matrix has been evaluated.
    ///
    /// - Returns: `true` if evaluated, `false` if pending deferred computations.
    bool isEvaluated() const;
    
    /// Sets the evaluation mode of the matrix.
    ///
    /// - Parameters:
    ///   - immediate: If `true`, operations are evaluated immediately;
    ///                if `false`, evaluation is deferred until `evaluate()` is called.
    void setEvaluationMode(bool immediate);
    
    /// Prints the matrix elements to standard output.
    ///
    /// Format is row-major, with elements separated by spaces.
    void print() const;
    
    /// Provides access to the underlying MLX array storage.
    ///
    /// - Returns: A const reference to the internal `mlx::core::array`.
    const mlx::core::array& data() const { return data_; }
};

} // namespace severine

#endif

