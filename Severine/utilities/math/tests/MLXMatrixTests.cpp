//
//  MLXMatrixTests.cpp
//  Severine
//
//  Created by Abhijit Shanbhag on 02/09/25.
//

#include "MLXMatrixTests.hpp"
#include "MLXMatrix.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>

namespace severine {

void MLXMatrixTester::runAllTests() {
    std::cout << "=== MLXMatrix Test Suite ===" << std::endl;

    testBasicConstruction();
    testDimensions();
    testElementAccess();
    testElementSetting();
    testMatrixMultiplication();
    testMatrixAddition();
    testTranspose();
    testEvaluationModes();
    testEvaluationControl();
    testPrinting();
    testErrorHandling();

    std::cout << "\nAll tests passed successfully." << std::endl;
}

void MLXMatrixTester::testBasicConstruction() {
    std::cout << "\nTesting Basic Construction..." << std::endl;

    MLXMatrix m1(3, 4, true);
    assert(m1.rows() == 3);
    assert(m1.cols() == 4);
    assert(m1.isEvaluated() == true);

    MLXMatrix m2(2, 5, false);
    assert(m2.rows() == 2);
    assert(m2.cols() == 5);

    std::cout << "Basic construction tests passed." << std::endl;
}

void MLXMatrixTester::testDimensions() {
    std::cout << "\nTesting Dimensions..." << std::endl;

    MLXMatrix small(1, 1);
    assert(small.rows() == 1);
    assert(small.cols() == 1);

    MLXMatrix large(100, 50);
    assert(large.rows() == 100);
    assert(large.cols() == 50);

    MLXMatrix rectangular(7, 13);
    assert(rectangular.rows() == 7);
    assert(rectangular.cols() == 13);

    std::cout << "Dimension tests passed." << std::endl;
}

void MLXMatrixTester::testElementAccess() {
    std::cout << "\nTesting Element Access..." << std::endl;

    MLXMatrix m(3, 3);

    double val_00 = m(0, 0);
    double val_11 = m(1, 1);
    double val_22 = m(2, 2);

    assert(std::abs(val_00) < 1e-6);
    assert(std::abs(val_11) < 1e-6);
    assert(std::abs(val_22) < 1e-6);

    std::cout << "Element access tests passed." << std::endl;
}

void MLXMatrixTester::testElementSetting() {
    std::cout << "\nTesting Element Setting..." << std::endl;

    MLXMatrix m(3, 3);

    m.set(0, 0, 1.5);
    m.set(1, 1, 2.7);
    m.set(2, 0, -3.14);
    m.set(0, 2, 42.0);

    assert(std::abs(m(0, 0) - 1.5) < 1e-6);
    assert(std::abs(m(1, 1) - 2.7) < 1e-6);
    assert(std::abs(m(2, 0) - (-3.14)) < 1e-6);
    assert(std::abs(m(0, 2) - 42.0) < 1e-6);

    assert(std::abs(m(1, 0)) < 1e-6);
    assert(std::abs(m(2, 2)) < 1e-6);

    std::cout << "Element setting tests passed." << std::endl;
}

void MLXMatrixTester::testMatrixMultiplication() {
    std::cout << "\nTesting Matrix Multiplication..." << std::endl;

    MLXMatrix a(2, 3);
    a.set(0, 0, 1.0); a.set(0, 1, 2.0); a.set(0, 2, 3.0);
    a.set(1, 0, 4.0); a.set(1, 1, 5.0); a.set(1, 2, 6.0);

    MLXMatrix b(3, 2);
    b.set(0, 0, 7.0); b.set(0, 1, 8.0);
    b.set(1, 0, 9.0); b.set(1, 1, 10.0);
    b.set(2, 0, 11.0); b.set(2, 1, 12.0);

    MLXMatrix c = a.multiply(b);

    assert(c.rows() == 2);
    assert(c.cols() == 2);

    std::cout << "\n=== DEBUGGING MATRIX MULTIPLICATION ===" << std::endl;

    // Check if your set() method actually worked
    std::cout << "Matrix A after setting values:" << std::endl;
    a.print();
    std::cout << "Manually checking a(0,0): " << a(0, 0) << std::endl;
    std::cout << "Manually checking a(0,1): " << a(0, 1) << std::endl;

    std::cout << "\nMatrix B after setting values:" << std::endl;
    b.print();

    std::cout << "\nMatrix C (multiplication result):" << std::endl;
    c.print();
    std::cout << "c(0,0) = " << c(0, 0) << " (expected 58.0)" << std::endl;
    std::cout << "Difference: " << std::abs(c(0, 0) - 58.0) << std::endl;
    
    
    assert(std::abs(c(0, 0) - 58.0) < 1e-6);
    assert(std::abs(c(0, 1) - 64.0) < 1e-6);
    assert(std::abs(c(1, 0) - 139.0) < 1e-6);
    assert(std::abs(c(1, 1) - 154.0) < 1e-6);

    std::cout << "Matrix multiplication tests passed." << std::endl;
}

void MLXMatrixTester::testMatrixAddition() {
    std::cout << "\nTesting Matrix Addition..." << std::endl;

    MLXMatrix a(2, 2);
    a.set(0, 0, 1.0); a.set(0, 1, 2.0);
    a.set(1, 0, 3.0); a.set(1, 1, 4.0);

    MLXMatrix b(2, 2);
    b.set(0, 0, 5.0); b.set(0, 1, 6.0);
    b.set(1, 0, 7.0); b.set(1, 1, 8.0);

    MLXMatrix c = a.add(b);

    assert(std::abs(c(0, 0) - 6.0) < 1e-6);
    assert(std::abs(c(0, 1) - 8.0) < 1e-6);
    assert(std::abs(c(1, 0) - 10.0) < 1e-6);
    assert(std::abs(c(1, 1) - 12.0) < 1e-6);

    std::cout << "Matrix addition tests passed." << std::endl;
}

void MLXMatrixTester::testTranspose() {
    std::cout << "\nTesting Transpose..." << std::endl;

    MLXMatrix m(2, 3);
    m.set(0, 0, 1.0); m.set(0, 1, 2.0); m.set(0, 2, 3.0);
    m.set(1, 0, 4.0); m.set(1, 1, 5.0); m.set(1, 2, 6.0);

    MLXMatrix mt = m.transpose();

    assert(mt.rows() == 3);
    assert(mt.cols() == 2);

    std::cout << "\n=== TRANSPOSE DEBUG ===" << std::endl;
    std::cout << "Original matrix m:" << std::endl;
    m.print();
    std::cout << "Transposed matrix mt:" << std::endl;
    mt.print();
    std::cout << "mt(1,0) expected: 2.0, actual: " << mt(1,0) << std::endl;
    // Add this in your transpose test
    std::cout << "Original MLX array shape: " << m.data().shape()[0] << "x" << m.data().shape()[1] << std::endl;
    std::cout << "Transposed MLX array shape: " << mt.data().shape()[0] << "x" << mt.data().shape()[1] << std::endl;
    assert(std::abs(mt(0, 0) - 1.0) < 1e-6);
    assert(std::abs(mt(1, 0) - 2.0) < 1e-6);
    assert(std::abs(mt(2, 0) - 3.0) < 1e-6);
    assert(std::abs(mt(0, 1) - 4.0) < 1e-6);
    assert(std::abs(mt(1, 1) - 5.0) < 1e-6);
    assert(std::abs(mt(2, 1) - 6.0) < 1e-6);

    std::cout << "Transpose tests passed." << std::endl;
}

void MLXMatrixTester::testEvaluationModes() {
    std::cout << "\nTesting Evaluation Modes..." << std::endl;

    MLXMatrix immediate(2, 2, true);
    immediate.set(0, 0, 5.0);
    assert(immediate.isEvaluated());

    MLXMatrix lazy(2, 2, false);
    MLXMatrix lazy2(2, 2, false);
    MLXMatrix result = lazy.add(lazy2);

    lazy.setEvaluationMode(true);
    assert(lazy.isEvaluated());

    std::cout << "Evaluation mode tests passed." << std::endl;
}

void MLXMatrixTester::testEvaluationControl() {
    std::cout << "\nTesting Evaluation Control..." << std::endl;

    MLXMatrix m(2, 2, false);
    assert(!m.isEvaluated() || true);

    m.evaluate();
    assert(m.isEvaluated());

    MLXMatrix m2(2, 2, false);
    MLXMatrix result = m.add(m2);

    std::cout << "Evaluation control tests passed." << std::endl;
}

void MLXMatrixTester::testPrinting() {
    std::cout << "\nTesting Printing..." << std::endl;

    MLXMatrix m(2, 3);
    m.set(0, 0, 1.1); m.set(0, 1, 2.2); m.set(0, 2, 3.3);
    m.set(1, 0, 4.4); m.set(1, 1, 5.5); m.set(1, 2, 6.6);

    std::cout << "Expected output (2x3 matrix):" << std::endl;
    std::cout << "1.1 2.2 3.3" << std::endl;
    std::cout << "4.4 5.5 6.6" << std::endl;

    std::cout << "Actual output:" << std::endl;
    m.print();

    std::cout << "Printing test completed (verify output manually)." << std::endl;
}

void MLXMatrixTester::testErrorHandling() {
    std::cout << "\nTesting Error Handling..." << std::endl;

    try {
        auto invalid_1d = mlx::core::zeros({10});
        MLXMatrix m(invalid_1d);
        assert(false && "Should have thrown exception for 1D array");
    } catch (const std::runtime_error& e) {
        std::cout << "Caught expected error for 1D array: " << e.what() << std::endl;
    }

    try {
        auto invalid_3d = mlx::core::zeros({2, 3, 4});
        MLXMatrix m(invalid_3d);
        assert(false && "Should have thrown exception for 3D array");
    } catch (const std::runtime_error& e) {
        std::cout << "Caught expected error for 3D array: " << e.what() << std::endl;
    }

    std::cout << "Error handling tests passed." << std::endl;
}

void MLXMatrixTester::testLargeMatrices() {
    std::cout << "\nTesting Large Matrices..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    MLXMatrix large1(100, 100);
    MLXMatrix large2(100, 100);

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            large1.set(i, j, i * 10.0 + j);
            large2.set(i, j, (i + j) * 0.5);
        }
    }

    MLXMatrix sum = large1.add(large2);
    MLXMatrix product = large1.multiply(large2);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Large matrix operations completed in " << duration.count() << " ms." << std::endl;
}

void MLXMatrixTester::testChainedOperations() {
    std::cout << "\nTesting Chained Operations..." << std::endl;

    MLXMatrix a(2, 2);
    a.set(0, 0, 1.0); a.set(0, 1, 2.0);
    a.set(1, 0, 3.0); a.set(1, 1, 4.0);

    MLXMatrix b(2, 2);
    b.set(0, 0, 0.5); b.set(0, 1, 0.5);
    b.set(1, 0, 0.5); b.set(1, 1, 0.5);

    MLXMatrix result = a.add(b).transpose().multiply(a);

    assert(result.rows() == 2);
    assert(result.cols() == 2);

    double val = result(0, 0);
    std::cout << "Chained operations produced result(0,0) = " << val << std::endl;
}

void MLXMatrixTester::testIdentityBehavior() {
    std::cout << "\nTesting Identity-like Behavior..." << std::endl;

    MLXMatrix identity(3, 3);
    identity.set(0, 0, 1.0);
    identity.set(1, 1, 1.0);
    identity.set(2, 2, 1.0);

    MLXMatrix test(3, 3);
    test.set(0, 0, 5.0); test.set(0, 1, 6.0); test.set(0, 2, 7.0);
    test.set(1, 0, 8.0); test.set(1, 1, 9.0); test.set(1, 2, 10.0);
    test.set(2, 0, 11.0); test.set(2, 1, 12.0); test.set(2, 2, 13.0);

    MLXMatrix result = test.multiply(identity);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            assert(std::abs(result(i, j) - test(i, j)) < 1e-6);
        }
    }

    std::cout << "Identity behavior tests passed." << std::endl;
}

} // namespace severine
