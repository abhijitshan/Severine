//
//  MLXMatrixTests.hpp
//  Severine
//
//  Created by Abhijit Shanbhag on 02/09/25.
//

#ifndef SEVERINE_MLX_MATRIX_TESTS_HPP
#define SEVERINE_MLX_MATRIX_TESTS_HPP

#include "../MLXMatrix.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>

namespace severine {

class MLXMatrixTester {
public:
    static void runAllTests();
    static void testLargeMatrices();
    static void testChainedOperations();
    static void testIdentityBehavior();
private:
    static void testBasicConstruction();
    static void testDimensions();
    static void testElementAccess();
    static void testElementSetting();
    static void testMatrixMultiplication();
    static void testMatrixAddition();
    static void testTranspose();
    static void testEvaluationModes();
    static void testEvaluationControl();
    static void testPrinting();
    static void testErrorHandling();
};

} // namespace severine

#endif // SEVERINE_MLX_MATRIX_TESTS_HPP
