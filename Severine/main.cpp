//
//  main.cpp
//  Severine
//
//  Changed by Abhijit Shanbhag on 17/08/25.
//
#include "iostream"
#include "mlx/mlx.h"
#include "utilities/error/error.h"
#include "utilities/metal/metalDeviceSet.h"
#include "utilities/math/MLXMatrix.hpp"
#include "utilities/math/tests/MLXMatrixTests.hpp"
int main() {
    try {
        severine::MLXMatrixTester::runAllTests();

        std::cout << "\n=== Performance Tests ===" << std::endl;
        severine::MLXMatrixTester::testLargeMatrices();
        severine::MLXMatrixTester::testChainedOperations();
        severine::MLXMatrixTester::testIdentityBehavior();

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
