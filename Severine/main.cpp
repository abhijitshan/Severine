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

int main() {
    try {
        MLXGPUInstance instance;
        
        // Basic GPU detection
        instance.check();
        
        // Attempt to set GPU device
        std::cout << "Attempting to set GPU device:" << std::endl;
        OperationStatusCode result = instance.setGPUDevice();
        
        // Handle result with error system
        if (!OperationStatusHandler::isSuccess(result)) {
            std::cout << "Device Setting Failed:" << std::endl;
            std::cout << "Error: " << OperationStatusHandler::errorToString(result) << std::endl;
            std::cout << "Category: " << OperationStatusHandler::getErrorCategory(result) << std::endl;
            
            if (OperationStatusHandler::isRecoverable(result)) {
                std::cout << "Attempting CPU fallback..." << std::endl;
                OperationStatusCode fallbackResult = instance.setCPUDevice();
                if (OperationStatusHandler::isSuccess(fallbackResult)) {
                    std::cout << "Successfully fell back to CPU" << std::endl;
                }
            }
        }
        
        std::cout << std::endl;
        
        // Display final device info
        instance.printDeviceInfo();
        
        // Validate configuration
        OperationStatusCode validationResult = instance.validateDeviceConfiguration();
        if (OperationStatusHandler::isSuccess(validationResult)) {
            std::cout << "Device configuration valid" << std::endl;
        } else {
            OperationStatusHandler::printError(validationResult);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        return static_cast<int>(OperationStatusCode::UnexpectedException);
    }
    
    return static_cast<int>(OperationStatusCode::Success);
}
