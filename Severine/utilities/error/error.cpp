//
//  error.cpp
//  Severine
//
//  Created by Abhijit Shanbhag on 17/08/25.
//

#include "error.h"
#include "iostream"

std::string OperationStatusHandler::errorToString(OperationStatusCode error) {
    switch (error) {
        case OperationStatusCode::Success:
            return "Success";
            
        // GPU Hardware Errors
        case OperationStatusCode::GPUNotDetected:
            return "GPU hardware not detected";
        case OperationStatusCode::GPUNotAccessible:
            return "GPU detected but not accessible";
        case OperationStatusCode::GPUInitializationFailed:
            return "GPU device initialization failed";
        case OperationStatusCode::GPUMemoryAllocationFailed:
            return "GPU memory allocation failed";
        case OperationStatusCode::GPUInsufficientCapabilities:
            return "GPU compute capabilities insufficient";
            
        // Metal Framework Errors
        case OperationStatusCode::MetalNotAvailable:
            return "Metal framework not available";
        case OperationStatusCode::MetalDeviceCreationFailed:
            return "Metal device creation failed";
        case OperationStatusCode::MetalCommandQueueFailed:
            return "Metal command queue creation failed";
        case OperationStatusCode::MetalBufferCreationFailed:
            return "Metal buffer creation failed";
        case OperationStatusCode::MetalPipelineCreationFailed:
            return "Metal compute pipeline creation failed";
            
        // MLX Framework Errors
        case OperationStatusCode::MLXInitializationFailed:
            return "MLX framework initialization failed";
        case OperationStatusCode::MLXDeviceSetFailed:
            return "MLX device setting failed";
        case OperationStatusCode::MLXArrayCreationFailed:
            return "MLX array creation failed";
        case OperationStatusCode::MLXComputationFailed:
            return "MLX computation failed";
        case OperationStatusCode::MLXMemoryManagementFailed:
            return "MLX memory management failed";
        case OperationStatusCode::MLXUnsupportedDeviceType:
            return "MLX device type unsupported";
            
        // System Errors
        case OperationStatusCode::InsufficientSystemMemory:
            return "Insufficient system memory";
        case OperationStatusCode::UnsupportedOperatingSystem:
            return "Operating system not supported";
        case OperationStatusCode::UnsupportedArchitecture:
            return "Architecture not supported";
        case OperationStatusCode::InvalidSystemConfiguration:
            return "System configuration invalid";
        case OperationStatusCode::PermissionDenied:
            return "Permission denied for hardware access";
            
        // Configuration Errors
        case OperationStatusCode::InvalidDeviceConfiguration:
            return "Invalid device configuration";
        case OperationStatusCode::InvalidMemoryConfiguration:
            return "Invalid memory configuration";
        case OperationStatusCode::InvalidComputeConfiguration:
            return "Invalid compute configuration";
        case OperationStatusCode::ConfigurationFileNotFound:
            return "Configuration file not found";
        case OperationStatusCode::ConfigurationParsingFailed:
            return "Configuration parsing failed";
            
        // Runtime Errors
        case OperationStatusCode::DeviceConnectionLost:
            return "Device connection lost during operation";
        case OperationStatusCode::OperationTimeout:
            return "Operation timeout exceeded";
        case OperationStatusCode::ResourceBusy:
            return "Resource busy or locked";
        case OperationStatusCode::MemoryLimitExceeded:
            return "Memory limit exceeded";
        case OperationStatusCode::ComputeOperationFailed:
            return "Compute operation failed";
            
        // Validation Errors
        case OperationStatusCode::InvalidParameters:
            return "Input parameters invalid";
        case OperationStatusCode::UnsupportedDataFormat:
            return "Data format unsupported";
        case OperationStatusCode::InvalidArrayDimensions:
            return "Array dimensions invalid";
        case OperationStatusCode::UnsupportedDataType:
            return "Data type unsupported";
        case OperationStatusCode::UnsupportedOperation:
            return "Operation not supported on current device";
            
        // Unknown Errors
        case OperationStatusCode::UnknownError:
            return "Unknown error occurred";
        case OperationStatusCode::UnexpectedException:
            return "Unexpected exception caught";
        case OperationStatusCode::InternalError:
            return "Internal framework error";
            
        default:
            return "Undefined error code";
    }
}

bool OperationStatusHandler::isSuccess(OperationStatusCode error) {
    return error == OperationStatusCode::Success;
}

bool OperationStatusHandler::isRecoverable(OperationStatusCode error) {
    switch (error) {
        case OperationStatusCode::Success:
        case OperationStatusCode::ResourceBusy:
        case OperationStatusCode::OperationTimeout:
        case OperationStatusCode::MLXMemoryManagementFailed:
        case OperationStatusCode::GPUMemoryAllocationFailed:
            return true;
        default:
            return false;
    }
}

std::string OperationStatusHandler::getErrorCategory(OperationStatusCode error) {
    int errorCode = static_cast<int>(error);
    
    if (errorCode == 0) return "Success";
    if (errorCode >= 100 && errorCode < 200) return "GPU Hardware";
    if (errorCode >= 200 && errorCode < 300) return "Metal Framework";
    if (errorCode >= 300 && errorCode < 400) return "MLX Framework";
    if (errorCode >= 400 && errorCode < 500) return "System";
    if (errorCode >= 500 && errorCode < 600) return "Configuration";
    if (errorCode >= 600 && errorCode < 700) return "Runtime";
    if (errorCode >= 700 && errorCode < 800) return "Validation";
    if (errorCode >= 900 && errorCode < 1000) return "Unknown";
    
    return "Undefined";
}

void OperationStatusHandler::printError(OperationStatusCode error) {
    std::cout << "Error: " << errorToString(error) << std::endl;
    std::cout << "Category: " << getErrorCategory(error) << std::endl;
    std::cout << "Recoverable: " << (isRecoverable(error) ? "Yes" : "No") << std::endl;
}
