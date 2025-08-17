//
//  metalDeviceSet.cpp
//  Severine
//
//  Created by Abhijit Shanbhag on 17/08/25.
//

#include "metalDeviceSet.h"
#include <iostream>

void MLXGPUInstance::checkDevice() {
    std::cout << "MLX Device Test" << std::endl;
    std::cout << std::endl;
    bool gpuAvailable = mlx::core::metal::is_available();
    std::cout << "GPU Available: " << (gpuAvailable ? "Yes" : "No") << std::endl;
    auto currentDevice = mlx::core::default_device();
    std::cout << "Current Device: ";
    if (currentDevice == mlx::core::Device::gpu) {
        std::cout << "GPU" << std::endl;
    } else if (currentDevice == mlx::core::Device::cpu) {
        std::cout << "CPU" << std::endl;
    } else {
        // TODO: Add Check Mechanisms later
        std::cout << "Unknown" << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Device Status:" << std::endl;
    std::cout << "-------------" << std::endl;
    
    if (gpuAvailable) {
        try {
            mlx::core::set_default_device(mlx::core::Device::gpu);
            std::cout << "Successfully set device to GPU" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Failed to set GPU device: " << e.what() << std::endl;
        }
    } else {
        std::cout << "No GPU/Metal support" << std::endl;
        std::cout << "Running on CPU only" << std::endl;
        
        mlx::core::set_default_device(mlx::core::Device::cpu);
        std::cout << "Set device to CPU" << std::endl;
    }
    auto finalDevice = mlx::core::default_device();
    std::cout << std::endl;
    std::cout << "Final MLX Device: ";
    if (finalDevice == mlx::core::Device::gpu) {
        std::cout << "GPU" << std::endl;
    } else {
        std::cout << "CPU" << std::endl;
    }
}

bool MLXGPUInstance::isGPUAvailable() {
    return mlx::core::metal::is_available();
}

std::string MLXGPUInstance::getDeviceString() {
    auto device = mlx::core::default_device();
    return (device == mlx::core::Device::gpu) ? "gpu" : "cpu";
}

void MLXGPUInstance::printDeviceInfo() {
    std::cout << "MLX Device: " << getDeviceString() << std::endl;
    std::cout << "GPU Available: " << (isGPUAvailable() ? "true" : "false") << std::endl;
}

void MLXGPUInstance::check() {
    std::cout << "MLX GPU Instance Check" << std::endl;
    std::cout << std::endl;
    
    bool gpuExists = mlx::core::metal::is_available();
    auto currentDevice = mlx::core::default_device();
    
    std::cout << "GPU Detection:" << std::endl;
    std::cout << "GPU Present: " << (gpuExists ? "Yes" : "No") << std::endl;
    std::cout << "Current Device: " << ((currentDevice == mlx::core::Device::gpu) ? "GPU" : "CPU") << std::endl;
    std::cout << std::endl;
}

OperationStatusCode MLXGPUInstance::setGPUDevice() {
    OperationStatusCode metalCheck = verifyMetalAvailability();
    if (!OperationStatusHandler::isSuccess(metalCheck)) {
        std::cout << "GPU not available, cannot set GPU device" << std::endl;
        return metalCheck;
    }
    
    return safeSetDevice(mlx::core::Device::gpu);
}

OperationStatusCode MLXGPUInstance::setCPUDevice() {
    return safeSetDevice(mlx::core::Device::cpu);
}

mlx::core::Device MLXGPUInstance::getCurrentDevice() {
    return mlx::core::default_device();
}

OperationStatusCode MLXGPUInstance::validateDeviceConfiguration() {
    try {
        auto currentDevice = mlx::core::default_device();
        
        if (currentDevice == mlx::core::Device::gpu) {
            if (!mlx::core::metal::is_available()) {
                return OperationStatusCode::InvalidDeviceConfiguration;
            }
        }
        
        return OperationStatusCode::Success;
    } catch (const std::exception&) {
        return OperationStatusCode::InvalidDeviceConfiguration;
    }
}

OperationStatusCode MLXGPUInstance::verifyMetalAvailability() {
    if (!mlx::core::metal::is_available()) {
        return OperationStatusCode::MetalNotAvailable;
    }
    return OperationStatusCode::Success;
}

OperationStatusCode MLXGPUInstance::safeSetDevice(mlx::core::Device device) {
    try {
        mlx::core::set_default_device(device);
        auto actualDevice = mlx::core::default_device();
        
        if (actualDevice == device) {
            std::string deviceName = (device == mlx::core::Device::gpu) ? "GPU" : "CPU";
            std::cout << "Successfully set device to " << deviceName << std::endl;
            return OperationStatusCode::Success;
        } else {
            std::cout << "Failed to set device" << std::endl;
            return OperationStatusCode::MLXDeviceSetFailed;
        }
    } catch (const std::exception& e) {
        std::cout << "Error setting device: " << e.what() << std::endl;
        return OperationStatusCode::MLXDeviceSetFailed;
    }
}
