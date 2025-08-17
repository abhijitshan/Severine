//
//  metalDeviceSet.h
//  Severine
//
//  Created by Abhijit Shanbhag on 17/08/25.
//


#ifndef METAL_DEVICE_SET_H
#define METAL_DEVICE_SET_H

#include <string>
#include <mlx/mlx.h>
#include "../error/error.h"

/// A utility class for managing and detecting MLX GPU instances on Apple Silicon devices.
///
/// This class provides functionality to check GPU availability, query device status,
/// and manage device selection between CPU and GPU for MLX computations.
/// - Important: The framework supports Metal compatible GPUs only, preferably M1 (2020) or later
class MLXGPUInstance {
public:
    /// Default constructor
    MLXGPUInstance() = default;
    
    /// Default destructor
    ~MLXGPUInstance() = default;
    
    /// Performs comprehensive device checking and status reporting.
    ///
    /// Checks GPU availability, displays current device status, and attempts
    /// to set the appropriate device based on hardware capabilities.
    void checkDevice();
    
    /// Checks if GPU is available for MLX computations.
    ///
    /// - Returns: `true` if Metal/GPU support is available, `false` otherwise
    bool isGPUAvailable();
    
    /// Gets the current device as a string representation.
    ///
    /// - Returns: String representation of current device which is either `gpu` or `cpu`
    std::string getDeviceString();
    
    /// Prints current device information to console.
    ///
    /// Displays the current MLX device and GPU availability status.
    void printDeviceInfo();

    /// Performs basic GPU detection and device status check.
    ///
    /// Simple check mechanism that displays GPU presence and current device type
    /// without attempting device modification.
    void check();

    /// Attempts to set the default MLX device to GPU.
    ///
    /// Checks GPU availability and sets the default device to GPU if possible.
    /// Provides error handling for cases where GPU setting fails.
    ///
    /// - Returns: `OperationStatusCode` indicating success or specific failure reason
    OperationStatusCode setGPUDevice();
    
    /// Attempts to set the default MLX device to CPU.
    ///
    /// Forces MLX to use CPU for computations.
    ///
    /// - Returns: `OperationStatusCode` indicating success or specific failure reason
    OperationStatusCode setCPUDevice();
    
    /// Gets current MLX device type
    ///
    /// - Returns: `mlx::core::Device` representing current device
    mlx::core::Device getCurrentDevice();
    
    /// Validates current device configuration
    ///
    /// - Returns: `OperationStatusCode` indicating configuration validity
    OperationStatusCode validateDeviceConfiguration();

private:
    /// Internal method to verify Metal availability
    ///
    /// - Returns: `OperationStatusCode` indicating Metal framework status
    OperationStatusCode verifyMetalAvailability();
    
    /// Internal method to safely set device
    ///
    /// - Parameter device: Target device to set
    /// - Returns: `OperationStatusCode` indicating operation result
    OperationStatusCode safeSetDevice(mlx::core::Device device);
};

#endif // !METAL_DEVICE_SET_H
