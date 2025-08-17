//
//  error.h
//  Severine
//
//  Created by Abhijit Shanbhag on 17/08/25.
//
#ifndef ERROR_H
#define ERROR_H

#include <string>

/// Error classification system for MLX GPU operations
///
/// Provides standardized error codes for GPU detection, device management,
/// and MLX framework operations to enable consistent error handling and debugging.
enum class OperationStatusCode {
    /// No error occurred - operation successful
    Success = 0,
    
    // GPU Hardware Errors (100-199)
    /// GPU hardware not detected on system
    GPUNotDetected = 100,
    /// GPU detected but not accessible
    GPUNotAccessible = 101,
    /// GPU device initialization failed
    GPUInitializationFailed = 102,
    /// GPU memory allocation failed
    GPUMemoryAllocationFailed = 103,
    /// GPU compute capabilities insufficient
    GPUInsufficientCapabilities = 104,
    
    // Metal Framework Errors (200-299)
    /// Metal framework not available on system
    MetalNotAvailable = 200,
    /// Metal device creation failed
    MetalDeviceCreationFailed = 201,
    /// Metal command queue creation failed
    MetalCommandQueueFailed = 202,
    /// Metal buffer creation failed
    MetalBufferCreationFailed = 203,
    /// Metal compute pipeline creation failed
    MetalPipelineCreationFailed = 204,
    
    // MLX Framework Errors (300-399)
    /// MLX framework initialization failed
    MLXInitializationFailed = 300,
    /// MLX device setting failed
    MLXDeviceSetFailed = 301,
    /// MLX array creation failed
    MLXArrayCreationFailed = 302,
    /// MLX computation failed
    MLXComputationFailed = 303,
    /// MLX memory management failed
    MLXMemoryManagementFailed = 304,
    /// MLX device type unsupported
    MLXUnsupportedDeviceType = 305,
    
    // System Errors (400-499)
    /// Insufficient system memory
    InsufficientSystemMemory = 400,
    /// Operating system not supported
    UnsupportedOperatingSystem = 401,
    /// Architecture not supported (non-Apple Silicon)
    UnsupportedArchitecture = 402,
    /// System configuration invalid
    InvalidSystemConfiguration = 403,
    /// Permission denied for hardware access
    PermissionDenied = 404,
    
    // Configuration Errors (500-599)
    /// Invalid device configuration
    InvalidDeviceConfiguration = 500,
    /// Invalid memory configuration
    InvalidMemoryConfiguration = 501,
    /// Invalid compute configuration
    InvalidComputeConfiguration = 502,
    /// Configuration file not found
    ConfigurationFileNotFound = 503,
    /// Configuration parsing failed
    ConfigurationParsingFailed = 504,
    
    // Runtime Errors (600-699)
    /// Device connection lost during operation
    DeviceConnectionLost = 600,
    /// Operation timeout exceeded
    OperationTimeout = 601,
    /// Resource busy or locked
    ResourceBusy = 602,
    /// Memory limit exceeded
    MemoryLimitExceeded = 603,
    /// Compute operation failed
    ComputeOperationFailed = 604,
    
    // Validation Errors (700-799)
    /// Input parameters invalid
    InvalidParameters = 700,
    /// Data format unsupported
    UnsupportedDataFormat = 701,
    /// Array dimensions invalid
    InvalidArrayDimensions = 702,
    /// Data type unsupported
    UnsupportedDataType = 703,
    /// Operation not supported on current device
    UnsupportedOperation = 704,
    
    // Unknown/Unexpected Errors (900-999)
    /// Unknown error occurred
    UnknownError = 900,
    /// Unexpected exception caught
    UnexpectedException = 901,
    /// Internal framework error
    InternalError = 902
};

/// Utility class for error handling and messaging
class OperationStatusHandler {
public:
    /// Converts error code to human-readable string
    ///
    /// - Parameter error: The error code to convert
    /// - Returns: Descriptive string for the error
    static std::string errorToString(OperationStatusCode error);
    
    /// Checks if error represents success
    ///
    /// - Parameter error: The error code to check
    /// - Returns: `true` if operation was successful, `false` otherwise
    static bool isSuccess(OperationStatusCode error);
    
    /// Checks if error is recoverable
    ///
    /// - Parameter error: The error code to check
    /// - Returns: `true` if error might be recoverable, `false` for fatal errors
    static bool isRecoverable(OperationStatusCode error);
    
    /// Gets error category for grouping
    ///
    /// - Parameter error: The error code to categorize
    /// - Returns: String representing the error category
    static std::string getErrorCategory(OperationStatusCode error);
    
    /// Prints formatted error information
    ///
    /// - Parameter error: The error code to display
    static void printError(OperationStatusCode error);
};

#endif // ERROR_H
