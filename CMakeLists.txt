cmake_minimum_required(VERSION 3.15)
project(HyperTune)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add FetchContent module
include(FetchContent)

# Fetch nlohmann/json
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# Add Crow subdirectory
add_subdirectory(Crow)

# List all HyperTune source files
set(HYPERTUNE_SOURCES
    HyperTune/src/hyperparameter.cpp
    HyperTune/src/bayesianOptimization.cpp
    HyperTune/src/tuner.cpp
    HyperTune/src/mlFmkModelAdapter.cpp
    HyperTune/src/NeuralNetworkAdapter.cpp
    HyperTune/src/randomForestAdapter.cpp
    HyperTune/src/randomSearch.cpp
    HyperTune/src/searchStrategy.cpp
)

# Add include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/external/crow/include
)

# Create HyperTune library
add_library(HyperTuneLib STATIC ${HYPERTUNE_SOURCES})
target_include_directories(HyperTuneLib PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Add executable
add_executable(HyperTune 
    src/main.cpp
    src/webserver.cpp
)

# Find required packages
find_package(Threads REQUIRED)

# Link libraries
target_link_libraries(HyperTune
    PRIVATE
    HyperTuneLib
    Threads::Threads
    Crow::Crow
    nlohmann_json::nlohmann_json
    ws2_32
    mswsock
)

# Set include directories for the target
target_include_directories(HyperTune
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)