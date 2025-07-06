cmake_minimum_required(VERSION 3.31)
project(PONG_AI)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)

# Source files
file(GLOB_RECURSE SOURCES "src/**/*.cpp")

# Main executable
add_executable(PONG_AI main.cpp ${SOURCES})

# Test executables
add_executable(test_tensor tests/test_tensor.cpp)
add_executable(test_neural_network tests/test_neural_network.cpp)
add_executable(test_agent_env tests/test_agent_env.cpp)
