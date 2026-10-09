cmake_minimum_required(VERSION 3.22.1)
project(neo3d LANGUAGES CXX)

add_library(neo3d SHARED
    native_engine.cpp
)
target_compile_features(neo3d PRIVATE cxx_std_20)
target_compile_options(neo3d PRIVATE -Wall -Wextra -Wpedantic)
find_library(android-lib android REQUIRED)
find_library(log-lib log REQUIRED)
find_library(vulkan-lib vulkan REQUIRED)
target_link_libraries(neo3d PRIVATE ${android-lib} ${log-lib} ${vulkan-lib})

target_include_directories(neo3d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../engine)
