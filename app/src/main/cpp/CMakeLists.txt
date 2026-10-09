cmake_minimum_required(VERSION 3.22.1)
project(neo3d LANGUAGES CXX)

# Keep the NDK shared C++ runtime explicit for APK packaging.
set(CMAKE_ANDROID_STL_TYPE c++_shared)

find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(VERT_SPV ${GENERATED_DIR}/mesh.vert.spv)
set(FRAG_SPV ${GENERATED_DIR}/mesh.frag.spv)
set(SKY_VERT_SPV ${GENERATED_DIR}/sky.vert.spv)
set(SKY_FRAG_SPV ${GENERATED_DIR}/sky.frag.spv)
set(SHADER_HEADER ${GENERATED_DIR}/shader_blobs.hpp)

add_custom_command(
    OUTPUT ${VERT_SPV}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_DIR}
    COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.0 -S vert -o ${VERT_SPV} ${SHADER_DIR}/mesh.vert
    DEPENDS ${SHADER_DIR}/mesh.vert
    VERBATIM
)
add_custom_command(
    OUTPUT ${FRAG_SPV}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_DIR}
    COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.0 -S frag -o ${FRAG_SPV} ${SHADER_DIR}/mesh.frag
    DEPENDS ${SHADER_DIR}/mesh.frag
    VERBATIM
)
add_custom_command(
    OUTPUT ${SKY_VERT_SPV}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_DIR}
    COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.0 -S vert -o ${SKY_VERT_SPV} ${SHADER_DIR}/sky.vert
    DEPENDS ${SHADER_DIR}/sky.vert
    VERBATIM
)
add_custom_command(
    OUTPUT ${SKY_FRAG_SPV}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_DIR}
    COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.0 -S frag -o ${SKY_FRAG_SPV} ${SHADER_DIR}/sky.frag
    DEPENDS ${SHADER_DIR}/sky.frag
    VERBATIM
)
add_custom_command(
    OUTPUT ${SHADER_HEADER}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spirv.py ${SHADER_HEADER} ${VERT_SPV} ${FRAG_SPV} ${SKY_VERT_SPV} ${SKY_FRAG_SPV}
    DEPENDS ${VERT_SPV} ${FRAG_SPV} ${SKY_VERT_SPV} ${SKY_FRAG_SPV} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spirv.py
    VERBATIM
)
add_custom_target(neo3d_shaders DEPENDS ${SHADER_HEADER})

add_library(neo3d SHARED native_engine.cpp)
add_dependencies(neo3d neo3d_shaders)
target_compile_features(neo3d PRIVATE cxx_std_20)
target_compile_options(neo3d PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(neo3d PRIVATE
    ${GENERATED_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../engine
)
find_library(android-lib android REQUIRED)
find_library(log-lib log REQUIRED)
find_library(vulkan-lib vulkan REQUIRED)
target_link_libraries(neo3d PRIVATE ${android-lib} ${log-lib} ${vulkan-lib})
