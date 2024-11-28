cmake_minimum_required(VERSION 3.0.0)
project(helmet VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

if (MSVC)
    add_compile_options("/utf-8")
    add_compile_options("/arch:AVX2")
    add_compile_options("/GL")
    add_link_options("/LTCG")
endif()

cmake_policy(SET CMP0074 NEW)

aux_source_directory(./src srcs)
file(GLOB headers inc/*.h inc/*.hpp)
include_directories(inc)

#Opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 设置 ONNX Runtime 路径
set(ONNXRUNTIME_DIR I:/package/CPPLibs/msvc/onnxruntime-win-gpu-x64-1.7.0/onnxruntime-win-gpu-x64-1.7.0)

# 添加 include 和 lib 路径
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(LCS ${srcs} ${headers} )
target_link_libraries(LCS PRIVATE  ${OpenCV_LIBS}  onnxruntime)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
