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

#CMinpack
find_package(CMinpack REQUIRED)
include_directories(${CMinpack_INCLUDE_DIRS})

#CUDA
find_package(CUDA REQUIRED)
message(STATUS "lib:${CUDA_LIBRARIES}")
message(STATUS "inc:${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})

# tensorrt
set(TENSORRT_ROOT_PATH I:/package/CPPLibs/msvc/TensorRT-8.4.3.1.Windows10.x86_64.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1)
set(TENSORRT_INCLUDE_DIRS ${TENSORRT_ROOT_PATH}/include/)   # 头文件夹
include_directories(${TENSORRT_INCLUDE_DIRS})
file(GLOB TENSORRT_LIB ${TENSORRT_ROOT_PATH}/lib/*.lib)

# boost
find_package(Boost REQUIRED COMPONENTS)
link_directories(${Boost_LIBRARY_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

add_executable(LCS ${srcs} ${headers} )
target_link_libraries(LCS PRIVATE  ${OpenCV_LIBS} cminpack::cminpack ${CUDA_LIBRARIES} ${TENSORRT_LIB})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
