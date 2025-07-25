# cmake 版本
cmake_minimum_required(VERSION 3.1)
# 项目名
project(unet)
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    MESSAGE("\n===This is Linux,CMAKE_SYSTEM_NAME:${CMAKE_SYSTEM_NAME}")
ELSEIF(CMAKE_SYSTEM_NAME MATCHES "Windows")
    MESSAGE("===This is Windows,CMAKE_SYSTEM_NAME:${CMAKE_SYSTEM_NAME}")
ELSE()
    MESSAGE("===This platform:${CMAKE_SYSTEM_NAME}")
ENDIF()

# 编译增加C++11 支持
add_definitions(-std=c++11)
# 设置C++标准为 C++ 11
set(CMAKE_CXX_STANDARD 11)
#指定生成的版本
set(CMAKE_BUILD_TYPE Release)
#设置可执行文件输出目录
#SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG D:/04vs_file/bin/)
#设置库输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG D:/04vs_file/bin/)

#option 选项开关
#启用后，将在中使用CUDA运行时库的静态版本CUDA_LIBRARIES。
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# -D_MWAITXINTRIN_H_INCLUDED for solving error: identifier "__builtin_ia32_mwaitx" is undefined
##-Wall  #-Wfatal-errors 
#指定编译选项
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast -D_MWAITXINTRIN_H_INCLUDED")


include_directories(${PROJECT_SOURCE_DIR}/include/)
include_directories(${PROJECT_SOURCE_DIR}/)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    # setup CUDA
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
	
    # tensorrt
    include_directories(/usr/include/aarch64-linux-gnu/)
    link_directories(/usr/lib/aarch64-linux-gnu/)
    # setup opencv
    find_package(OpenCV)
	include_directories(${OpenCV_INCLUDE_DIRS})
ELSEIF(CMAKE_SYSTEM_NAME MATCHES "Windows")
	# setup CUDA
	set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11; -g; -G;-gencode; arch=compute_75;code=sm_75)
	find_package(CUDA REQUIRED)
	include_directories(${CUDA_INCLUDE_DIRS})
	#enable_language(CUDA)  # add this line, then no need to setup cuda path in vs
	
	# tensorrt
	set(TRT_DIR "D:/tools/TensorRT-8.6.1.6-cu118//")  #3
	include_directories(${TRT_DIR}\\include)
	link_directories(${TRT_DIR}\\lib)
	
	# setup opencv
	set(OpenCV_DIR "D:/tools/opencv455/build/")  #2
    find_package(OpenCV REQUIRED)
	include_directories(${OpenCV_INCLUDE_DIRS})
	# Threads
	set(THREADS_PREFER_PTHREAD_FLAG ON)
	find_package(Threads REQUIRED)
	message(STATUS "Threads library status: ")
	message(STATUS "    Threads_FOUND: ${Threads_FOUND}")
	message(STATUS "    include path: ${Threads_FIND_QUIETLY}")
ENDIF()

#打印信息
message("===message=========================================")
message(STATUS "CUDA library status:")
message(STATUS "    CUDA_FOUND: ${CUDA_FOUND}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
message(STATUS "OpenCV library status: ")
message(STATUS "    OpenCV_FOUND: ${OpenCV_FOUND}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message("===message=========================================")

#生成目标文件
add_executable(unet ${PROJECT_SOURCE_DIR}/unet.cpp)
target_link_libraries(unet nvinfer)
target_link_libraries(unet ${CUDA_LIBRARIES})
target_link_libraries(unet ${OpenCV_LIBS})
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
	add_definitions(-O2 -pthread)
ELSEIF(CMAKE_SYSTEM_NAME MATCHES "Windows")
ENDIF()

