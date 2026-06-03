cmake_minimum_required(VERSION 3.20)

project(AAMED_OpenCV VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs flann)

set(AAMED_CORE_SOURCES
    src/adaptApproxPolyDP.cpp
    src/Contours.cpp
    src/EllipseNonMaximumSuppression.cpp
    src/FLED.cpp
    src/FLED_Export.cpp
    src/FLED_drawAndWriteFunctions.cpp
    src/FLED_Initialization.cpp
    src/FLED_PrivateFunctions.cpp
    src/Group.cpp
    src/LinkMatrix.cpp
    src/Node_FC.cpp
    src/Segmentation.cpp
    src/Validation.cpp
)

add_library(aamed_core STATIC ${AAMED_CORE_SOURCES})
target_include_directories(aamed_core PUBLIC src)
target_link_libraries(aamed_core PUBLIC ${OpenCV_LIBS})
target_compile_definitions(aamed_core PUBLIC AAMED_OPENCV_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
target_compile_options(aamed_core PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX>
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-mavx>
)

add_executable(aamed_demo src/main.cpp)
target_link_libraries(aamed_demo PRIVATE aamed_core)

add_executable(aamed_eval tools/aamed_eval.cpp)
target_compile_definitions(aamed_eval PRIVATE AAMED_OPENCV_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

set_target_properties(aamed_demo aamed_eval PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
