cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0148 OLD)
project(Xgboost_Tests)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure XGBoost build options before adding the subdirectory
set(BUILD_TESTING OFF CACHE BOOL "Build tests for XGBoost")
set(BUILD_CLI OFF CACHE BOOL "Build XGBoost CLI")
set(DISABLE_PYTHON ON CACHE BOOL "Disable Python support in XGBoost")

add_subdirectory(xgboost)

add_executable(Xgboost_Tests classification1.cpp)
target_link_libraries(Xgboost_Tests PRIVATE xgboost)

# Copy DLL to output directory (Windows only)
if(WIN32)
    add_custom_command(TARGET Xgboost_Tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/xgboost/lib/xgboost.dll"
            $<TARGET_FILE_DIR:Xgboost_Tests>
    )
endif()