# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

project(hexl_example LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 17)

# Example using source
find_package(HEXL 1.2.6
    HINTS ${HEXL_HINT_DIR}
    REQUIRED)
if (NOT TARGET HEXL::hexl)
    message(FATAL_ERROR "TARGET HEXL::hexl not found")
endif()

add_executable(example ../example.cpp)
target_link_libraries(example PRIVATE HEXL::hexl)
