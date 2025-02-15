cmake_minimum_required(VERSION 3.10)
project(HyperionFuzzy)

# Find PyBind11
find_package(pybind11 REQUIRED)

# Find dlib
find_package(dlib REQUIRED)
include(DirList)

add_library(hypersphere SHARED
    cpp/hypersphere.cpp
    pybind/hypersphere_bindings.cpp
)
target_include_directories(hypersphere PUBLIC "include")
target_link_libraries(hypersphere PRIVATE pybind11::pybind11)

add_library(optimize SHARED
    cpp/optimize_hypersphere.cpp
    pybind/optimize_bindings.cpp
)
target_include_directories(optimize PUBLIC "include")
target_link_libraries(optimize PRIVATE pybind11::pybind11 dlib::dlib)

add_library(fuzzy SHARED
    cpp/fuzzy_contribution.cpp
    pybind/fuzzy_bindings.cpp
)
target_include_directories(fuzzy PUBLIC "include")
target_link_libraries(fuzzy PRIVATE pybind11::pybind11)