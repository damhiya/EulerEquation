cmake_minimum_required(VERSION 3.12)
project(EulerEquation)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(main main.cpp)
target_link_libraries(main Eigen3::Eigen)
