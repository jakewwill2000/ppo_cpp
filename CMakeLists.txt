cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ppo_cpp)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(run_tests src/rollout_buffer.cpp tests/run_tests.cpp)
target_link_libraries(run_tests "${TORCH_LIBRARIES}")
set_property(TARGET run_tests PROPERTY CXX_STANDARD 14)