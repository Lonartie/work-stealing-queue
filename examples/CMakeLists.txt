cmake_minimum_required(VERSION 3.18)

add_executable(simple   simple.cpp)
add_executable(evaluate evaluate.cpp)

target_link_libraries(simple   PRIVATE wsq)
target_link_libraries(evaluate PRIVATE wsq)
