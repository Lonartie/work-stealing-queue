cmake_minimum_required(VERSION 3.18)
project(WorkStealingQueue VERSION 1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# C++ standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Build type default
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to Release")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Header-only interface target
# ---------------------------------------------------------------------------
add_library(wsq INTERFACE)
target_include_directories(wsq INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(wsq INTERFACE cxx_std_20)

# ---------------------------------------------------------------------------
# Compiler warnings / optimisation flags
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(wsq INTERFACE
    $<$<CONFIG:Release>:-O3 -march=native>
    $<$<CONFIG:Debug>:-O0 -g -fsanitize=address,undefined>
  )
  target_link_options(wsq INTERFACE
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
  )
elseif(MSVC)
  target_compile_options(wsq INTERFACE
    $<$<CONFIG:Release>:/O2>
    $<$<CONFIG:Debug>:/Od /Zi>
  )
endif()

# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)
target_link_libraries(wsq INTERFACE Threads::Threads)

# ---------------------------------------------------------------------------
# Sub-projects
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(unittests)
add_subdirectory(examples)
