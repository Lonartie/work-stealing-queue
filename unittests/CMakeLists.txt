cmake_minimum_required(VERSION 3.18)

add_executable(unittest_wsq unittest_wsq.cpp)
target_link_libraries(unittest_wsq PRIVATE wsq)

# ---------------------------------------------------------------------------
# Test grid: scale x n_thieves
#
# scale     controls item counts / round counts (1 = quick, 4 = thorough)
# n_thieves is the number of concurrent steal threads in Section 3
#
# Test names follow the pattern:
#   unittest_wsq.scale=S.thieves=T
#
# Run the full grid:    ctest
# Run just scale=1:     ctest -R "scale=1"
# Run just 4 thieves:   ctest -R "thieves=4"
# Run a soak pass:      ctest -R "scale=4"
# Verbose output:       ctest --output-on-failure -V
# ---------------------------------------------------------------------------

set(WSQ_TEST_SCALES   "1;2;3;4"       CACHE STRING "Scale values for the test grid")
set(WSQ_TEST_THIEVES  "1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16" CACHE STRING "Thief-thread counts for the test grid")

foreach(scale IN LISTS WSQ_TEST_SCALES)
  foreach(thieves IN LISTS WSQ_TEST_THIEVES)
    set(test_name "unittest_wsq.scale=${scale}.thieves=${thieves}")
    add_test(
      NAME    ${test_name}
      COMMAND unittest_wsq ${scale} ${thieves}
    )
    # Give larger configurations more wall-clock budget
    # base=60s, +30s per unit of scale, +5s per extra thief beyond 1
    math(EXPR timeout "60 + 30 * ${scale} + 5 * (${thieves} - 1)")
    set_tests_properties(${test_name} PROPERTIES TIMEOUT ${timeout})
  endforeach()
endforeach()
