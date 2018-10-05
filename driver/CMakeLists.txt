SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic" )
include_directories("../neuralnet/include")
add_executable (driver driver.cpp)
target_link_libraries(driver neuralnet)


install(TARGETS driver
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

