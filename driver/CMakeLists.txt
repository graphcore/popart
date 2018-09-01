# note : this pydriver dir is the same as that in tensorflow_enigma : TODO merge
# them somehow.
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic" )
include_directories("../neuralnet/include")
add_executable (pydriver pydriver.cpp)
target_link_libraries(pydriver neuralnet)


install(TARGETS pydriver
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

