# cmake option definitions for popart

# colorful ninja, idea from
# medium.com/@alasher/colored-c-compiler-output-with-ninja-clang-gcc-10bfe7f2b949
option (FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." TRUE)
if (${FORCE_COLORED_OUTPUT})
    MESSAGE(STATUS "CMAKE_CXX_COMPILER_ID is \""   ${CMAKE_CXX_COMPILER_ID} "\"")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
       add_compile_options (-fdiagnostics-color=always)
     elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang" OR
             "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" )
       add_compile_options (-fcolor-diagnostics)
    endif ()
endif ()

option(POPART_USE_STACKTRACE "Enable boost stacktrace reports in error messages" ON)
if (${POPART_USE_STACKTRACE})
    # Building with Boost Stacktrace using the default header only implementation
    # Note this is only supported for any compiler on POSIX or MinGW.
    message(STATUS "Building popart with Boost Stacktrace")
    add_definitions(-DPOPART_USE_STACKTRACE -DBOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED)
endif()
