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

option(POPART_BUILD_TESTING "Build the popart tests" ON)
option(POPART_BUILD_EXAMPLES "Build the popart examples" ON)

# Generate compile_commands.json file for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use GOLD linker if g++ to speed up compilation
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # Based on https://github.com/frobware/c-hacks/blob/master/
  # cmake/use-gold-linker.cmake
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -fuse-ld=gold -Wl,--version OUTPUT_VARIABLE stdout ERROR_QUIET)
  if("${stdout}" MATCHES "GNU gold")
    foreach(BINARY_TYPE EXE MODULE SHARED STATIC)
      set(CMAKE_${BINARY_TYPE}_LINKER_FLAGS "${CMAKE_${BINARY_TYPE}_LINKER_FLAGS} -fuse-ld=gold")
    endforeach()
  endif()
endif()

# Paths used for packaging.
# Default to the original paths to ensure backwards compatability.
if(NOT POPART_PKG_DIR)
  set(POPART_PKG_DIR "../../../pkg")
endif()
if(NOT POPART_CBT_VIEW_DIR)
  set(POPART_CBT_VIEW_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
endif()
message(STATUS "Package directory is ${POPART_PKG_DIR}")
message(STATUS "View directory is ${POPART_CBT_VIEW_DIR}")
