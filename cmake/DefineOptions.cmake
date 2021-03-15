# cmake option definitions for popart

if(DEFINED POPLAR_INSTALL_DIR AND POPLAR_INSTALL_DIR AND NOT EXISTS "${POPLAR_INSTALL_DIR}")
  message(WARNING "POPLAR_INSTALL_DIR does not exist: ${POPLAR_INSTALL_DIR}")
endif()
set(POPLAR_INSTALL_DIR "" CACHE PATH
    "Absolute path to Poplar install directory"
)

set(C10_DIR "" CACHE PATH
    "Absolute path to directory to install Cifar-10 dataset to"
)

# Paths used for packaging.
# Default POPART_PKG_DIR to original path to ensure backwards compatability.
if(NOT DEFINED POPART_PKG_DIR)
  message(STATUS "Package directory defaulted to ../../../pkg")
endif()
if(NOT DEFINED POPART_CBT_VIEW_DIR)
  message(VERBOSE "View directory defaulted to ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

set(POPART_PKG_DIR "../../../pkg" CACHE PATH
    "Absolute path to where the Popart package produced by CPack will be moved"
)

set(POPART_CBT_VIEW_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH
      "Absolute path to directory of the view that is building Popart, used to\
  compute a version hash for the Popart package."
)
mark_as_advanced(POPART_CBT_VIEW_DIR)

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

## We have removed many custom Find<pkg> modules in popart. These used variables
## like ONNX_INSTALL_DIR etc. as hints. We now use the the standard methods of
## pointing cmake at a package location, e.g. Protobuf_ROOT.
##
## Thus, we convert any *_INSTALL_DIR variables passed to *_ROOT variables, and
## print a deprecation warning.

# Verbatim package names given to find_package
set(popart_deps_with_var_needed_for_bwd_compat
  ONNX
  Protobuf
  poprithms
  spdlog
  pybind11
)

foreach(dep ${popart_deps_with_var_needed_for_bwd_compat})
  string(TOUPPER ${dep} DEP_UPPER)
  set(dep_old_var ${DEP_UPPER}_INSTALL_DIR)
  set(dep_root ${dep}_ROOT)
  
  if(DEFINED ${dep_old_var})
    message(DEPRECATION "Using the variable ${dep_old_var} to point Popart at ${dep} is deprecated. Please use ${dep_root} instead.")
    set(${dep_root} ${${dep_old_var}})
  endif()
endforeach()
