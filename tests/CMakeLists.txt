# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
find_package(Boost 1.70 REQUIRED COMPONENTS unit_test_framework filesystem random system)

set(boost_targets
  unit_test_framework
  filesystem
  random
  system
  boost # All header-only libraries.
)
list(TRANSFORM boost_targets PREPEND Boost:: OUTPUT_VARIABLE boost_targets)

# Workaround for this bug: https://github.com/boostorg/system/issues/26
# on certain systems/compilers (e.g. compiling the host-runtime-view
# on Centos 7.6 with GCC 7.3).
# TODO: When CMake 3.12 is required use add_compile_definitions() instead.
add_definitions(-DBOOST_ERROR_CODE_HEADER_ONLY)

find_package(trompeloeil 35 REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${GCCS_CMAKE_DIR}")
include(AddUnitTest)
include(AddPythonUnitTest)

set(DEFAULT_TEST_VARIANTS "Cpu" CACHE STRING
  "The device variants to run tests on when no variants are set explicitly")

# Generate tests for these devices and variants. Cmake variable ENABLED_TEST_VARIANTS can
# be set with -DPOPART_CMAKE_ARGS=-DENABLED_TEST_VARIANTS=Hw for Hw only tests, for example.
# -DPOPART_CMAKE_ARGS="-DENABLED_TEST_VARIANTS=Cpu$<SEMICOLON>IpuModel$<SEMICOLON>Hw"
# Will enable all tests - note that you need to use '$<SEMICOLON>' not ';'
# ENABLED_TEST_VARIANTS are the variants of the tests that will be run, i.e. a
# test will be run iff it is in ENABLED_TEST_VARIANTS
# DEFAULT_TEST_VARIANTS are the default VARIANTS to use if no VARIANT is specified.
# (Ususally Cpu)
set(ENABLED_TEST_VARIANTS_DEFAULT Cpu)
set(ENABLED_TEST_VARIANTS ${ENABLED_TEST_VARIANTS_DEFAULT}
  CACHE STRING "Generate tests for these devices and variants")

# To enable backwards compatibility, if the user has specified "IpuModel" variant
# (without a trailing digit) then add both IpuModel2 and IpuModel21 variants.
if(ENABLED_TEST_VARIANTS MATCHES "IpuModel[0-9]")
  # CMake regex doesn't have a negation character. So use if/else
else()
  list(APPEND ENABLED_TEST_VARIANTS "IpuModel2;IpuModel21")
endif()

message(STATUS "Popart default test variants: ${DEFAULT_TEST_VARIANTS}")
message(STATUS "Popart enabled test variants: ${ENABLED_TEST_VARIANTS}")

# Use of the pytest tmpdir fixture leaves files in /tmp after running the tests.
# By setting --basetemp when running pytest, we can set the temp directory to be
# a subdirectory of the build directory, which will be cleared up with the build.
set(PYTEST_BASETEMP ${CMAKE_CURRENT_BINARY_DIR}/pytest_tmp)
file(MAKE_DIRECTORY ${PYTEST_BASETEMP})

# Add comparator checks to the test files
if(${POPART_STRICT_COMPARATOR_CHECKS})
  add_compile_definitions(POPART_STRICT_COMPARATOR_CHECKS)
endif(${POPART_STRICT_COMPARATOR_CHECKS})

# Allows user to provide additional compiler flags
set(COMMON_ADD_EXECUTABLE_COMPILE_FLAGS "")

# List of libraries required to build executable
# equivalent to target_link_libraries()
set(COMMON_ADD_EXECUTABLE_LIBRARIES
  popart-internal
  $<TARGET_OBJECTS:testdevice>
  ${CMAKE_THREAD_LIBS_INIT}
  ${boost_targets}
)

# Include paths required to build executable
# equivalent to target_include_directories()
set(COMMON_ADD_EXECUTABLE_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/testutil/testdevice/include
)

# This is a wrapper around add_python_unit_test in AddPythonUnitTest.cmake in GCCS.
# It adds a few PopART specific properties, then passes through the remaining arguments
# to GCCS.
function(add_popart_py_unit_test name)
  set(filename "${CMAKE_CURRENT_SOURCE_DIR}/${name}.py")
  get_filename_component(dir ${filename} DIRECTORY)
  get_filename_component(name ${filename} NAME_WE)
  file(RELATIVE_PATH dir "${PROJECT_SOURCE_DIR}/tests" ${dir})

  set(OPTIONS "")
  set(ONE_VALUE_KEYWORDS "MATCHEXPR;NUM_WORKERS;TEST_IPUMODEL;ALLOW_SKIP")
  set(MULTI_VALUE_KEYWORDS "VARIANTS;LABELS;DEPENDS;PROPERTIES")
  cmake_parse_arguments(ARGS "${OPTIONS}" "${ONE_VALUE_KEYWORDS}" "${MULTI_VALUE_KEYWORDS}" "${ARGN}")

  # Setting TMPDIR=${PYTEST_BASETEMP} as `pytest --forked` is somehow
  # preventing poplar from cleaning up its temporary files if the test raises
  # an exception.
  list(APPEND test_env "TMPDIR=${PYTEST_BASETEMP}")

  if(POPART_LOG_DEVICE_ACCESS_IN_TESTS)
    list(APPEND test_env "POPART_LOG_DEVICE_ACCESS_IN_TESTS=${CMAKE_BINARY_DIR}/deviceaccess.log")
    list(APPEND test_env "POPART_TEST_NAME=${name}")
  endif()

  # Prepend the directory to the test name.
  set(name "${dir}/${name}")

  add_python_unit_test(${name} ${filename}
    VARIANTS ${ARGS_VARIANTS}
    LABELS ${ARGS_LABELS}
    DEPENDS ${ARGS_DEPENDS}
    ENV ${test_env}
    PROPERTIES ${ARGS_PROPERTIES}
    TEST_IPUMODEL ${ARGS_TEST_IPUMODEL}
    ALLOW_SKIP ${ARGS_ALLOW_SKIP}
    MATCHEXPR ${ARGS_MATCHEXPR}
    NUM_WORKERS ${ARGS_NUM_WORKERS}
    PYTHON3_PATH ${Python3_EXECUTABLE}
  )
endfunction()

if(${POPART_ENABLE_COVERAGE} AND ${UPLOAD_COVERAGE_REPORT})
  if(RESULTS_SERVER_COOKIE_FILE AND EXISTS ${RESULTS_SERVER_COOKIE_FILE})
    add_test(NAME upload_coverage_report
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/upload_coverage.py"
      ${CMAKE_BINARY_DIR}
      https://elasticsearch.eks.sourcevertex.net
      --cookie-file=${RESULTS_SERVER_COOKIE_FILE}
    )

    # We want the coverage report generation to run after all other tests to
    # be able to reliably clean all previously generated coverage output.
    # Otherwise, the unit test coverage reports will contain coverage
    # data from tests that ran before, or alongside, the coverage script.
    set_property(TEST upload_coverage_report PROPERTY DEPENDS)
    set_property(TEST upload_coverage_report PROPERTY RUN_SERIAL TRUE)
  else()
    message(WARNING " Path to authentication cookie not provided. Skipping coverage report upload. \
Set the RESULTS_SERVER_COOKIE_FILE to the absolute path of the file containing the cookie if you \
want to upload coverage reports to elasticsearch."
    )
  endif()
endif()

# Add subdirectories for tests:
add_subdirectory(testutil)

# Don't add integration tests in Debug mode on CI - this saves a lot of time on debug CI builds.
if((CMAKE_BUILD_TYPE STREQUAL "Debug") AND (DEFINED ENV{CI}))
  message(STATUS "On a CI + Debug build, skipping integration tests...")
else()
  message(STATUS "Not on a CI + Debug build, adding integration tests...")
  add_subdirectory(integration)
  add_subdirectory(onnx_backend)
endif()

add_subdirectory(unittests)

install(DIRECTORY .
  DESTINATION ${INSTALL_TESTS}
  FILES_MATCHING REGEX ".*\.(py|sh|md)$")

add_subdirectory(linters)
