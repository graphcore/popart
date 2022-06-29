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

set(DEFAULT_TEST_VARIANTS "Cpu" CACHE STRING
    "The device variants to run tests on when no variants are set explicitly")

# Generate tests for these devices and variants. Cmake variable ENABLED_TEST_VARIANTS can
# be set with -DPOPART_CMAKE_ARGS=-DENABLED_TEST_VARIANTS=Hw for Hw only tests, for example.
# -DPOPART_CMAKE_ARGS="-DENABLED_TEST_VARIANTS=Cpu$<SEMICOLON>IpuModel$<SEMICOLON>Hw"
# Will enable all tests - note that you need to use '$<SEMICOLON>' not ';'
set(ENABLED_TEST_VARIANTS "Cpu;IpuModel" CACHE STRING "Default (non-hw) test variants")

# Use of the pytest tmpdir fixture leaves files in /tmp after running the tests.
# By setting --basetemp when running pytest, we can set the temp directory to be
# a subdirectory of the build directory, which will be cleared up with the build.
set(PYTEST_BASETEMP ${CMAKE_CURRENT_BINARY_DIR}/pytest_tmp)
file(MAKE_DIRECTORY ${PYTEST_BASETEMP})

# Add comparator checks to the test files
if(${POPART_STRICT_COMPARATOR_CHECKS})
  add_compile_definitions(POPART_STRICT_COMPARATOR_CHECKS)
endif(${POPART_STRICT_COMPARATOR_CHECKS})

# Remove any requested variants not present in ${ENABLED_TEST_VARIANTS}
# If no variants were requested filtered ${DEFAULT_TEST_VARIANTS} are returned.
function(sanitise_variants variant_name)
  set(variants ${${variant_name}})
  set(old_variants "${variants}")
  if (NOT old_variants)
    set(old_variants ${DEFAULT_TEST_VARIANTS})
  endif()
  set(new_variants "")
  # Only allow each variant if it is enabled:
  foreach(variant ${old_variants})
    if (${variant} IN_LIST ENABLED_TEST_VARIANTS)
      list(APPEND new_variants ${variant})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES new_variants)
  set(${variant_name} "${new_variants}" PARENT_SCOPE)
endfunction()

# A VARIANT is a colon separated pair "target:[runconfig]". This function
# extracts the "target" part storing it in the second argument.
function(extract_target variant target)
  string(REPLACE ":" ";" vspec ${${variant}})
  list(GET vspec 0 HEAD)
  set(${target} ${HEAD} PARENT_SCOPE)
endfunction()

# A VARIANT is a colon separated pair "target:[runconfig]". This function
# extracts the "runconfig" part storing it in the second argument.
function(extract_runconfig variant run_config)
  string(REPLACE ":" ";" vspec ${${variant}})
  list(LENGTH vspec vlen)
  if (${vlen} EQUAL "1")
    set(${run_config} "default" PARENT_SCOPE)
  else()
    list(GET vspec 1 config)
    set(${run_config} ${config} PARENT_SCOPE)
  endif()
endfunction()

# A VARIANT is a colon separated pair "target:[runconfig]". This function
# extracts a list containing just the "target" parts. Where targets are listed
# with multiple runconfigs in the original list the duplicates are removed
# from the returned list.
function(extract_targets variants targets)
  set(extracted_targets "")
  foreach(variant ${${variants}})
    string(REPLACE ":" ";" vspec ${variant})
    list(GET vspec 0 HEAD)
    list(APPEND extracted_targets ${HEAD})
  endforeach()
  list(REMOVE_DUPLICATES extracted_targets)
  set(${targets} ${extracted_targets} PARENT_SCOPE)
endfunction()

function(link_boost target)
  target_link_libraries(${target}
    PRIVATE
      ${boost_targets}
  )
  if(BUILD_SHARED_LIBS)
    target_compile_definitions(${target}
      PRIVATE
        BOOST_TEST_DYN_LINK
  )
  endif()
endfunction()

# Usage: add_test_exectuable(<exe_name> <src1> ... TESTUTILS <util1> ...)
#
# `exe_name`: name of executable.
#
# `TESTUTILS`, optional:
#     are target names of test util libraries (either a library from
#     `tests/testutil` or `trompeloeil`) that a test may use. It is an optional
#     argument.
#
function(add_test_executable name)
  set(options "")
  set(oneValueKeywords "")
  set(multiValueKeywords "FILES;TEST_UTILS")
  set(argsWithFilesKeyword "FILES;${ARGN}")

  cmake_parse_arguments(add_test_exe
    "${options}" "${oneValueKeywords}" "${multiValueKeywords}"
    "${argsWithFilesKeyword}"
  )

  add_executable(${name} ${add_test_exe_FILES})
  target_compile_features(${name} PUBLIC cxx_std_14)
  target_include_directories(${name}
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(${name}
    PRIVATE
      popart-internal
      ${CMAKE_THREAD_LIBS_INIT}
      ${add_test_exe_TEST_UTILS}
  )
  link_boost(${name})

  set_property(TARGET ${name}
              APPEND_STRING PROPERTY
              COMPILE_FLAGS "-DTEST_TARGET=TestDeviceType::${TEST_TARGET}")
endfunction()

# Optional arguments:
# VARIANTS
# Run with the specified VARIANTS (or all enabled VARIANTS if
# none are specified). The first parameter is the test name, the following are
# source files, and finally everything after VARIANTS are variant
# specifications.
#
# Mark the test as requiring two IPUs - the relevant fixture will be added
#
# Currently supported targets are "Cpu;IpuModel;Sim;Hw"
#
#
# LABELS
# Semicolon separated list of labels to be set for this test. Supported labels:
#   - multicard: Run the Hw variant on the POD diff builder only, not the
#                C200 builder.
#
# DUAL
# A variation specification must be of the form "target:[config]"
#
# TEST_UTILS
# The target names of test utils that this test will require. To see the target
# names of the test utils, look in /tests/testutil/CMakeLists.txt, where they
# are created.
#
function(add_popart_cpp_unit_test name)
  set(oneValueArgs "VARIANTS;LABELS;TEST_UTILS")
  cmake_parse_arguments(test_args "" "FILES" "${oneValueArgs}" "FILES;${ARGN}")
  set(FILES "${test_args_FILES}")

  sanitise_variants(test_args_VARIANTS)
  extract_targets(test_args_VARIANTS TARGETS)

  if(DEFINED test_args_TEST_UTILS)
    set(TEST_UTILS_ARGS "TEST_UTILS;${test_args_TEST_UTILS}")
  else()
    set(TEST_UTILS_ARGS "")
  endif()

  foreach(TEST_TARGET ${TARGETS})
    set(executable_name "${TEST_TARGET}_${name}")
    add_test_executable(${executable_name} ${FILES} ${TEST_UTILS_ARGS})
  endforeach()

  foreach(VARIANT ${test_args_VARIANTS})
    extract_target(VARIANT TEST_TARGET)
    extract_runconfig(VARIANT TEST_CONFIG)

    set(test_name "${TEST_TARGET}_${TEST_CONFIG}_${name}")
    set(executable_name "${TEST_TARGET}_${name}")
    add_test(NAME "${test_name}"
      COMMAND ${executable_name}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    set(test_env ${TEST_ENVIRONMENT})
    if (${TEST_CONFIG} STREQUAL "cpp")
      list(APPEND test_env "IPU_POPLAR_RT_GP_SUFFIX=_c")
    endif()
    if (POPART_LOG_DEVICE_ACCESS_IN_TESTS)
      list(APPEND test_env "POPART_LOG_DEVICE_ACCESS_IN_TESTS=${CMAKE_BINARY_DIR}/deviceaccess.log")
      list(APPEND test_env "POPART_TEST_NAME=${test_name}")
    endif()

    set_tests_properties(${test_name} PROPERTIES
                         ENVIRONMENT "${test_env}")
    set_tests_properties(${test_name} PROPERTIES
                          LABELS "${test_args_LABELS}")

    if (${TEST_TARGET} STREQUAL "Hw")
      # Make sure tests that use physical IPUs only run if an appropriate
      # number were available according to the relevant test fixture:
      set_tests_properties(${test_name}
        PROPERTIES FIXTURES_REQUIRED SingleIpuIsAvailable)
    endif()
  endforeach()
endfunction()

# Python unit test function. The first parameter is the Python test file name
# (without the `.py` extension).
# Uses VARIANTS and LABELS as above for add_popart_cpp_unit_test
# Takes optional MATCHEXPR parameter, which is a string that will be passed
#   verbatim to pytest as a selector when running the test `-k <matchexpr_value>`.
# Takes optional NUM_WORKERS parameter that is passed to pytest as `-n NUM_WORKERS`.
#   It specifies the number of workers to launch for this particular test. In particular,
#   NUM_WORKERS can have a number value or be the string `auto`, in which case the number
#   of launched workers is equal to the number of available CPUs. In order to avoid
#   launching too many threads at once tests that use the NUM_WORKERS option are run
#   sequentially after any other tests have run (so the pytest subtests run in parallel,
#   but the individual ctest tests run sequentially)
function(add_popart_py_unit_test name)
  set(filename "${CMAKE_CURRENT_SOURCE_DIR}/${name}.py")
  get_filename_component(dir ${filename} DIRECTORY)
  get_filename_component(name ${filename} NAME_WE)
  file(RELATIVE_PATH dir "${PROJECT_SOURCE_DIR}/tests" ${dir})

  set(options "")
  set(oneValueArgs MATCHEXPR NUM_WORKERS)
  set(multiValueArgs LABELS VARIANTS)
  cmake_parse_arguments(test_args "${options}" "${oneValueArgs}" "${multiValueArgs}" "${ARGN}")

  sanitise_variants(test_args_VARIANTS)
  extract_targets(test_args_VARIANTS TARGETS)

  foreach(VARIANT ${test_args_VARIANTS})
    extract_target(VARIANT TEST_TARGET)
    extract_runconfig(VARIANT TEST_CONFIG)

    set(pytest_arguments -s "${filename}")

    if (NOT DEFINED test_args_MATCHEXPR)
      set(test_name "${TEST_TARGET}_${TEST_CONFIG}_${dir}/${name}")
    else()
      # Note, solid brackets and dash can form part of parameterised pytest names.
      string(REGEX REPLACE " " "_" MATCHEXPR_NAME "${test_args_MATCHEXPR}")
      set(MATCHEXPR_VALUE "${test_args_MATCHEXPR}")
      set(test_name "${TEST_TARGET}_${TEST_CONFIG}_${dir}/${name}-${MATCHEXPR_NAME}")
      list(APPEND pytest_arguments -k "${MATCHEXPR_VALUE}")
    endif()

    # Adding ${test_name} after PYTEST_BASETEMP because when running tests in
    # parallel, pytest removes the directories of running tests.
    # This is because by default, pytest removes entries in basetemp older
    # than the newest 3 entries.
    file(MAKE_DIRECTORY "${PYTEST_BASETEMP}/${test_name}")
    list(APPEND pytest_arguments "--basetemp=${PYTEST_BASETEMP}/${test_name}")

    if (DEFINED test_args_NUM_WORKERS)
      # Set number of threads to run test with
      list(APPEND pytest_arguments -n "${test_args_NUM_WORKERS}")
    endif()

    add_test(NAME "${test_name}"
            COMMAND ${Python3_EXECUTABLE} -m pytest ${pytest_arguments}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    if (DEFINED test_args_NUM_WORKERS)
      # Run this test serially, so that it doesn't starve other tests
      set_property(TEST "${test_name}" PROPERTY RUN_SERIAL TRUE)
    endif()

    # Setting TMPDIR=${PYTEST_BASETEMP} as `pytest --forked` is somehow
    # preventing poplar from cleaning up its temporary files if the test raises
    # an exception.
    set(test_env ${TEST_ENVIRONMENT})
    list(APPEND test_env "TEST_TARGET=${TEST_TARGET}")
    list(APPEND test_env "TMPDIR=${PYTEST_BASETEMP}")
    if (POPART_LOG_DEVICE_ACCESS_IN_TESTS)
      list(APPEND test_env "POPART_LOG_DEVICE_ACCESS_IN_TESTS=${CMAKE_BINARY_DIR}/deviceaccess.log")
      list(APPEND test_env "POPART_TEST_NAME=${test_name}")
    endif()

    set_tests_properties(${test_name} PROPERTIES
                         ENVIRONMENT "${test_env}")
    set_tests_properties(${test_name} PROPERTIES
                          LABELS "${test_args_LABELS}")
  endforeach()
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
    set_property(TEST upload_coverage_report PROPERTY DEPENDS )
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
if(( CMAKE_BUILD_TYPE STREQUAL "Debug" ) AND ( DEFINED ENV{CI} ))
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
