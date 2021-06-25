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
#
# Semicolon separated list of labels to be set for this test
#
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
      WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})

    set(test_env ${TEST_ENVIRONMENT})
    if (${TEST_CONFIG} STREQUAL "cpp")
      list(APPEND test_env "IPU_POPLAR_RT_GP_SUFFIX=_c")
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

# Python unit test function
# Uses VARIANTS as above for add_popart_cpp_unit_test
# Takes optional MATCHEXPR parameter, which is a string that will be passed
# verbatim to pytest as a selector when running the test `-k <matchexpr_value>`.
function(add_popart_py_unit_test name filename)
  set(options "")
  set(oneValueArgs MATCHEXPR)
  set(multiValueArgs LABELS VARIANTS)
  cmake_parse_arguments(test_args "${options}" "${oneValueArgs}" "${multiValueArgs}" "${ARGN}")

  sanitise_variants(test_args_VARIANTS)
  extract_targets(test_args_VARIANTS TARGETS)

  foreach(VARIANT ${test_args_VARIANTS})
    extract_target(VARIANT TEST_TARGET)
    extract_runconfig(VARIANT TEST_CONFIG)

    set(fork_test "")
    if ("${VARIANT}" STREQUAL "Hw")
      # Make sure IPUs have detached between each pytest
      set(fork_test "--forked")
    endif()

    if (NOT DEFINED test_args_MATCHEXPR)
      set(test_name "${TEST_TARGET}_${TEST_CONFIG}_${name}")
      add_test(NAME "${test_name}"
            COMMAND ${Python3_EXECUTABLE} -m pytest -s ${CMAKE_CURRENT_SOURCE_DIR}/${filename} ${fork_test}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
    else()
      # Note, solid brackets and dash can form part of parameterised pytest names.
      string(REGEX REPLACE " " "_" MATCHEXPR_NAME "${test_args_MATCHEXPR}")
      set(MATCHEXPR_VALUE "${test_args_MATCHEXPR}")
      set(test_name "${TEST_TARGET}_${TEST_CONFIG}_${name}-${MATCHEXPR_NAME}")

      add_test(NAME "${test_name}"
            COMMAND "${Python3_EXECUTABLE}" -m pytest -s "${CMAKE_CURRENT_SOURCE_DIR}/${filename}" -k "${MATCHEXPR_VALUE}" ${fork_test}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
    endif()
    set_tests_properties(${test_name} PROPERTIES
                         ENVIRONMENT TEST_TARGET=${TEST_TARGET})
  endforeach()
endfunction()

add_subdirectory(testutil)
add_subdirectory(popart)
add_subdirectory(torch)
add_subdirectory(onnx_backend)
add_subdirectory(unittests)
add_subdirectory(aapi)

install(DIRECTORY .
	DESTINATION ${INSTALL_TESTS}
	FILES_MATCHING REGEX ".*\.(py|sh|md)$")
