# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

# Add a Python unit test.
#
# Examples:
#
# To add a python unit test specify the single test file after the name:
#
# add_python_unit_test(my_python_test my_python_test.py VARIANTS Hw Sim)
#
# Options:
#
# ~~~
#   VARIANTS
#     A list of strings that specifies the targets to run the test on. If a variant is not
#     in ENABLED_TEST_VARIANTS then it will be ignored. If VARIANTS is not specified then
#     it defaults to DEFAULT_TEST_VARIANTS. If VARIANTS is NoTarget then the
#     --device-type option will not be passed to the test.
#   LABELS
#     Any additional labels you want to add to a test: set_tests_properties(test PROPERTIES LABELS ...)
#     Note that the multicard label is used by CI to identify tests that require > 1 IPUs. This can
#     also be used to segment out particular test, for example all python tests are tagged with a "python"
#     label, so running `ctest -L python` will run just python tests.
#     See https://cmake.org/cmake/help/latest/prop_dir/LABELS.html
#   ENV
#     The environment variables to use when running the test. This can be used to define
#     environment variables specific to your test. For environment variables to use when
#     running all tests it's simplest to append them to the TEST_ENVIRONMENT variable.
#   PROPERTIES
#     A list of properties to pass to set_test_properties(test PROPERTIES ...).
#     See https://cmake.org/cmake/help/latest/command/set_tests_properties.html
#   TEST_IPUMODEL
#     If set to TRUE allows to tests on IpuModel1, IpuModel2 and/or IpuModel21 even when
#     Sim1, Sim2 and/or Sim2 variants are specified. Default behavioural is FALSE and was
#     chosen to reduce test time.
#   MATCHEXPR
#     a string that will be passed verbatim to pytest as a selector when running the test
#     `-k <matchexpr_value>`. This allows for selecting specific tests within a file, useful
#     when splitting long tests into parts.
#   NUM_WORKERS
#     A numeric value or be the string `auto`, in which case the number of launched
#     workers is equal to the number of available CPUs. In order to avoid
#     launching too many threads at once tests that use the NUM_WORKERS option are run
#     sequentially after any other tests have run (so the pytest subtests run in parallel,
#     but the individual ctest tests run sequentially).
#   DEPENDS
#     When specifiying FILES (and not COMMAND) add the contents of DEPENDS as dependencies
#     of the test executable created using FILES. This is useful if your test executable
#     relies on something not covered by the add_test_executable function.
# ~~~
function(add_python_unit_test name file)
  set(options ALLOW_SKIP TEST_IPUMODEL)
  set(one_value_args MATCHEXPR NUM_WORKERS)
  set(multi_value_args VARIANTS LABELS DEPENDS ENV PROPERTIES)

  cmake_parse_arguments(ARGS "${options}" "${one_value_args}"
                        "${multi_value_args}" "${ARGN}")

  if(ARGS_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unparsed arguments: ${ARGS_UNPARSED_ARGUMENTS}")
  endif()

  sanitise_variants(ARGS_VARIANTS ARGS_TEST_IPUMODEL ARGS_LABELS)

  if(NOT ARGS_VARIANTS)
    # When all variants are disabled don't even add the test. Note this only
    # occurs when no variants are enabled, otherwise sanitise_variants will set
    # the VARIANTS to DEFAULT_TEST_VARIANTS.
    if(DEBUG_TEST_GENERATION)
      message(STATUS "${CMAKE_CURRENT_FUNCTION}: Skipping test ${name}
              because no VARIANTS enabled.")
    endif()
    return()
  endif()

  if(DEBUG_TEST_GENERATION)
    message(
      STATUS
        "${CMAKE_CURRENT_FUNCTION}: Adding test '${name}' with variants: ${ARGS_VARIANTS}"
    )
  endif()

  extract_targets(ARGS_VARIANTS TARGETS)

  set(pytest_arguments -s "${file}")

  if(DEFINED ARGS_NUM_WORKERS)
    # Set number of threads to run test with
    list(APPEND pytest_arguments -n "${ARGS_NUM_WORKERS}")
  endif()

  add_parent_dir_labels(ARGS_LABELS)

  list(APPEND ARGS_LABELS "python")

  foreach(VARIANT ${ARGS_VARIANTS})
    extract_target(VARIANT test_target)
    extract_runconfig(VARIANT test_config)

    set_test_target(test_target_arg ${test_target})

    list(APPEND pytest_arguments ${test_target_arg})

    if(NOT DEFINED ARGS_MATCHEXPR)
      string(JOIN "_" test_name ${test_target} ${test_config} ${name})
    else()
      # Note, solid brackets and dash can form part of parameterised pytest
      # names.
      string(REGEX REPLACE " " "_" MATCHEXPR_NAME "${ARGS_MATCHEXPR}")
      set(MATCHEXPR_VALUE "${ARGS_MATCHEXPR}")
      set(test_name "${test_target}_${test_config}_${name}-${MATCHEXPR_NAME}")
      list(APPEND pytest_arguments -k "${MATCHEXPR_VALUE}")
    endif()

    # Adding ${test_name} after PYTEST_BASETEMP because when running tests in
    # parallel, pytest removes the directories of running tests. This is because
    # by default, pytest removes entries in basetemp older than the newest 3
    # entries.
    file(MAKE_DIRECTORY "${PYTEST_BASETEMP}/${test_name}")
    list(APPEND pytest_arguments --basetemp "${PYTEST_BASETEMP}/${test_name}")

    add_test(
    NAME ${test_name}
    COMMAND ${Python3_EXECUTABLE} -m pytest ${pytest_arguments}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    if(DEFINED ARGS_NUM_WORKERS)
      # Run this test serially, so that it doesn't starve other tests
      set_property(TEST "${test_name}" PROPERTY RUN_SERIAL TRUE)
    endif()

    if(ARGS_PROPERTIES)
      set_tests_properties(${test_name} PROPERTIES ${ARGS_PROPERTIES})
    endif()

    set(test_env ${TEST_ENVIRONMENT} ${ARGS_ENV})

    set_tests_properties(${test_name} PROPERTIES ENVIRONMENT "${test_env}")

    if(ARGS_ALLOW_SKIP)
      set_tests_properties(${test_name} PROPERTIES SKIP_RETURN_CODE
                                                   ${TEST_SKIP_RETURN_CODE})
    endif()

    if(ARGS_LABELS)
      set_tests_properties(${test_name} PROPERTIES LABELS "${ARGS_LABELS}")
    endif()

    add_target_available_check(${test_target} ${test_name})

  endforeach()
endfunction()
