# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
add_test(NAME "examples_test"
  COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_LIST_DIR}/test_examples.py --basetemp=${PYTEST_BASETEMP}/examples_test)
