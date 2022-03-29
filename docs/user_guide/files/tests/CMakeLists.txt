# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
add_test(NAME "test_popart_doc_examples"
  COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_LIST_DIR}/test_popart_doc_examples.py --basetemp=${PYTEST_BASETEMP}/examples_test)
