# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
add_test(NAME "test_popxl_doc_examples"
  COMMAND ${Python3_EXECUTABLE} -m pytest "${CMAKE_CURRENT_LIST_DIR}/test_popxl_doc_examples.py" --basetemp=${PYTEST_BASETEMP}/examples_test)
add_test(NAME "test_popxl_doc_examples_custom_op"
  COMMAND ${Python3_EXECUTABLE} -m pytest "${CMAKE_CURRENT_LIST_DIR}/test_popxl_doc_examples_custom_op.py" --basetemp=${PYTEST_BASETEMP}/examples_test)
# Run custom op test in isolation so we don't end up with two tests racing to
# build the same custom operation.
# T68682 disable test due to timed out.
set_tests_properties("test_popxl_doc_examples_custom_op"  PROPERTIES DISABLED TRUE)
