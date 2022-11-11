# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
add_test(NAME "test_popxl_doc_examples"
  COMMAND ${Python3_EXECUTABLE} -m pytest -s "${CMAKE_CURRENT_LIST_DIR}/test_popxl_doc_examples.py" --basetemp=${PYTEST_BASETEMP}/examples_test)
add_test(NAME "test_popxl_doc_examples_custom_op"
  COMMAND ${Python3_EXECUTABLE} -m pytest -s "${CMAKE_CURRENT_LIST_DIR}/test_popxl_doc_examples_custom_op.py" --basetemp=${PYTEST_BASETEMP}/examples_test_custom_op)
# Run custom op test in isolation so we don't end up with two tests racing to
# build the same custom operation.
# TODO T68682 test disabled due to timed out.
set_tests_properties("test_popxl_doc_examples_custom_op" PROPERTIES DISABLED TRUE)

# Add docOnly labels to run these tests when using the `!doc-only` flag in phabricator.
# These, and only tests with these labels will run on CI when the !doc-only flag is used.
# If a new documentation tests is added you must set this label on this test. See T69710.
set_tests_properties("test_popxl_doc_examples" PROPERTIES LABELS "docOnly")
set_tests_properties("test_popxl_doc_examples_custom_op" PROPERTIES LABELS "docOnly")
