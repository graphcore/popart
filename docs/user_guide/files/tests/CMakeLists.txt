add_test(NAME "examples_test"
  COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_LIST_DIR}/test_examples.py)
