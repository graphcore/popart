# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

# Copy the files in python_files
add_custom_target(popart_internal_ir_python_files ALL
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python_files"
    BYPRODUCTS
      "${CMAKE_CURRENT_BINARY_DIR}/../popxl/"
    COMMAND cmake -E copy_directory
      "${CMAKE_CURRENT_SOURCE_DIR}/python_files"
      "${CMAKE_CURRENT_BINARY_DIR}/../popxl/"
  COMMENT "Copying python module")

configure_file(__init__.py.in
    "${CMAKE_CURRENT_BINARY_DIR}/../popxl/__init__.py"
    @ONLY
)

install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/../popxl/"
    DESTINATION "${INSTALL_PYDIR}/popxl/"
    PATTERN "*.cmake" EXCLUDE
    PATTERN "CMakeFiles" EXCLUDE
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/../popxl/__init__.py"
    DESTINATION "${INSTALL_PYDIR}/popxl/"
)
