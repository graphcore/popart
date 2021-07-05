# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copy the files in python/popart/python_files
add_custom_target(popart_python_files ALL
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python_files"
    BYPRODUCTS
      "${CMAKE_CURRENT_BINARY_DIR}/popart"
    COMMAND cmake -E copy_directory
      "${CMAKE_CURRENT_SOURCE_DIR}/python_files"
      "${CMAKE_CURRENT_BINARY_DIR}/popart"
  COMMENT "Copying python module")

install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/popart"
    DESTINATION ${INSTALL_PYDIR}
)

configure_file(__init__.py.in
    "${CMAKE_CURRENT_BINARY_DIR}/popart/__init__.py"
    @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/popart/__init__.py"
    DESTINATION "${INSTALL_PYDIR}/popart"
)


foreach(opset IN LISTS POPART_OPSETS)
  # Install stub files for `popart_core` and `popart_opset##`.
  install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${opset}/"
      DESTINATION "${INSTALL_PYDIR}/popart/"
      FILES_MATCHING PATTERN "*.pyi"
  )

  add_subdirectory(${opset})
endforeach()

add_subdirectory(popart_core)
