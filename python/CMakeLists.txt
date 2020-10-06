# for more useful warnings
include(EnableCompilerWarnings)


# expansion of pybind macro, impossible to declare name 
# of certain classes in headers

include_directories("../willow/include")
include_directories("${PROJECT_BINARY_DIR}/willow/include")

find_package(pybind11 REQUIRED)
find_package(Poplar REQUIRED)

function(add_popart_python_module module_name source_file)
  pybind11_add_module(${module_name} ${source_file} np_utils.cpp)

  target_include_directories (${module_name} SYSTEM PUBLIC
    ${pybind11_INCLUDE_DIR}
  )
  target_include_directories (${module_name} PUBLIC
    ${POPLAR_INCLUDE_DIR}
    ${POPLIBS_INCLUDE_DIR}
    ${LIBPVTI_INCLUDE_DIR}
  )

  # Not defining them results in compiler warnings that they will default to 0
  target_compile_definitions(${module_name}
    PRIVATE PROTOBUF_INLINE_NOT_IN_HEADERS=0
    PRIVATE PYBIND11_HAS_OPTIONAL=0
    PRIVATE PYBIND11_HAS_EXP_OPTIONAL=0
    PRIVATE PYBIND11_HAS_VARIANT=0
  #  PRIVATE PYBIND11_PYTHON_VERSION=3.6
  )

  target_link_libraries(${module_name} PRIVATE
    ${POPLAR_LIB}
    popart
    # ...
  )



  # This was useful on OS/X for a while, see 
  # https://stackoverflow.com/questions/47697761 and
  # https://stackoverflow.com/questions/40079424/prepend-to-rpath
  # set_target_properties(popart_core PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
  #
  # but then it stopped working, and this seems to work, 
  # see https://gist.github.com/robertmaynard/5750737
  if (APPLE)
  set_target_properties(${module_name} PROPERTIES INSTALL_RPATH "@loader_path/.")
  endif()

  install(TARGETS ${module_name}
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
endfunction()

add_popart_python_module(popart_core popart.cpp)
add_popart_python_module(popart_opset6 popart_opset6.gen.cpp)
add_popart_python_module(popart_opset7 popart_opset7.gen.cpp)
add_popart_python_module(popart_opset8 popart_opset8.gen.cpp)
add_popart_python_module(popart_opset9 popart_opset9.gen.cpp)
add_popart_python_module(popart_opset10 popart_opset10.gen.cpp)
add_popart_python_module(popart_opset11 popart_opset11.gen.cpp)

# Copy the files in popart/popart/python/popart.
add_custom_command(
  TARGET popart_core
  PRE_BUILD
  COMMAND cmake -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/popart ${CMAKE_CURRENT_BINARY_DIR}/popart
  COMMENT "Copying python module"
)

install(DIRECTORY
  popart
  DESTINATION ${INSTALL_PYDIR})

configure_file(__init__.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/popart/__init__.py
  @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/popart/__init__.py
  DESTINATION ${INSTALL_PYDIR}/popart)
