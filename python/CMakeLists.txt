# for more useful warnings
include(EnableCompilerWarnings)


# expansion of pybind macro, impossible to declare name 
# of certain classes in headers
add_compile_options("-Wno-missing-prototypes")

include_directories("../willow/include")
include_directories("${PROJECT_BINARY_DIR}/willow/include")

find_package(pybind11 REQUIRED)
find_package(Poplar REQUIRED)

pybind11_add_module(popart_core popart.cpp)

target_include_directories (popart_core SYSTEM PUBLIC
  ${pybind11_INCLUDE_DIR}
)
target_include_directories (popart_core PUBLIC
  ${POPLAR_INCLUDE_DIR}
  ${POPLIBS_INCLUDE_DIR}
)

# Not defining them results in compiler warnings that they will default to 0
target_compile_definitions(popart_core
  PRIVATE PROTOBUF_INLINE_NOT_IN_HEADERS=0
  PRIVATE PYBIND11_HAS_OPTIONAL=0
  PRIVATE PYBIND11_HAS_EXP_OPTIONAL=0
  PRIVATE PYBIND11_HAS_VARIANT=0
#  PRIVATE PYBIND11_PYTHON_VERSION=3.6
)

target_link_libraries(popart_core PRIVATE
  ${POPLAR_LIB}
  popart
  # ...
)

add_custom_command(
  TARGET popart_core
  PRE_BUILD
  COMMAND cmake -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/popart ${CMAKE_CURRENT_BINARY_DIR}/popart
  COMMENT "Copying python module"
)

# This was useful on OS/X for a while, see 
# https://stackoverflow.com/questions/47697761 and
# https://stackoverflow.com/questions/40079424/prepend-to-rpath
# set_target_properties(popart_core PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
#
# but then it stopped working, and this seems to work, 
# see https://gist.github.com/robertmaynard/5750737
if (APPLE)
set_target_properties(popart_core PROPERTIES INSTALL_RPATH "@loader_path/.")
endif()

install(TARGETS popart_core
  DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY
  popart
  DESTINATION ${INSTALL_PYDIR})

# poponnx aliases popart, for backwards compatibility
install(DIRECTORY
  poponnx
  DESTINATION ${INSTALL_PYDIR})

configure_file(__init__.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/popart/__init__.py
  @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/popart/__init__.py
  DESTINATION ${INSTALL_PYDIR}/popart)
