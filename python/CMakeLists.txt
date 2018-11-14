# for more useful warnings
include(EnableCompilerWarnings)

# expansion of pybind macro, impossible to declare name 
# of certain classes in headers
add_compile_options("-Wno-missing-prototypes")

include_directories("../willow/include")

find_package(pybind11 REQUIRED)

# https://pybind11.readthedocs.io/en/master/faq.html
# suggests not using find_package(PythonLibs), because 
# pybind11 finds Python itself
message(STATUS "Python include dir: " ${PYTHON_INCLUDE_DIR})
message(STATUS "Python library : " ${PYTHON_LIBRARY})
# find_package(PythonLibs)

pybind11_add_module(poponnx_core poponnx.cpp)

target_include_directories (poponnx_core SYSTEM PUBLIC
  ${pybind11_INCLUDE_DIR}
)
target_include_directories (poponnx_core PUBLIC
  # ...
)

# Not defining them results in compiler warnings that they will default to 0
target_compile_definitions(poponnx_core
  PRIVATE PROTOBUF_INLINE_NOT_IN_HEADERS=0
  PRIVATE PYBIND11_HAS_OPTIONAL=0
  PRIVATE PYBIND11_HAS_EXP_OPTIONAL=0
  PRIVATE PYBIND11_HAS_VARIANT=0
)

target_link_libraries(poponnx_core PRIVATE
  ${POPLAR_LIB}
  poponnx
  # ...
)

# This was useful on OS/X for a while, see 
# https://stackoverflow.com/questions/47697761 and
# https://stackoverflow.com/questions/40079424/prepend-to-rpath
# set_target_properties(poponnx_core PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
#
# but then it stopped working, and this seems to work, 
# see https://gist.github.com/robertmaynard/5750737
if (APPLE)
set_target_properties(poponnx_core PROPERTIES INSTALL_RPATH "@loader_path/.")
endif()

install(TARGETS poponnx_core
  DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY
  poponnx
  DESTINATION ${INSTALL_PYDIR})

