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

pybind11_add_module(pywillow pywillow.cpp)

target_include_directories (pywillow PUBLIC 
  # ...
)

target_link_libraries(pywillow PRIVATE
  ${POPLAR_LIB}
  willow
  # ...
)

# This is useful on OS/X, see 
# https://stackoverflow.com/questions/47697761 and
# https://stackoverflow.com/questions/40079424/prepend-to-rpath
set_target_properties(pywillow PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS pywillow
  DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES
  writer.py
  torchwriter.py
  DESTINATION ${CMAKE_INSTALL_PYDIR})

