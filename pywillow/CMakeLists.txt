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

# This was useful on OS/X for a while, see 
# https://stackoverflow.com/questions/47697761 and
# https://stackoverflow.com/questions/40079424/prepend-to-rpath
# set_target_properties(pywillow PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
#
# but then it stopped working, and this seems to work, 
# see https://gist.github.com/robertmaynard/5750737
set_target_properties(pywillow PROPERTIES INSTALL_RPATH "@loader_path/.")

install(TARGETS pywillow
  DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES
  writer.py
  torchwriter.py
  DESTINATION ${INSTALL_PYDIR})

