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
  willow
  # ...
)

install(TARGETS pywillow
  # destination : this directory
  DESTINATION ${PYWILLOW_DIR})

