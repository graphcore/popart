# for more useful warnings
include(EnableCompilerWarnings)

cmake_policy(SET CMP0079 NEW)

# expansion of pybind macro, impossible to declare name 
# of certain classes in headers

# See pybind11's installed cmake files for more documentation.
# We tell it what cpp std modules should be compiled with.
# Note that pybind11 will use the python targets found by the
# find_package(Python3 ...) command.
set(PYBIND11_CPP_STANDARD -std=c++14)
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")

# Create a convenience target that depends on the whole python API, so
# consumers of the API can link just one target instead of all of them.
add_library(popart-py INTERFACE)

function(add_popart_python_module module_name)
  # Newer versions of Pybind11 use CMake targets and CMake considers imported
  # targets as SYSTEM by default so they removed the SYSTEM option.
  if(pybind11_VERSION VERSION_LESS "2.6.0")
    set(OLD_PYBIND11_SYSTEM_FLAG SYSTEM)
  endif()

  message("ARGN=${ARGN}")

  # Creates a module library.
  pybind11_add_module(${module_name}
      # Use -isystem for pybind includes
      ${OLD_PYBIND11_SYSTEM_FLAG}
      # Use ThinLTO if available
      THIN_LTO
      # Sources
    ${ARGN})

  target_compile_features(${module_name} PUBLIC cxx_std_14)

  target_link_libraries(${module_name} PRIVATE popart)

  # For some unknown reason, you need the latter command to get
  # add_custom_command(DEPENDS popart-py ...) to work.
  target_link_libraries(popart-py INTERFACE ${module_name})
  add_dependencies(popart-py ${module_name})

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
      DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endfunction()

add_subdirectory(popart)
