#[[
  SuperProjectConfig.cmake

  ------------------------------------------------------------------------------
  This file was needed when Popart used to live in its own view. It is now
  unnecessary. Please do not add to it.
  ------------------------------------------------------------------------------

  A SuperProjectConfig.cmake file is used to configure the view itself, not 
  Popart. It is run as part of the CBT super-project. It is not run as part of
  Popart's own build.

  poplar_packaging already has a SuperProjectConfig.cmake that configures the
  Poplar view. We should keep all view-wide configuration to that file only, NOT
  HERE. For example, setting CMAKE_POSITION_INDEPEDENT_CODE for the whole view
  should be done there only.

  VERY IMPORTANT:   DO NOT DEFINE NEW POPART CMAKE VARIABLES HERE.

  Popart CMake variables should be defined in Popart's actual CMake files, not
  this file. It is correct to force the user to configure the view like
  -DPOPART_CMAKE_ARGS="-DSOME_ARG=1", instead of -DSOME_ARG=1 (expecting that
  this file will pass through the variable to Popart itself). 

  The below code for enabling this is for backwards compatability only as this
  had already been done, causing bugs - see T32416.

  From a design perspective, Popart is its own individual project, so its
  variables should be defined in Popart, not as part of the view. Popart should
  build just fine outside of the view.
]]

#[[
  macro(_popart_pass_through_cache_var var is_path)

  Usage:
    _popart_pass_through_cache_var(<var> [PATH])
 
  Implicitly passes through a variable to Popart that has been defined directly
  on the super project, like -Dvar=val, instead of explicitly like
  -DPOPART_CMAKE_ARGS=-Dvar=val.
 
  In general, for a path, an absolute path should be given, but for backwards
  compatibility we support paths relative to the view build dir. Popart itself
  (not the superproject, which this file is part of) documents that it expects
  an absolute path; thus if you pass a relative path in POPART_CMAKE_ARGS, it
  will not get evalulated correctly.
 
  If user for some reason defined both directly and in POPART_CMAKE_ARGS, the
  definition in POPART_CMAKE_ARGS will get overwritten with this definition.
]]
macro(_popart_pass_through_cache_var var is_path)
  if(DEFINED ${var})   
    set(_val "${${var}}")

    if("${is_path}" STREQUAL "PATH")
      get_filename_component(
          _val "${_val}"
          ABSOLUTE
          BASE_DIR "${CMAKE_BINARY_DIR}"
      )
    endif()

    list(APPEND POPART_CMAKE_ARGS -D${var}=${_val})

    unset(_val)
  endif()
endmacro()

# Automatically set for the user to the VIEW_DIR var provided by CBT.
get_filename_component(
    POPART_CBT_VIEW_DIR "${VIEW_DIR}"
    ABSOLUTE
    BASE_DIR "${CMAKE_BINARY_DIR}"
)

_popart_pass_through_cache_var(POPLAR_INSTALL_DIR PATH)
_popart_pass_through_cache_var(PoplarRunner_INSTALL_DIR PATH)
_popart_pass_through_cache_var(C10_DIR PATH)
_popart_pass_through_cache_var(POPART_PKG_DIR PATH)
_popart_pass_through_cache_var(POPART_CBT_VIEW_DIR PATH)

# TODO(T34104): Move this to poplar_packaging/SuperProjectConfig.cmake
set(CMAKE_CONFIGURATION_TYPES "Release" "Debug" "MinSizeRel" "RelWithDebInfo")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
# Enable IN_LIST operator
cmake_policy(SET CMP0057 NEW)
if(NOT CMAKE_BUILD_TYPE)
  list(GET CMAKE_CONFIGURATION_TYPES 0 CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified")
endif()
if(NOT CMAKE_BUILD_TYPE IN_LIST CMAKE_CONFIGURATION_TYPES)
  message(FATAL_ERROR "CMAKE_BUILD_TYPE must be one of ${CMAKE_CONFIGURATION_TYPES}")
endif()
