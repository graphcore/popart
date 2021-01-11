
# Other projects in the view (in particular Boost) will not get built with fPIC
# unless we set this here. (CBT causes projects to inherit certain cache entires
# including this one, then boost_scripts will add fPIC to the boost cxx flags if
# it is set)
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL
  "Default value for POSITION_INDEPENDENT_CODE of targets.")

#
# Usage:
#   _popart_pass_through_cache_var(<var> [PATH])
#
# Implicitly passes through a variable to Popart that has been defined directly
# on the super project, like -Dvar=val, instead of explicitly like
# -DPOPART_CMAKE_ARGS=-Dvar=val.
#
# In general, for a path, an absolute path should be given, but for backwards
# compatibility we support paths relative to the view build dir. Popart itself
# (not the superproject, which this file is part of) documents that it expects
# an absolute path; thus if you pass a relative path in POPART_CMAKE_ARGS, it
# will not get evalulated correctly.
#
# If user for some reason defined both directly and in POPART_CMAKE_ARGS, the
# definition in POPART_CMAKE_ARGS will get overwritten with this definition.
macro(_popart_pass_through_cache_var var is_path)
  if(DEFINED ${var})   
    set(_val "${${var}}")

    if(is_path STREQUAL "PATH")
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
