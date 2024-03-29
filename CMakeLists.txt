# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
cmake_minimum_required(VERSION 3.12.0)

#------ set version

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/version.json VERSION_JSON)
string(REGEX REPLACE
       ".*major[^:]*: \"([^\"]*)\".*" "\\1"
       POPART_VERSION_MAJOR
       ${VERSION_JSON})
string(REGEX REPLACE
       ".*minor[^:]*: \"([^\"]*)\".*" "\\1"
       POPART_VERSION_MINOR
       ${VERSION_JSON})
string(REGEX REPLACE
       ".*point[^:]*: \"([^\"]*)\".*" "\\1"
       POPART_VERSION_PATCH
       ${VERSION_JSON})

set(POPART_VERSION "${POPART_VERSION_MAJOR}.${POPART_VERSION_MINOR}.${POPART_VERSION_PATCH}")

# Docs version doesn't have the build number or view hash.
set(DOCS_VERSION "${POPART_VERSION}")
set(VERSION "${POPART_VERSION}")
if (DEFINED BUILD_NUMBER)
  # Version used in the file names of the buildbot's package will include build number.
  set(VERSION "${POPART_VERSION}+${BUILD_NUMBER}")
endif()

#------ end of set version

### Project declaration
project(popart VERSION ${POPART_VERSION} LANGUAGES CXX)

# Where to find Findxxxx.cmake files
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules ${PROJECT_SOURCE_DIR}/cmake)

# cmake option definitions for popart
include(DefineOptions)

# Calls enable_testing(), which sets BUILD_TESTING=ON
include(CTest)

#------ Set install directories

include(GNUInstallDirs)

# Distribution path
set(INSTALL_PYDIR python)

# Tests path
set(INSTALL_TESTS ${CMAKE_INSTALL_PREFIX}/tests)

#------ end of install directories

#------ Definitions that configure the build

# Don't allow C++ extensions.
set (CMAKE_CXX_EXTENSIONS OFF)

# Add Poplar SDK dir to prefix path so we can find its libraries.
if(POPLAR_INSTALL_DIR)
  message(VERBOSE "Appending to CMAKE_PREFIX_PATH: POPLAR_INSTALL_DIR = ${POPLAR_INSTALL_DIR}")
  list(APPEND CMAKE_PREFIX_PATH ${POPLAR_INSTALL_DIR})
endif()

# Hardcode rpaths to things we link against outside of popart's build tree.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_MACOSX_RPATH 1)
if (POPLAR_INSTALL_DIR)
  set(CMAKE_INSTALL_RPATH "${POPLAR_INSTALL_DIR}/lib")
endif()

# Prefer to find static libraries
set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#------ end of build configuration

#------ find dependencies
# Versions are not given where packages don't support them.

# Use the first Python installation found.
set(Python_FIND_STRATEGY LOCATION)
# Always search for the virtualenv Python first.
set(Python_FIND_VIRTUALENV FIRST)
find_package(Python3 3.6 REQUIRED COMPONENTS Interpreter Development)

if(NOT BUILD_SHARED_LIBS)
  if(NOT DEFINED Boost_USE_STATIC_LIBS)
    set(Boost_USE_STATIC_LIBS ON)
  endif()
  if(NOT DEFINED Boost_USE_STATIC_RUNTIME)
    set(Boost_USE_STATIC_RUNTIME ON)
  endif()
endif()
set(boost_components
    # FindBoost bug: finding filesystem component does not transitively find and
    # link the system component.
    system
    filesystem
    graph
    random
)
find_package(Boost 1.70 REQUIRED COMPONENTS ${boost_components})

list(TRANSFORM boost_components PREPEND Boost:: OUTPUT_VARIABLE boost_targets)

# The header-only component. Many of the libs are header-only, and come under
# this target. You do not pass it as a COMPONENT; it will always be created.
list(APPEND boost_targets Boost::boost)

find_package(spdlog 1.8.0 REQUIRED)

find_package(popef REQUIRED)
find_package(poplar REQUIRED)
find_package(poplibs REQUIRED)
find_package(poprithms REQUIRED)
find_package(popdist REQUIRED)
find_package(libpvti REQUIRED)
find_package(pva REQUIRED)
find_package(gcl REQUIRED)

find_package(poprithms REQUIRED)

find_package(gccs REQUIRED)
include(${GCCS_CMAKE_DIR}/EnableCompilerWarnings.cmake)
enable_compiler_warnings()

# Onnx's config file fails to find protobuf, so we have to do it ourselves.
# We use the find module provided by CMake.
set(Protobuf_USE_STATIC_LIBS ON)
find_package(Protobuf 3.6 REQUIRED)
find_package(ONNX 1.6 REQUIRED)

# Load custom utilities.
include(cmake/Utils.cmake)

#------ end of find dependencies

#------ configure sources

execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY "${POPART_CBT_VIEW_DIR}"
  OUTPUT_VARIABLE VIEW_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if("${VIEW_HASH}" STREQUAL "")
  set(VIEW_SHORT_HASH "0000000000")
else()
  string(SUBSTRING ${VIEW_HASH} 0 10 VIEW_SHORT_HASH)
endif()
message(STATUS "View hash is " ${VIEW_SHORT_HASH})

set(SNAPSHOT ${VIEW_SHORT_HASH})

# Store a list of the available ONNX opset versions. This is used in the docs/
# and python/ subdirs.
execute_process(
  COMMAND bash -c "find . -regex \"^./popart_opset[0-9]+$\" -type d"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/popart/
  OUTPUT_VARIABLE POPART_OPSETS
)
string(REPLACE "./" "" POPART_OPSETS ${POPART_OPSETS})
string(REPLACE "\n" ";" POPART_OPSETS ${POPART_OPSETS})
list(REMOVE_ITEM POPART_OPSETS "")

add_subdirectory(willow)

# light-weight wrapper using pybind11, plus python support libraries
# VERSION needs to be defined before adding this directory
add_subdirectory(python)

# Examples and tests
if (POPART_BUILD_TESTING AND BUILD_TESTING)
  add_subdirectory(tests)
endif()

#------ end of configure sources

#------ cpack information

set(CPACK_PACKAGE_VENDOR graphcore)
set(CPACK_GENERATOR TGZ)
set(CPACK_PACKAGE_VERSION_MAJOR ${POPART_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${POPART_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${POPART_VERSION_PATCH})

# Set OS_TYPE (used later in prodinfo yml).
if(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
  set(OS_TYPE osx)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
  set(OS_TYPE linux)
else()
  set(OS_TYPE unknown)
endif()

# Set PACKAGE_OS_TYPE used in CPack package.
if(${OS_TYPE} STREQUAL osx)
  set(PACKAGE_OS_TYPE osx)
elseif(${OS_TYPE} STREQUAL linux)
  # For now we assume we are packaging for ubuntu 18.04, this
  # will have to change at a later date to a be a parameter of the build
  set(PACKAGE_OS_TYPE ubuntu_18_04)
else()
  set(PACKAGE_OS_TYPE unknown)
endif()

# SWDB_PACKAGE_NAME is used to set the name of the main poplar package,
# e.g. ubuntu_16_04_installer
if (DEFINED SWDB_PACKAGE_NAME)
  set(PACKAGE_OS_TYPE ${SWDB_PACKAGE_NAME})
  string(REPLACE "_installer" "" PACKAGE_OS_TYPE ${PACKAGE_OS_TYPE})
  message(STATUS "Overriding package type '${PACKAGE_OS_TYPE}'")
endif()

set(PACKAGE_FILE_NAME popart-${PACKAGE_OS_TYPE}-${VERSION}-${SNAPSHOT})
set(CPACK_PACKAGE_FILE_NAME ${PACKAGE_FILE_NAME})
set(CPACK_PACKAGING_INSTALL_PREFIX "/${PACKAGE_FILE_NAME}")
set(CPACK_ARCHIVE_UNSPECIFIED_FILE_NAME ${PACKAGE_FILE_NAME})
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON)
get_cmake_property(CPACK_COMPONENTS_ALL COMPONENTS)
list(REMOVE_ITEM CPACK_COMPONENTS_ALL "popart-docs")

set(SUPPORTED_POPART_OPS_GEN_FILE_NAME popart_supported_ops_gen-${VERSION}-${SNAPSHOT}.rst)
set(SUPPORTED_POPXL_OPS_GEN_FILE_NAME popxl_supported_ops_gen-${VERSION}-${SNAPSHOT}.rst)
set(POPART_USER_GUIDE_PDF_NAME popart-user-guide-${VERSION}-${SNAPSHOT}.pdf)
set(POPART_USER_GUIDE_HTML_NAME popart-user-guide-html-${VERSION}-${SNAPSHOT}.zip)
set(PRODINFO_FILE_NAME popart-${VERSION}-${SNAPSHOT}.yml)
set(POPXL_USER_GUIDE_PDF_NAME popxl-user-guide-${VERSION}-${SNAPSHOT}.pdf)
set(POPXL_USER_GUIDE_HTML_NAME popxl-user-guide-html-${VERSION}-${SNAPSHOT}.zip)

include(CPack)

if(BUILD_DOCS)
add_custom_target(package_and_move
  # CMAKE_BINARY_DIR is not the CWD for this command when it is nested in the popart_super repo
  # Specifying CPackConfig.cmake manually for this reason
  COMMAND ${CMAKE_CPACK_COMMAND} --config ${CMAKE_BINARY_DIR}/CPackConfig.cmake
  COMMAND ${CMAKE_COMMAND} -E echo "Moved package to POPART_PKG_DIR = ${POPART_PKG_DIR}"
  COMMAND bash -c "mkdir -p ${POPART_PKG_DIR}"
  COMMAND bash -c "mv ${PACKAGE_FILE_NAME}.tar.gz ${POPART_PKG_DIR}"
  COMMAND bash -c "mv ${PRODINFO_FILE_NAME} ${POPART_PKG_DIR}"
  COMMAND bash -c "mv ${SUPPORTED_POPART_OPS_GEN_FILE_NAME} ${POPART_PKG_DIR}"
  COMMAND bash -c "mv ${SUPPORTED_POPXL_OPS_GEN_FILE_NAME} ${POPART_PKG_DIR}"
  COMMAND bash -c "mv ${POPART_USER_GUIDE_PDF_NAME} ${POPART_PKG_DIR}"
  COMMAND bash -c "mv ${POPART_USER_GUIDE_HTML_NAME} ${POPART_PKG_DIR}"
  COMMAND bash -c "mv ${POPXL_USER_GUIDE_PDF_NAME} ${POPART_PKG_DIR}"
  COMMAND bash -c "mv ${POPXL_USER_GUIDE_HTML_NAME} ${POPART_PKG_DIR}"
  )
else()
add_custom_target(package_and_move
  # CMAKE_BINARY_DIR is not the CWD for this command when it is nested in the popart_super repo
  # Specifying CPackConfig.cmake manually for this reason
  COMMAND ${CMAKE_CPACK_COMMAND} --config ${CMAKE_BINARY_DIR}/CPackConfig.cmake
  COMMAND ${CMAKE_COMMAND} -E echo "Moved package to POPART_PKG_DIR = ${POPART_PKG_DIR}"
  COMMAND bash -c "mkdir -p ${POPART_PKG_DIR}"
  COMMAND bash -c "mv ${PACKAGE_FILE_NAME}.tar.gz ${POPART_PKG_DIR}"
  COMMAND bash -c "mv ${PRODINFO_FILE_NAME} ${POPART_PKG_DIR}"
)
endif()
#------ end of cpack information

#------ product info file
if(BUILD_DOCS)
  # Upload supported_ops to Artifactory so it can be used to builed the published docs
  string(APPEND PKG_ELEMENTS "  - name: Poplar advanced runtime supported ops (RST)\n"
                             "    file: ${SUPPORTED_POPART_OPS_GEN_FILE_NAME}\n"
                             "    type: rst\n")

  string(APPEND PKG_ELEMENTS "  - name: PopXL supported ops (RST)\n"
                             "    file: ${SUPPORTED_POPXL_OPS_GEN_FILE_NAME}\n"
                             "    type: rst\n")

  string(APPEND PKG_ELEMENTS "  - name: Poplar advanced runtime user guide (PDF)\n"
                             "    file: ${POPART_USER_GUIDE_PDF_NAME}\n"
                             "    type: pdf\n")

  string(APPEND PKG_ELEMENTS "  - name: Poplar advanced runtime user guide (HTML)\n"
                             "    file: ${POPART_USER_GUIDE_HTML_NAME}\n"
                             "    type: html_zip\n")

  string(APPEND PKG_ELEMENTS "  - name: PopXL user guide (PDF)\n"
                             "    file: ${POPXL_USER_GUIDE_PDF_NAME}\n"
                             "    type: pdf\n")

  string(APPEND PKG_ELEMENTS "  - name: PopXL user guide (HTML)\n"
                             "    file: ${POPXL_USER_GUIDE_HTML_NAME}\n"
                             "    type: html_zip\n")

endif()

execute_process(
  COMMAND git config --get remote.origin.url
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
  OUTPUT_VARIABLE VIEW_REMOTE_URL
  OUTPUT_STRIP_TRAILING_WHITESPACE)

configure_file(popart.prodinfo.yml
  ${CMAKE_CURRENT_BINARY_DIR}/${PRODINFO_FILE_NAME}
  @ONLY)
#------ end of product info file

#------ enable script
if(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
  set(LD_PATH_VARNAME "DYLD_LIBRARY_PATH")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
  set(LD_PATH_VARNAME "LD_LIBRARY_PATH")
else()
  message(FATAL_ERROR "Unknown system name")
endif()

# Create enable.sh for Bash
set(SHEBANG "#!/bin/bash")
set(SCRIPT_PATH "\${BASH_SOURCE[0]}")
configure_file(enable-shell.in enable.sh @ONLY)

# Create enable.zsh for Zsh
set(SHEBANG "#!/bin/zsh")
set(SCRIPT_PATH "\${(%):-%x}")
configure_file(enable-shell.in enable.zsh @ONLY)

# Install all enable files
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/enable.sh
              ${CMAKE_CURRENT_BINARY_DIR}/enable.zsh
        DESTINATION .)
#------ end of enable script

# Docs

# There are tests for the python examples which need copying
install(DIRECTORY docs/
  DESTINATION docs COMPONENT popart-docs
  FILES_MATCHING REGEX ".*\.(py)$")

add_subdirectory(docs)
