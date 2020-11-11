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
set(INSTALL_EXAMPLES examples)

# Tests path
set(INSTALL_TESTS ${CMAKE_INSTALL_PREFIX}/tests)

#------ end of install directories

#------ Definitions that configure the build

# All C++ code in this project will be compiled as C++14
set (CMAKE_CXX_STANDARD 14)

# Hardcode rpaths to things we link against outside of popart's build tree.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_INSTALL_RPATH "${POPLAR_INSTALL_DIR}/lib")

#with cmake >= 3.12 this should be add_compile_definitions(ONNX_NAMESPACE=onnx)
add_definitions(-DONNX_NAMESPACE=onnx)

# This allows us to deprecate the legacy op factory functions
# in our own code without breaking customers code.
add_definitions(-DDEPRECATE_LEGACY_OP_FACTORY=1)

# Prefer to find static libraries
set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#------ end of build configuration

#------ find dependencies

find_package(PoplarRunner)
find_package(Threads REQUIRED)
find_package(PROTOBUF REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

if (PoplarRunner_FOUND)
  add_definitions(-DPOPLAR_RUNNER)
endif()

# Using installed cmake find (ONNX must be installed)
# For a hand-crafted Findonnx script, see a very early commit
# or repo popart if available
find_package(ONNX REQUIRED)

#Link Boost statically
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)
find_package(Boost REQUIRED COMPONENTS filesystem system program_options)

#------ end of find dependencies

#------ configure sources

set(POPART_PYTHON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python)

add_subdirectory(willow)

# light-weight wrapper using pybind11, plus python support libraries
# VERSION needs to be defined before adding this directory
add_subdirectory(python)

# Examples and tests
if (POPART_BUILD_TESTING AND BUILD_TESTING)
  add_subdirectory(tests)
endif()
if (POPART_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

#------ end of configure sources

#------ cpack information

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

set(USER_GUIDE_PDF_NAME popart-user-guide-${VERSION}-${SNAPSHOT}.pdf)
set(USER_GUIDE_HTML_NAME popart-user-guide-html-${VERSION}-${SNAPSHOT}.zip)
set(PYTHON_API_PDF_NAME popart-python-api-${VERSION}-${SNAPSHOT}.pdf)
set(PYTHON_API_HTML_NAME popart-python-api-html-${VERSION}-${SNAPSHOT}.zip)
set(CPP_API_PDF_NAME popart-cpp-api-${VERSION}-${SNAPSHOT}.pdf)
set(CPP_API_HTML_NAME popart-cpp-api-html-${VERSION}-${SNAPSHOT}.zip)
set(PRODINFO_FILE_NAME popart-${VERSION}-${SNAPSHOT}.yml)

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
  COMMAND bash -c "mv ${USER_GUIDE_PDF_NAME} ${POPART_PKG_DIR}"
  COMMAND bash -c "mv ${USER_GUIDE_HTML_NAME} ${POPART_PKG_DIR}"
  COMMAND bash -c "mv ${PYTHON_API_PDF_NAME} ${POPART_PKG_DIR}"
  COMMAND bash -c "mv ${PYTHON_API_HTML_NAME} ${POPART_PKG_DIR}"
  COMMAND bash -c "mv ${CPP_API_HTML_NAME} ${POPART_PKG_DIR}"
  COMMAND bash -c "mv ${CPP_API_PDF_NAME} ${POPART_PKG_DIR}"
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
  string(APPEND PKG_ELEMENTS "  - name: Poplar advanced runtime user guide (PDF)\n"
                             "    file: ${USER_GUIDE_PDF_NAME}\n"
                             "    type: pdf\n")

  string(APPEND PKG_ELEMENTS "  - name: Poplar advanced runtime user guide (HTML)\n"
                             "    file: ${USER_GUIDE_HTML_NAME}\n"
                             "    type: html_zip\n")

  string(APPEND PKG_ELEMENTS "  - name: Poplar advanced runtime Python API (PDF)\n"
                             "    file: ${PYTHON_API_PDF_NAME}\n"
                             "    type: pdf\n")

  string(APPEND PKG_ELEMENTS "  - name: Poplar advanced runtime Python API (HTML)\n"
                             "    file: ${PYTHON_API_HTML_NAME}\n"
                             "    type: html_zip\n")

  string(APPEND PKG_ELEMENTS "  - name: Poplar advanced runtime C++ API (PDF)\n"
                             "    file: ${CPP_API_PDF_NAME}\n"
                             "    type: pdf\n")

  string(APPEND PKG_ELEMENTS "  - name: Poplar advanced runtime C++ API (HTML)\n"
                             "    file: ${CPP_API_HTML_NAME}\n"
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

configure_file(enable.sh.in
  ${PROJECT_BINARY_DIR}/enable.sh
  @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/enable.sh
        DESTINATION .)
#------ end of enable script

# Docs

# There are tests for the python examples which need copying
install(DIRECTORY docs/
  DESTINATION docs
  FILES_MATCHING REGEX ".*\.(py)$")

add_subdirectory(docs)
