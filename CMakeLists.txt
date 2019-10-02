cmake_minimum_required(VERSION 3.4)

project(popart)

enable_testing()

# All C++ code in this project will be compiled as C++14
set (CMAKE_CXX_STANDARD 14)

# colorful ninja, idea from
# medium.com/@alasher/colored-c-compiler-output-with-ninja-clang-gcc-10bfe7f2b949
option (FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." TRUE)
if (${FORCE_COLORED_OUTPUT})
    MESSAGE(STATUS "CMAKE_CXX_COMPILER_ID is \""   ${CMAKE_CXX_COMPILER_ID} "\"")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
       add_compile_options (-fdiagnostics-color=always)
     elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang" OR
             "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" )
       add_compile_options (-fcolor-diagnostics)
    endif ()
endif ()

if(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
  set(OS_TYPE osx)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
  set(OS_TYPE linux)
else()
  set(OS_TYPE unknown)
endif()

# Distribution paths
set(INSTALL_PYDIR python)
set(INSTALL_EXAMPLES examples)

# Tests path
set(INSTALL_TESTS ${CMAKE_INSTALL_PREFIX}/tests)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_INSTALL_RPATH "${POPLAR_INSTALL_DIR}/lib")

#with cmake >= 3.12 this should be add_compile_definitions(ONNX_NAMESPACE=onnx)
add_definitions(-DONNX_NAMESPACE=onnx)

#Where to find Findxxxx.cmake files
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules ${PROJECT_SOURCE_DIR}/cmake)

# Prefer to find static libraries
set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})


find_package(Threads REQUIRED)

find_package(PROTOBUF REQUIRED)

# Using installed cmake find (ONNX must be installed)
# For a hand-crafted Findonnx script, see a very early commit
# or repo popart if available
find_package(ONNX REQUIRED)

# The logging library
find_package(Spdlog REQUIRED)

#Link Boost statically
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Boost REQUIRED COMPONENTS filesystem system program_options)
# find_package(CBLAS REQUIRED)

set(POPART_PYTHON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python)

#------ cpack information
execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../..
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

file(READ ${PROJECT_SOURCE_DIR}/version.json VERSION_JSON)
string(REGEX REPLACE
       ".*major[^:]*: \"([^\"]*)\".*" "\\1"
       CPACK_PACKAGE_VERSION_MAJOR
       ${VERSION_JSON})
string(REGEX REPLACE
       ".*minor[^:]*: \"([^\"]*)\".*" "\\1"
       CPACK_PACKAGE_VERSION_MINOR
       ${VERSION_JSON})
string(REGEX REPLACE
       ".*point[^:]*: \"([^\"]*)\".*" "\\1"
       CPACK_PACKAGE_VERSION_PATCH
       ${VERSION_JSON})

set(VERSION ${CPACK_PACKAGE_VERSION_MAJOR})
set(VERSION ${VERSION}.${CPACK_PACKAGE_VERSION_MINOR})
set(VERSION ${VERSION}.${CPACK_PACKAGE_VERSION_PATCH})

execute_process(COMMAND
  git rev-parse --short=10 HEAD
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE)

add_subdirectory(willow)

# Examples and tests
add_subdirectory(tests)
add_subdirectory(examples)

# light-weight wrapper using pybind11, plus python support libraries
# VERSION needs to be defined before adding this directory
add_subdirectory(python)

if(${OS_TYPE} STREQUAL osx)
  set(PACKAGE_OS_TYPE osx)
elseif(${OS_TYPE} STREQUAL linux)
  # For now we assume we are packaging for ubuntu 18.04, this
  # will have to change at a later date to a be a parameter of the build
  set(PACKAGE_OS_TYPE ubuntu_18_04)
else()
  set(PACKAGE_OS_TYPE unknown)
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

set(PKG_UPLOAD_DIR ../../../pkg)

if(BUILD_DOCS)
add_custom_target(package_and_move
  # CMAKE_BINARY_DIR is not the CWD for this command when it is nested in the popart_super repo
  # Specifying CPackConfig.cmake manually for this reason
  COMMAND ${CMAKE_CPACK_COMMAND} --config ${CMAKE_BINARY_DIR}/CPackConfig.cmake
  COMMAND bash -c "mkdir -p ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "mv ${PACKAGE_FILE_NAME}.tar.gz ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "mv ${PRODINFO_FILE_NAME} ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "mv ${USER_GUIDE_PDF_NAME} ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "mv ${USER_GUIDE_HTML_NAME} ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "mv ${PYTHON_API_PDF_NAME} ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "mv ${PYTHON_API_HTML_NAME} ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "mv ${CPP_API_HTML_NAME} ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "mv ${CPP_API_PDF_NAME} ${PKG_UPLOAD_DIR}"
)
else()
add_custom_target(package_and_move
  # CMAKE_BINARY_DIR is not the CWD for this command when it is nested in the popart_super repo
  # Specifying CPackConfig.cmake manually for this reason
  COMMAND ${CMAKE_CPACK_COMMAND} --config ${CMAKE_BINARY_DIR}/CPackConfig.cmake
  COMMAND bash -c "mkdir -p ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "mv ${PACKAGE_FILE_NAME}.tar.gz ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "mv ${PRODINFO_FILE_NAME} ${PKG_UPLOAD_DIR}"
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
configure_file(enable.sh.in
  ${PROJECT_BINARY_DIR}/enable.sh
  @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/enable.sh
        DESTINATION .)
#------ end of enable script

# Docs
add_subdirectory(docs)

# Copies CTestCustom.ctest to the build dir
configure_file(CTestCustom.ctest
               ${CMAKE_CURRENT_BINARY_DIR})
