cmake_minimum_required(VERSION 3.4)

project(poponnx)

enable_testing()

# All C++ code in this project will be compiled as C++11
add_definitions(-std=c++11)

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

find_package(Threads REQUIRED)

# There is a FindProtobuf.cmake file on my computer at
# /usr/local/share/cmake/Modules/FindProtobuf.cmake
find_package(Protobuf REQUIRED) 

# Using installed cmake find (ONNX must be installed)
# For a hand-crafted Findonnx script, see a very early commit
# or repo poponnx if available
find_package(ONNX REQUIRED)

# The logging library
find_package(Spdlog REQUIRED)

#Link Boost statically
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Boost REQUIRED COMPONENTS filesystem system)
# find_package(CBLAS REQUIRED)

set(POPONNX_PYTHON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python)
add_subdirectory(willow)

# light-weight wrapper using pybind11, plus python support libraries
add_subdirectory(python)

# Examples and tests
add_subdirectory(tests)
add_subdirectory(examples)

#------ cpack information
execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
  OUTPUT_VARIABLE VIEW_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE)
string(SUBSTRING ${VIEW_HASH} 0 10 VIEW_SHORT_HASH)

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

if(${OS_TYPE} STREQUAL osx)
  set(PACKAGE_OS_TYPE osx)
elseif(${OS_TYPE} STREQUAL linux)
  # For now we assume we are packaging for ubuntu 18.04, this
  # will have to change at a later date to a be a parameter of the build
  set(PACKAGE_OS_TYPE ubuntu_18_04)
else()
  set(PACKAGE_OS_TYPE unknown)
endif()


set(PACKAGE_FILE_NAME poponnx-${PACKAGE_OS_TYPE}-${VERSION}-${SNAPSHOT})
set(CPACK_PACKAGE_FILE_NAME ${PACKAGE_FILE_NAME})

set(USER_GUIDE_PDF_NAME poponnx-user-guide-${VERSION}-${SNAPSHOT}.pdf)
set(USER_GUIDE_HTML_NAME poponnx-user-guide-html-${VERSION}-${SNAPSHOT}.zip)
set(PRODINFO_FILE_NAME poponnx-${VERSION}-${SNAPSHOT}.yml)

include(CPack)

set(PKG_UPLOAD_DIR ../../pkg)

if(BUILD_DOCS)
add_custom_target(package_and_move
  COMMAND ${CMAKE_CPACK_COMMAND}
  COMMAND bash -c "mkdir -p ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "mv ${PACKAGE_FILE_NAME}.tar.gz ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "mv ${PRODINFO_FILE_NAME} ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "mv ${USER_GUIDE_PDF_NAME} ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "mv ${USER_GUIDE_HTML_NAME} ${PKG_UPLOAD_DIR}"
)
else()
add_custom_target(package_and_move
  COMMAND ${CMAKE_CPACK_COMMAND}
  COMMAND bash -c "mkdir -p ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "mv ${PACKAGE_FILE_NAME}.tar.gz ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "mv ${PRODINFO_FILE_NAME} ${PKG_UPLOAD_DIR}"
)
endif()
#------ end of cpack information

#------ product info file
if(BUILD_DOCS)
  string(APPEND PKG_ELEMENTS "  - name: ONNX for Poplar user guide (PDF)\n"
                             "    file: ${USER_GUIDE_PDF_NAME}\n"
                             "    type: pdf\n")

  string(APPEND PKG_ELEMENTS "  - name: ONNX for Poplar user guide (HTML)\n"
                             "    file: ${USER_GUIDE_HTML_NAME}\n"
                             "    type: html_zip\n")
endif()

execute_process(
  COMMAND git config --get remote.origin.url
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
  OUTPUT_VARIABLE VIEW_REMOTE_URL
  OUTPUT_STRIP_TRAILING_WHITESPACE)

configure_file(poponnx.prodinfo.yml
  ${CMAKE_BINARY_DIR}/${PRODINFO_FILE_NAME}
  @ONLY)
#------ end of product info file

#------ enable script
configure_file(enable.sh.in
  ${CMAKE_BINARY_DIR}/enable.sh
  @ONLY)

install(FILES ${CMAKE_BINARY_DIR}/enable.sh
        DESTINATION .)
#------ end of enable script

# Docs
add_subdirectory(docs)

