cmake_minimum_required(VERSION 3.6)
project(boost)

include(ExternalProject)
include(GNUInstallDirs)
set(BOOST_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/boost_install)

set(Boost_build_args --prefix=${BOOST_INSTALL_DIR})
if(BUILD_SHARED_LIBS)
  list(APPEND Boost_build_args link=shared)
else()
  # Work around issues with long paths in Windows, and static link the CRT to 
  # reduce dependencies when running in WinPE.
  list(APPEND Boost_build_args link=static runtime-link=static --abbreviate-paths)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND Boost_build_args variant=debug)
else()
  list(APPEND Boost_build_args variant=release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(BOOST_BUILD_TOOLSET clang)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(BOOST_BUILD_TOOLSET msvc)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(BOOST_BUILD_TOOLSET gcc)
else()
  message(FATAL_ERROR "unsupported compiler \"${CMAKE_CXX_COMPILER_ID}\"")
endif()

list(APPEND Boost_build_args toolset=${BOOST_BUILD_TOOLSET})

if(MINGW)
  if(CMAKE_GENERATOR STREQUAL "MSYS Makefiles")
    set(Boost_CONFIGURE_COMMAND
        cmd /C${CMAKE_CURRENT_SOURCE_DIR}/boost/bootstrap.bat mingw)
  else()
    set(Boost_CONFIGURE_COMMAND
        ${CMAKE_CURRENT_SOURCE_DIR}/boost/bootstrap.bat mingw)
  endif()
elseif(MSVC)
  set(Boost_CONFIGURE_COMMAND
      ${CMAKE_CURRENT_SOURCE_DIR}/boost/bootstrap.bat msvc)
else()
  set(Boost_CONFIGURE_COMMAND
      ${CMAKE_CURRENT_SOURCE_DIR}/boost/bootstrap.sh
       --with-toolset=${BOOST_BUILD_TOOLSET})

  if(CMAKE_POSITION_INDEPENDENT_CODE)
    string(APPEND CMAKE_CXX_FLAGS " -fPIC")
  endif()
endif()

if(CMAKE_CXX_FLAGS)
  list(APPEND Boost_build_args "cxxflags=${CMAKE_CXX_FLAGS}")
endif()

list(APPEND Boost_build_args "cxxstd=14")

# Avoid building libraries with external dependencies.

# As of Boost 1.68 this is a list of Boost libraries that aren't
# header-only and a non-exhuastive list of where they are used. See
# `./b2 --show-libraries` to see the list.
#
# * atomic - gcserver
# * chrono
# * container - exchange, poplar
# * context
# * contract
# * coroutine
# * date_time - gcserver
# * exception
# * fiber
# * filesystem - graphcore_binary
# * graph
# * graph_parallel
# * iostreams
# * locale
# * log - poplar
# * math - arch_man, poplibs
# * mpi
# * program_options - poplar, enigma, etc.
# * python
# * random - exchange, poplar
# * regex - ciss, poplar
# * serialization
# * signals
# * stacktrace
# * system - graphcore_target_access
# * test - poplar, poplibs, etc.
# * thread - mcu_server
# * timer - graphcore_binary
# * type_erasure
# * wave

list(APPEND Boost_build_args
  --with-atomic
  --with-container
  --with-date_time
  --with-filesystem
  --with-log
  --with-math
  --with-program_options
  --with-random
  --with-regex
  --with-system
  --with-test
  --with-thread
  --with-timer
)
if(EXTERNAL_PROJECT_NUM_JOBS)
  list(APPEND Boost_build_args "-j${EXTERNAL_PROJECT_NUM_JOBS}")
endif()

# Set a variable for the compiler command that boost build should use. The
# compiler command may contain ':', e.g. C:/path/to/compiler.exe
# Unquoted colons are field separators in jam files so the command must be
# quoted.
set(BOOST_BUILD_COMPILER_CMD "\"${CMAKE_CXX_COMPILER}\"")
if(NOT CMAKE_CXX_COMPILER_ARG1 STREQUAL "")
  # CXX_COMPILER_ARG1 has a leading space that should be ignored.
  string(STRIP "${CMAKE_CXX_COMPILER_ARG1}" BOOST_BUILD_COMPILER_ARG1)
  string(APPEND BOOST_BUILD_COMPILER_CMD " \"${BOOST_BUILD_COMPILER_ARG1}\"")
endif()

configure_file(user-config.jam.in user-config.jam @ONLY)

if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # We don't do this with MSVC to work around
  # https://github.com/boostorg/build/issues/455
  list(APPEND Boost_build_args
       "--user-config=${CMAKE_CURRENT_BINARY_DIR}/user-config.jam")
endif()

# LOG_INSTALL is used to workaround an issue when building with MSVC. Boost
# build performs various compile tests, some of which may fail. Without LOG_INSTALL
# the compiler error messages go to stdout and when MSVC sees the compiler
# error it considers the build to have failed, even if the boost build command
# succeeded. This seem to be the same issue as the on described here:
# https://svn.boost.org/trac/boost/ticket/4626
# However the workaround suggested there (unsetting VS_UNICODE_OUTPUT) didn't
# fix the issue.
if(CMAKE_GENERATOR MATCHES "Visual Studio")
  set(LOG_OPTION LOG_INSTALL 1)
else()
  set(LOG_OPTION)
endif()

message(STATUS "Boost Build Args: ${Boost_build_args}")

# Boost uses its own unique build system (yeay) called b2. Build it in this dir.
set(BOOST_B2_DIR ${CMAKE_CURRENT_BINARY_DIR}/boost_b2)

# Even so, it still always dumps some files in the source tree. This is done
# in lines ~220-239 of bootstrap.sh.
set(BOOST_DIRTY_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/boost/tools/build/src/engine/bin.macosxx86_64/b2
  ${CMAKE_CURRENT_SOURCE_DIR}/boost/tools/build/src/engine/bin.macosxx86_64/bjam
  ${CMAKE_CURRENT_SOURCE_DIR}/boost/tools/build/src/engine/bin.linuxx86_64/b2
  ${CMAKE_CURRENT_SOURCE_DIR}/boost/tools/build/src/engine/bin.linuxx86_64/bjam
  ${CMAKE_CURRENT_SOURCE_DIR}/boost/tools/build/src/engine/bootstrap/jam0
)

if(WIN32)
  set(CONFIG_DIR "<SOURCE_DIR>")
else()
  set(CONFIG_DIR "${BOOST_B2_DIR}")
endif()

ExternalProject_Add(Boost
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/boost
  CONFIGURE_COMMAND
    ${CMAKE_COMMAND} -E make_directory ${BOOST_B2_DIR} &&
    ${CMAKE_COMMAND} -E chdir ${CONFIG_DIR} ${Boost_CONFIGURE_COMMAND} &&
    ${CMAKE_COMMAND} -E remove -f ${BOOST_DIRTY_FILES}
  BUILD_COMMAND ""
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -E chdir <SOURCE_DIR>
    ${CONFIG_DIR}/b2
      --build-dir=<BINARY_DIR>
      ${Boost_build_args}
      install
  ${LOG_OPTION}
  URL https://dl.bintray.com/boostorg/release/1.70.0/source/boost_1_70_0.tar.bz2
)
ExternalProject_Add_Step(Boost force_rebuild DEPENDERS build ALWAYS 1)

install(DIRECTORY ${BOOST_INSTALL_DIR}/lib/
        COMPONENT libboost
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        USE_SOURCE_PERMISSIONS)
install(DIRECTORY ${BOOST_INSTALL_DIR}/include/
        COMPONENT libboost-dev
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        USE_SOURCE_PERMISSIONS)

install(FILES FindBoost.cmake
        DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake )
