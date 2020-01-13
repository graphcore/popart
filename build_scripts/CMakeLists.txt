cmake_minimum_required(VERSION 3.12.0)
project(popart_super_project)

set(INSTALL_DIR ${PROJECT_BINARY_DIR}/install)

# for the ExternalProject_Add command
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
include(GNUInstallDirs)

ExternalProject_Add(spdlog
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/spdlog
  CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/spdlog"
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v0.16.3)

ExternalProject_Add(protobuf
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/protobuf
  SOURCE_SUBDIR cmake
  CMAKE_ARGS -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DBUILD_SHARED_LIBS=OFF
             -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/protobuf
             CFLAGS=-fPIC
             CXXFLAGS=-fPIC
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG d750fbf648256c7c631f51ffdbf67d7c18b0114e)

ExternalProject_Add(pybind11
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/pybind11
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG a6355b00f84d997a9ddcf209b6464447432be78a)

ExternalProject_Add(onnx
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/onnx
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/onnx
             -DBUILD_ONNX_PYTHON=1
             -DProtobuf_PROTOC_EXECUTABLE=${INSTALL_DIR}/protobuf/bin/protoc
             -DProtobuf_INCLUDE_DIR:STRING=${INSTALL_DIR}/protobuf/include
             -DProtobuf_LIBRARY:STRING=${INSTALL_DIR}/protobuf/lib/libprotobuf.a
             -DProtobuf_PROTOC_LIBRARY:STRING=${INSTALL_DIR}/protobuf/lib/libprotoc.a
             -DProtobuf_USE_STATIC_LIBS:STRING=1
             -DBUILD_SHARED_LIBS=OFF
             -DBUILD_ONNX_PYTHON=ON
	     -DONNX_ML=0
             -Dpybind11_DIR=${INSTALL_DIR}/pybind11/share/cmake/pybind11
  DEPENDS pybind11
          protobuf
  GIT_REPOSITORY https://github.com/onnx/onnx.git
  GIT_TAG v1.6.0)

ExternalProject_Add(boost
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/boost
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/boost
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON )

ExternalProject_Add(popart
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/../
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/popart
             -DPROTOBUF_INSTALL_DIR=${INSTALL_DIR}/protobuf
             -DONNX_INSTALL_DIR=${INSTALL_DIR}/onnx 
             -Dpybind11_DIR=${INSTALL_DIR}/pybind11/share/cmake/pybind11
             -DSPDLOG_INCLUDE_DIR=${INSTALL_DIR}/spdlog/include
             -DBOOST_ROOT=${INSTALL_DIR}/boost
             -DPOPLAR_INSTALL_DIR=${POPLAR_ROOT}
  DEPENDS spdlog
          protobuf
          pybind11
          onnx
          boost )

