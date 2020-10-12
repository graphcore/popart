FROM ubuntu:18.04

RUN apt-get update

# -- BUILD TOOLS --

# Install wget, git.
RUN apt-get install wget git -y
# Install python.
RUN apt-get install python3 -y
# Install pip3.
RUN apt-get install python3-pip -y
# Install ninja.
RUN apt-get install ninja-build -y

# Install cmake.
WORKDIR /tmp/
RUN wget https://cmake.org/files/v3.17/cmake-3.17.2.tar.gz
RUN tar xzvf cmake-3.17.2.tar.gz
RUN rm cmake-3.17.2.tar.gz
WORKDIR /tmp/cmake-3.17.2
RUN ./bootstrap --parallel=8 -- -DCMAKE_USE_OPENSSL=OFF
RUN make -j8
RUN make install
RUN rm -rf /tmp/cmake-3.17.2

# -- PIP PACKAGES --

# Protobuf
RUN pip3 install protobuf>=3.7.0
# ONNX 1.7.0 or compatible.
RUN pip3 install onnx==1.7.0
# Pytest and Pytest-forked
RUN pip3 install pytest
RUN pip3 install pytest-forked
# Numpy
RUN pip3 install numpy
# Torch 1.6.0
RUN pip3 install torch==1.6.0
# Torch Vision 0.7.0
RUN pip3 install torchvision==0.7.0

# -- THIRD PARTY LIBRARIES --

# Spdlog 0.16.3
RUN apt-get install libspdlog-dev -y

# Pybind11 2.5.0
WORKDIR /workspace/libs/third-party/
ENV PYBIND11_INSTALL_DIR /workspace/libs/third-party/pybind11-2.5.0/install_dir/
RUN wget https://github.com/pybind/pybind11/archive/v2.5.0.tar.gz
RUN tar xvfz v2.5.0.tar.gz
RUN rm v2.5.0.tar.gz
WORKDIR /workspace/libs/third-party/pybind11-2.5.0
RUN mkdir build
RUN mkdir install_dir
WORKDIR /workspace/libs/third-party/pybind11-2.5.0/build
RUN cmake .. -DCMAKE_INSTALL_PREFIX=$PYBIND11_INSTALL_DIR -DCMAKE_GENERATOR="Ninja"
RUN ninja
RUN ninja install

# Boost 1.70.0
WORKDIR /workspace/libs/third-party/
ENV BOOST_INSTALL_DIR /workspace/libs/third-party/boost_1_70_0/install_dir/
RUN wget https://dl.bintray.com/boostorg/release/1.70.0/source/boost_1_70_0.tar.gz
RUN tar xvfz boost_1_70_0.tar.gz
RUN rm boost_1_70_0.tar.gz
WORKDIR /workspace/libs/third-party/boost_1_70_0
RUN mkdir install_dir
RUN ./bootstrap.sh --prefix=/workspace/libs/third-party/boost_1_70_0/install_dir
RUN ./b2 -j8 link=static runtime-link=static --abbreviate-paths variant=release toolset=gcc "cxxflags= -fno-semantic-interposition -fPIC" cxxstd=14 --with-test --with-system --with-filesystem --with-program_options install

# Protobuf 3.7.0
WORKDIR /workspace/libs/third-party/
ENV PROTOBUF_INSTALL_DIR /workspace/libs/third-party/protobuf-3.7.0/install_dir/
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v3.7.0/protobuf-cpp-3.7.0.tar.gz
RUN tar xvfz protobuf-cpp-3.7.0.tar.gz
RUN rm protobuf-cpp-3.7.0.tar.gz
WORKDIR /workspace/libs/third-party/protobuf-3.7.0
RUN mkdir install_dir
RUN CXXFLAGS=-fPIC CFLAGS=-fPIC ./configure --prefix=/workspace/libs/third-party/protobuf-3.7.0/install_dir/
RUN make -j8
RUN make check
RUN make install

# ONNX 1.7.0
RUN ln -s /usr/bin/python3 /usr/bin/python
WORKDIR /workspace/libs/third-party/
ENV ONNX_INSTALL_DIR /workspace/libs/third-party/onnx-1.7.0/install_dir/
RUN wget https://github.com/onnx/onnx/archive/v1.7.0.tar.gz
RUN tar xvfz v1.7.0.tar.gz
RUN rm v1.7.0.tar.gz
WORKDIR /workspace/libs/third-party/onnx-1.7.0
RUN mkdir build
RUN mkdir install_dir
WORKDIR /workspace/libs/third-party/onnx-1.7.0/build
RUN cmake .. -DONNX_ML=0 -DProtobuf_PROTOC_EXECUTABLE=/workspace/libs/third-party/protobuf-3.7.0/install_dir/bin/protoc -DCMAKE_INSTALL_PREFIX=$ONNX_INSTALL_DIR
RUN make -j8
RUN make install

# -- Set workdir.

WORKDIR /workspace

