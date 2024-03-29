# Building PopART

These instructions assume you are building PopART on Ubuntu 20.04. These instructions describe how to install every required dependency. If you are starting from an existing Ubuntu 20.04 installation you may already have some of these dependencies installed. If so, please ensure the versions of these dependencies are compatible with these instructions. Other Linux-based operating systems may also work but package names and supported versions of packages may vary.

## Installing Required Tooling

You will need the following tools installed on your system if they are not installed already:

### Wget, Git

```sh
sudo apt-get install wget git -y
```

### PIP3 (package installer for python 3)

```sh
sudo apt-get install python3-pip -y
ln -s /usr/bin/python3 /usr/bin/python
```

**NOTE**: If you have python 2.x installed on your system you can ignore the symlink.

### Ninja (version 1.8.2, optional)

```sh
sudo apt-get install ninja-build -y
```

### pkg-config

```sh
sudo apt-get install pkg-config -y
```

## Installing Required PIP Packages

All Python dependencies necessary for building and running PopART can be found in `requirements.txt`. Install them by running
```sh
python3 -m pip install -r requirements.txt
```

If you would like to contribute to PopART, additional dependencies are required for development.
These are found in `requirements/dev.txt` and can be installed by running:
```sh
python3 -m pip install -r requirements/dev.txt
```

## Installing Third-Party Library Dependencies

PopART compiles against a number of libraries that must be available on your system:

### Spdlog (version 1.8.0)

The version of the spdlog library in Ubuntu 20.04 (`spdlog-dev`) is not compatible with PopART. Instead, you need to build version 1.8.0 from source. To do this, in a directory of your choice, download the source from the [spdlog GitHub page](https://github.com/gabime/spdlog/tree/v1.8.0) and build and install as follows:

```sh
export SPDLOG_INSTALL_DIR=$(pwd)/spdlog-1.8.0/install_dir/
git clone --branch v1.8.0 https://github.com/gabime/spdlog.git
cd spdlog && mkdir build && cd build
cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$SPDLOG_INSTALL_DIR && cmake --build . --target install
```

**NOTE**: You will need the value of `SPDLOG_INSTALL_DIR` later.

### Pybind11 (version 2.6.2 or compatible)

The version of the pybind11 library in Ubuntu 20.04 (`pybind11-dev`) is 2.4.3, which is not compatible with PopART. Instead, you need to build version 2.6.2 from source. To do this, in a directory of your choice, download the source from the [pybind GitHub page](https://github.com/pybind/pybind11/releases) and build and install as follows:

```sh
export PYBIND11_INSTALL_DIR=$(pwd)/pybind11-2.6.2/install_dir/
wget https://github.com/pybind/pybind11/archive/v2.6.2.tar.gz
tar xvfz v2.6.2.tar.gz
rm v2.6.2.tar.gz
pushd pybind11-2.6.2
mkdir build
mkdir install_dir
cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$PYBIND11_INSTALL_DIR \
  -GNinja
ninja
ninja install
popd
```

**NOTE**: If you prefer building with `make` instead of `ninja`, remove the `-DCMAKE_GENERATOR="Ninja"` switch.

**NOTE**: You will need the value of `PYBIND11_INSTALL_DIR` later.

For more information, see: <https://github.com/pybind/pybind11/blob/master/docs/compiling.rst>.

### Boost (version 1.80.0 or compatible)

The Boost library in Ubuntu 20.04 (`libboost-dev`) is 1.71.1, which is not compatible with PopART. Instead, you have to build version 1.80.0 from source. To do this, in a directory of your choice, download the source from the [Boost download page](https://www.boost.org/users/history/version_1_80_0.html) and build and install as follows:

```sh
export BOOST_INSTALL_DIR=$(pwd)/boost_1_80_0/install_dir/
wget https://boostorg.jfrog.io/artifactory/main/release/1.70.0/source/boost_1_80_0.tar.gz
tar xvfz boost_1_80_0.tar.gz
rm boost_1_80_0.tar.gz
pushd boost_1_80_0
mkdir install_dir
./bootstrap.sh --prefix=$BOOST_INSTALL_DIR
./b2 -j8 link=static runtime-link=static --abbreviate-paths variant=release toolset=gcc "cxxflags= -fno-semantic-interposition -fPIC" cxxstd=14 --with-test --with-system --with-filesystem --with-program_options --with-graph --with-random install
popd
```

**NOTE**: The `-j8` switch is used to reduce build times by building with up to 8 threads.

**NOTE**: You will need the value of `BOOST_INSTALL_DIR` later.

For more information, see: <https://www.boost.org/doc/libs/1_80_0/more/getting_started/unix-variants.html>.

### Building ONNX (version 1.6.0 or compatible) from source

The ONNX library also needs to be compiled from source. To do this, in a directory of your choice, download the source from the [ONNX GitHub page](https://github.com/onnx/onnx/releases) and build and install as follows:

```sh
export ONNX_INSTALL_DIR=$(pwd)/onnx-1.6.0/install_dir/
wget https://github.com/onnx/onnx/archive/v1.6.0.tar.gz
tar xvfz v1.6.0.tar.gz
rm v1.6.0.tar.gz
pushd onnx-1.6.0
mkdir install_dir
cmake .. \
  -DONNX_ML=0 \
  -DProtobuf_PROTOC_EXECUTABLE=$PROTOBUF_INSTALL_DIR/bin/protoc \
  -DCMAKE_INSTALL_PREFIX=$ONNX_INSTALL_DIR
make -j8
make install
popd
```

**NOTE**: The `-j8` switch is used to reduce build times by building with up to 8 threads.

**NOTE**: You will need the value of `ONNX_INSTALL_DIR` later.

For more information, see: <https://github.com/onnx/onnx>.

### CapnProto (version 0.7.0 or compatible)

CapnProto releases can be downloaded from the [capnproto download page](https://capnproto.org/install.html). In a directory of your choice, download and install as follows:

```sh
export CAPNPROTO_INSTALL_DIR=$(pwd)/capnproto-0.7.0/install_dir/
wget https://capnproto.org/capnproto-c++-0.7.0.tar.gz
tar xvfz capnproto-c++-0.7.0.tar.gz
rm capnproto-c++-0.7.0.tar.gz
pushd capnproto-c++-0.7.0
./configure --prefix=$CAPNPROTO_INSTALL_DIR
make -j8 check
make install
popd
```

**NOTE**: The `-j8` switch is used to reduce test times by testing with up to 8 threads.

**NOTE**: You will need the value of `CAPNPROTO_INSTALL_DIR` later

For more information, see: <https://capnproto.org/install.html>

### Trompeloeil (version 35 or compatible)

Trompeloeil can be downloaded from the [trompeloeil GitHub page](https://github.com/rollbear/trompeloeil/releases/tag/v35). In a directory of your choice, download and install as follows:

```sh
export TROMPELOEIL_INSTALL_DIR=$(pwd)/trompeloeil-35/install_dir/
wget https://github.com/rollbear/trompeloeil/archive/refs/tags/v35.tar.gz
tar xvfz v35.tar.gz
rm v35.tar.gz
pushd trompeloeil-35
mkdir build ; cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$TROMPELOEIL_INSTALL_DIR
cmake --build . --target install
popd
```

## Installing Graphcore Library Dependencies

### Poprithms

You can checkout Graphcore's poprithms library in a suitable directory from the public [GitHub repository](https://github.com/graphcore/poprithms) and install it as follows:

```sh
export POPRITHMS_INSTALL_DIR=$(pwd)/poprithms/install_dir/
git clone https://github.com/graphcore/poprithms.git
pushd poprithms
mkdir build; cd build;
cmake .. \
  -DBOOST_ROOT=$BOOST_INSTALL_DIR \
  -DCMAKE_INSTALL_PREFIX=$POPRITHMS_INSTALL_DIR \
  -DCMAKE_GENERATOR="Ninja"
ninja
ninja install
popd
```

**NOTE**: If you prefer building with `make` instead of `ninja`, remove the `-DCMAKE_GENERATOR="Ninja"` switch.

**NOTE**: Builds can be further accelerated by using [ccache](https://ccache.dev/).

**NOTE**: You will need the value of `POPRITHMS_INSTALL_DIR` later.

For more information, see: [https://github.com/graphcore/poprithms](https://github.com/graphcore/poprithms).

### Poplar SDK

To obtain the Poplar SDK you need to [register](https://www.graphcore.ai/support) for access to Graphcore's [support portal](https://login.graphcore.ai/). Once you have access you can download the latest Ubuntu 18.04 from the support portal, unpack it in a suitable directory. For the remainder of this document the instructions assume you've set an environment variable `POPLAR_INSTALL_DIR` to point to the directory where Poplar is unpacked. Note that the Poplar SDK contains more than just Poplar and you will have to point the variable specifically to a subdirectory named something like `poplar-ubuntu_18_04-xxxxx`.

For more information, see:  <https://www.graphcore.ai/developer>.

## Configuring & Building PopART

Note that only Ubuntu 20.04 is supported for building PopART externally.

To build PopART, run the following commands in the directory where you checked out the repository:

```sh
export POPART_INSTALL_DIR=$(pwd)/popart/install_dir/
export PKG_CONFIG_PATH="$CAPNPROTO_INSTALL_DIR/lib/pkgconfig:$PKG_CONFIG_PATH"
git clone https://github.com/graphcore/popart.git
push popart
mkdir build; cd build;
cmake .. \
  -DBOOST_ROOT=$BOOST_INSTALL_DIR \
  -DCapnProto_ROOT=$CAPNPROTO_INSTALL_DIR \
  -DONNX_ROOT=$ONNX_INSTALL_DIR \
  -DPOPLAR_INSTALL_DIR=$POPLAR_INSTALL_DIR \
  -Dpoprithms_ROOT=$POPRITHMS_INSTALL_DIR \
  -DProtobuf_ROOT=$PROTOBUF_INSTALL_DIR \
  -Dpybind11_ROOT=$PYBIND11_INSTALL_DIR \
  -Dspdlog_ROOT=$SPDLOG_INSTALL_DIR \
  -Dtrompeloeil_ROOT=$TROMPELOEIL_INSTALL_DIR \
  -DCMAKE_INSTALL_PREFIX=$POPART_INSTALL_DIR \
  -GNinja
ninja
ninja install
popd
```

You could use any method supported by CMake to point it at dependencies. See the`find_package` documentation in the [CMake documentation](https://cmake.org/cmake/help/v3.16/command/find_package.html). We have chosen to use `<verbatim pkg name>_ROOT` variables that point to the package installation directory.

**DEPRECATION**: `<uppercase pkg name>_INSTALL_DIR` variables, except `POPLAR_INSTALL_DIR`, have been deprecated and will be removed in a future release.

**NOTE**: Other CMake switches are available:

* `-DPOPART_BUILD_TESTING=0` - Switch that can be used to avoid compiling PopART test.
* `-DPOPART_STRICT_COMPARATOR_CHECKS=1` - Check for `nullptr` and invalid pointers when comparing containers of pointers.

**NOTE**: If you prefer building with `make` instead of `ninja`, remove the `-GNinja` switch.

**NOTE**: Builds can be further accelerated by using [ccache](https://ccache.dev/).

**NOTE**: CapnProto’s CMake export simply wraps pkg-config. PKG_CONFIG_PATH is set in order to tell pkg-config where to find CapnProto.

## Using PopART

### Application Examples

There are a number of advanced PopART applications available in Graphcore's [example repository](https://github.com/graphcore/examples) on Github. See the [README](https://github.com/graphcore/examples#readme) for a full list of examples implemented using PopART and PopXL

## Licensing

The code is provided under the MIT license, see the License.txt file.

### TensorFlow

The project includes derived work from the following:
TensorFlow, https://github.com/tensorflow/tensorflow/

Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Relevant files:
* `tests/integration/optimizer_tests/rmsprop_update_numpy.py`

### LLVM Project

The project includes derivative work from the following:
LLVM Project, http://llvm.org/doxygen/MachineOutliner_8cpp_source.html

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Relevant files:
* `willow/src/subgraph/suffixtree.cpp`

### pybind11_mkdoc

The project includes derivative work from the following:
pybind11_mkdoc, https://github.com/pybind/pybind11_mkdoc

pybind11_mkdoc is licensed under the following MIT license:

The MIT License (MIT)

Copyright (c) 2020 Wenzel Jakob

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Relevant files:
* `scripts/pybind11_mkdoc/`

### ONNX

The project includes derivative work from the following:
ONNX, https://github.com/onnx/onnx

ONNX is licensed under the following MIT license:

MIT License

Copyright (c) ONNX Project Contributors
All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Relevant files:
* `tests/integration/operators_test/rnn_helper.py`
* `tests/integration/operators_test/gru_test.py`
* `tests/integration/operators_test/lstm_test.py`

### optional-lite

The project includes derivative work from the following:
optional-lite, https://github.com/martinmoene/optional-lite

Copyright (c) 2014-2018 Martin Moene

Boost Software License - Version 1.0 - August 17th, 2003

Distributed under the Boost Software License, Version 1.0.

Permission is hereby granted, free of charge, to any person or organization
obtaining a copy of the software and accompanying documentation covered by
this license (the "Software") to use, reproduce, display, distribute,
execute, and transmit the Software, and to prepare derivative works of the
Software, and to permit third-parties to whom the Software is furnished to
do so, all subject to the following:

The copyright notices in the Software and this entire statement, including
the above license grant, this restriction and the following disclaimer,
must be included in all copies of the Software, in whole or in part, and
all derivative works of the Software, unless such copies or derivative
works are solely in the form of machine-executable object code generated by
a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

Relevant Files:
* `willow/include/popart/vendored/optional.hpp`

### any-lite

The project includes derivative work from the following:
any-lite, https://github.com/martinmoene/any-lite

Copyright (c) 2016-2018 Martin Moene

Boost Software License - Version 1.0 - August 17th, 2003

Distributed under the Boost Software License, Version 1.0.

Permission is hereby granted, free of charge, to any person or organization
obtaining a copy of the software and accompanying documentation covered by
this license (the "Software") to use, reproduce, display, distribute,
execute, and transmit the Software, and to prepare derivative works of the
Software, and to permit third-parties to whom the Software is furnished to
do so, all subject to the following:

The copyright notices in the Software and this entire statement, including
the above license grant, this restriction and the following disclaimer,
must be included in all copies of the Software, in whole or in part, and
all derivative works of the Software, unless such copies or derivative
works are solely in the form of machine-executable object code generated by
a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

Relevant Files:
* `willow/include/popart/vendored/anylite.hpp`
