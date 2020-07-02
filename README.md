Building PopART
---------------

## Build Requirements

### Poplar SDK

In order to build PopART, the Poplar SDK must be downloaded and installed.
Please see https://www.graphcore.ai/developer for details.

### CMake Version 3.10.2 or greater

On Ubuntu 18.04:

    apt install cmake

### Ninja Version 1.8.2 (optional)

These instructions use Ninja. However, you may choose to use an alternative build system.

On Ubuntu 18.04:

    $ apt install ninja-build

### Boost Version 1.70.0 (or compatible with)

Download Boost 1.70 source from here: https://www.boost.org/users/history/version_1_70_0.html

Within a suitable directory, extract the archive and run:

    $ mkdir install # Make a note of this path for later.
    $ ./bootstrap.sh --prefix=install
    $ ./b2 link=static runtime-link=static --abbreviate-paths variant=release toolset=gcc "cxxflags= -fno-semantic-interposition -fPIC" cxxstd=14 --with=all install

Note: Consider using '-j8' (or similar) with './b2' to reduce build time by increasing concurrency.

For more information, see: https://www.boost.org/doc/libs/1_70_0/more/getting_started/unix-variants.html

### Spdlog Version 0.16.3 (or compatible with)

On Ubuntu 18.04:

    $ apt install libspdlog-dev

### ONNX Version 1.6.1

PopART is known to work with ONNX 1.6.1

You can install `onnx` using `pip`.

### protobuf Version 3.6.1

PopART is known to work with protobuf 3.6.1.

See https://developers.google.com/protocol-buffers/docs/downloads for more information.

### Python 3

Ubuntu 18.04 ships with Python 3 already installed.

### Python packages

You will need to install the following packages:

* protobuf
* pytest
* numpy

For running some tests and for PyTorch integration, you will also need to install PyTorch.

These can all be installed using `pip`.

## Code style

Please run the `./format.sh` script in the base `popart` directory
before making a pull request (aka a diff). This uses `clang-format`
on C++ code and `yapf` on Python code. Please use `yapf` version 0.24.
`yapf` can be installed with `pip`.

## Configure and build

Note that only Ubuntu 18.04 is supported for building PopART externally.

To build PopART:

```
$ cd build_scripts
$ mkdir build
$ cd build
$ cmake .. -DPOPLAR_INSTALL_DIR=/the/poplar/install/ -DPROTOBUF_INSTALL_DIR=~/the/protobuf/install/ -DBOOST_ROOT=~/the/boost/install/ -DPOPRITHMS_INSTALL_DIR=~/the/poprithms/install -DSPDLOG_INCLUDE_DIR=~/the/spdlog/installs/include/  -DONNX_INSTALL_DIR=~/the/onnx/install/ -DCMAKE_GENERATOR="Ninja" -DCMAKE_INSTALL_PREFIX=~/my/popart/install
$ ninja
$ ninja install
```

Note: builds can be accelerated by using the `ninja` build system and `ccache`.

After building, use PopART by sourcing the enable script:
```
source build_scripts/build/install/popart/enable.sh
```

## Run unit tests

Run the unit test suite to ensure that any changes you have made to the
source code haven't broken existing functionality:
```
cd build_scripts/build
./test.sh popart -j 60
```
