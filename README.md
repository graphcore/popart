POPONNX OVERVIEW
----------------
TODO

CONTRIBUTING
------------

CODE STYLE
----------
Please run the ./format.sh script in the base poponnx directory
before making a pull request (aka a diff). This uses clang-format 
on C++ code and yapf on Python code. Please use yapf version 0.24. 
yapf can be installed with pip.

Compiler warnings: If using clang++ (preferred compiler), many useful 
extra warnings are enabled (see cmake/EnableCompilerWarnings.cmake). 
Warnings with g++ are being worked on (an open task)


CONFIGURE and BUILD
-------------------

These are directions for building directly from the poponnx repository, 
not from the poponnx_iew repo, which is the suggested way to build.
To build from the poponnx_view directory, see the instructions in the wiki:
https://phabricator.sourcevertex.net/w/onnx/


On Ubuntu, in a clean build directory:
```
cmake -DONNX_DIR=/where/onnx/installed/share/cmake/ONNX 
      -DCMAKE_PREFIX_PATH=/where/pybind11/installed/share/cmake/pybind11
      -DPOPLAR_INSTALL_DIR=/where/poplar/installed
      -DCMAKE_INSTALL_PREFIX=/where/to/install/poponnx
      /path/to/base/poponnx/dir
```

where I assume above that ONNX and pybind are already built and installed.

Next, install pytorch, the directions are on the pytorch website.

Now, 
export LD_LIBRARY_PATH=~/where/poponnx/installed/lib:/where/poplar/installed/lib

and you should be ready to use poponnx with an `import poponnx`


