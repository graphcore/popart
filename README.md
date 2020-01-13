POPART OVERVIEW
----------------
TODO

CONTRIBUTING
------------

CODE STYLE
----------
Please run the ./format.sh script in the base popart directory
before making a pull request (aka a diff). This uses clang-format 
on C++ code and yapf on Python code. Please use yapf version 0.24. 
yapf can be installed with pip.

Compiler warnings: If using clang++ (preferred compiler), many useful 
extra warnings are enabled (see cmake/EnableCompilerWarnings.cmake). 
Warnings with g++ are being worked on (an open task)


CONFIGURE and BUILD
-------------------

To build PopART:

```
cd build_scripts
mkdir build
cd build
cmake .. -DPOPLAR_ROOT=<path_to_poplar>
make -j 60 popart
```

After building, use PopART by sourcing the enable script:
```
source build_scripts/build/install/popart/enable.sh
```

RUN UNIT TESTS
--------------
Run the unit test suite to ensure that any changes you have made to the
source code haven't broken existing functionality:
```
cd build_scripts/build
./test.sh popart -j 60
```
