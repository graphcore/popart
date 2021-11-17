This directory holds files used to extract documentation from C++ Doxygen comments, convert them into Sphinx-compatible docstrings, and make them available to use when binding C++ to Python.

# Files

## pybind11_mkdoc

Consists of 3 files:
* `__main__.py` - Extremely minimal file that allows the package to be run instead of imported.
* `__init__.py` - Contains the package entry point, which parses the command line arguments and calls `mkdoc` from `mkdoc_lib.py`.
* `mkdoc_lib.py` - Contains all the heavy logic of the program. It goes over all the input files, extracts all comments from them, converts the comments to a different format and generates `pydocs_popart_core.hpp` that contains the newly generated comment documenation as static C++ char arrays.

The license from the official project is also included in the `LICENSE` file.

## gen_python_docs.sh

This script calls `pybind11_mkdoc` with the appropriate parameters to generate Python documentation and saves the output in `willow/include/popart/docs/pydocs_popart_core.hpp`. These docs can then be accessed via `DOC` and `SINGLE_LINE_DOC` macros when binding C++ to Python (look for examples in `popart.cpp`). 

The script must be run from the root `popart` directory.

Note: `SINGLE_LINE_DOC(...)` is just the concatenation of `DOC(...)` into a single line (with a single space in-between any 2 lines). These should be used to document enum class values and nothing else, because their documentation formatting displays incorrectly if it spans more than a single line.

# Dependencies

The only dependency is `libclang`, which is already included in `requirements.txt` and `requirements/build.txt`.