# Contributing to PopART

Thank you for wanting to contribute the the PopART project.

## Code style

To ensure code quality PopART uses [pre-commit](https://pre-commit.com) hooks.
Install the hooks in the following way

```sh
pip3 install pre-commit
pre-commit install
```

This will run the hooks every time you run `git commit`.
To run the linters on all files use `pre-commit run --all-files`.

Alternatively one could use the `./format.sh` script in the base `popart` directory before making a pull request. This uses `clang-format`
on C++ code and `yapf` on python code. Please use `clang-format` version 9.0.0  and `yapf` version 0.27.0.

**NOTE**: `yapf` can be installed with `pip3`.

### Unit Tests

Please run the unit test suite in the base `popart` directory to ensure that any changes you have made to the source code have not broken existing functionality:

```sh
source $POPART_INSTALL_DIR/enable.sh 
cd build
ctest -j 8
```
