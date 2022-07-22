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

The first time you run pre-commit you will get instructions on how to install
the linters you need. This includes `cpplint` 1.6.0, `clang-format` 9.0.0 for
C++ code, `black` 22.6.0, `pylint` 2.13.9 which are installed via pip.

### Unit Tests

Please run the unit test suite in the base `popart` directory to ensure that any changes you have made to the source code have not broken existing functionality:

```sh
source $POPART_INSTALL_DIR/enable.sh 
cd build
ctest -j 8
```
