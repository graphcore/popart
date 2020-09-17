# Writing PopART unit tests

The test devices are determined by the CMake variable `TEST_TARGET`, this is `Cpu` by default. By changing `DEFAULT_TEST_VARIANTS` to `Cpu;IpuModel;Hw` in [tests/popart/CMakeLists.txt]() you can compile a variant of every test to run on every device type.

Alternatively, to compile tests for different devices, run cmake with the following arguments:
* `-DPOPART_CMAKE_ARGS="-DENABLED_TEST_VARIANTS=Cpu$<SEMICOLON>IpuModel$<SEMICOLON>Hw"` Will enable all tests.
* `-DPOPART_CMAKE_ARGS="-DENABLED_TEST_VARIANTS=Hw"` Will enable hardware tests only.
* `-DPOPART_CMAKE_ARGS="-DENABLED_TEST_VARIANTS=Cpu$<SEMICOLON>IpuModel"` Will enable non-hardware tests only.

Also, you can run `./test.sh -R Hw_` to only run hardware tests, similarly for `IpuModel_` or `Cpu_`
## C++ tests

Make sure to include `popart/testdevice.hpp` in all cpp unit tests. When acquiring a device use `createTestDevice(TEST_TARGET)` to acquire your device. `TEST_TARGET` is defined at compile time and is determined by what you specify in the arguments for `add_popart_cpp_unit_test` in `CMakeLists.txt`, defaulting to `Cpu`. See [tests/popart/CMakeLists.txt]() for details. [tests/popart/ipu_hash_tests.cpp]() is an example of a test on IPUModel only.

For example, if creating `example_test.cpp` for an `IpuModel` device:
1) Include `popart/testdevice.hpp` and use `createTestDevice(TEST_TARGET)` in your test when acquiring a device.
2) Add `add_popart_cpp_unit_test(example_test example_test.cpp VARIANT "IpuModel")` to the `CMakeLists.txt` in the test's directory.

This will create a test called `IpuModel_default_example_test` which will run on an IPU model via `test.sh` or `ctest`. 

If you want additional devices to run the test, append additional devices, separated by semicolons in the `add_popart_cpp_unit_test` call. E.g. 
```
add_popart_cpp_unit_test(example_test example_test.cpp VARIANT "IpuModel;Hw;Cpu")
```
will run on IPU Model, IPU (hardware) and CPU, creating one test for each.

Try to separate out tests that require parts on different devices. For example `example_test_0` run on CPU only and `example_test_1` run on hardware only. If combined into one test, you will require logic to ensure that the CPU part doesn't run on IPU and vice versa.

---

## Python tests

Python unit tests are run using `pytest`; they can be ran directly with `pytest` via 
```
pytest --forked test/popart/.../<test_name>.py
```
or via `ctest` and `test.sh` with
```
cd <build folder>
. ./test.sh popart -R <test_name>
```
The logic of adding tests for a device type as in `ctest` is replicated; in the `CMakeLists.txt` in the parent folder, add 
```
add_popart_py_unit_test(<test_name> MATCHEXPR <expr> VARIANTS <variant1>;<variant2>;<variant3>)
```
> Note: tests will not be added automatically any more, you must specify them in `CMakeLists.txt`. e.g.:
> * `add_popart_py_unit_test(train_then_infer_test VARIANTS IpuModel)` Model only
> * `add_popart_py_unit_test(variable_inference_test VARIANTS Cpu;IpuModel;Hw)` All targets
> * `add_popart_py_unit_test(variable_inference_test)` default - CPU only.

Note that the MATCHEXPR is optional for python tests and directly passes whatever arguments it is given
to `pytest`'s `-k` argument. This means that you pass in a python-like boolean expression where variables
represent substrings that match test names. This feature is useful for splitting files with computationally
heavy tests into multiple ctests or for excluding specific tests from a ctest. Examples are:

> * `add_popart_py_unit_test(variable_inference_test MATCHEXPR add_variable_fp16)` -- only run test_add_variable_fp16.
> * `add_popart_py_unit_test(variable_inference_test MATCHEXPR not add_variable_fp16 VARIANTS Hw)` -- run all tests except test_add_variable_fp16 (with Hw target only).
> * `add_popart_py_unit_test(variable_inference_test MATCHEXPR add_variable_fp16 or add_variable_fp32)` -- run tests add_variable_fp16 and add_variable_fp32.

To enforce the running with a specific device in `pytest` you need to:
1) `import test_util as tu` - This give you access to various testing functions. The folder's `__init__.py` should have added the directory to enable you to import this.
2) In your session use 
`deviceInfo=tu.create_test_device(numIpus: int = 1,
                       opts: Dict = None,
                       pattern: popart.SyncPattern = popart.SyncPattern.Full)` instead of `popart.DeviceManager().createIpuModelDevice(opts)`
or whatever.
3) Add decorators to tell pytest what device to use whn running directly. Note you only need to do this on your `test_` functions.
    1) `@tu.requires_ipu` will run on IPU hardware
    2) `@tu.requires_ipu_model` will run on IPU Model.
    3) No decorator will run on CPU.
4) Add your test to the relevant `CMakeLists.txt` as above.



To run `pytest` and specify a device type, use:
```
TEST_TARGET=<variant> pytest --forked tests/popart/../<test_name>.py
```
e.g. 
```
TEST_TARGET=IpuModel pytest --forked tests/popart/mapping_test.py 
```
See existing tests for more info on how to use the `test_util` functions.