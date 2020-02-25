# Writing PopART unit tests

The test devices are determined by the CMake variable `TEST_TARGET`, this is `Cpu` by default. By changing `DEFAULT_TEST_VARIANTS` to `Cpu;IpuModel;Hw` in [tests/popart/CMakeLists.txt]() you can compile a variant of every test to run on every device type.

Also, you can run `./test.sh -R Hw_` to only run hardware tests, similarly for `IpuModel_` or `Cpu_`
## C++ tests

Make sure to include `popart/testdevice.hpp` in all cpp unit tests. When acquiring a device use `createTestDevice(TEST_TARGET)` to acquire your device. `TEST_TARGET` is defined at compile time and is determined by what you specify in the arguments for `add_popart_cpp_unit_test` in `CMakeLists.txt`, defaulting to `Cpu`. See [tests/popart/CMakeLists.txt]() for details. [tests/popart/ipu_hash_tests.cpp]() is an example of a test on IpuModel only.

For example, if creating `example_test.cpp` for an `IpuModel` device:
1) Include `popart/testdevice.hpp` and use `createTestDevice(TEST_TARGET)` in your test when acquiring a device.
2) Add `add_popart_cpp_unit_test(example_test example_test.cpp VARIANT "IpuModel")` to the `CMakeLists.txt` in the test's directory.

This will create a test called `IpuModel_default_example_test` which will run on an IPU model via `test.sh` or `ctest`. 

If you want additional devices to run the test, append additional devices, separated by semicolons in the `add_popart_cpp_unit_test` call. E.g. `add_popart_cpp_unit_test(example_test example_test.cpp VARIANT "IpuModel;Hw;Cpu")` will run on Ipu Model, Ipu (hardware) and Cpu, creating one test for each.

Try to separate out tests that require parts on different devices. For example `example_test_0` run on Cpu only and `example_test_1` run on hardware only. If combined into one test, you will require logic to ensure that the Cpu part doesn't run on Ipu and vice versa.

---

## Python tests

TODO: `D21466` will enable multi-device testing in python.