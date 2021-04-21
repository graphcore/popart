# Unit tests in this folder should be quick to run (<<0.1s) and should test one
# class in isolation. If the class has other classes it depends you should
# consider mocking those dependencies so you are only testing one class. By
# adding the `TEST_UTILS trompeloeil` you should have access to the trompeloeil
# mocking framework. An example of a unit test that uses trompeloeil can be
# found in `example_unittest.cpp`.
#
# In unit tests, the code under test should not depend on timers, it should not
# be creating or reading files, it should not be calling libraries such as
# Poplar or use any other resources such as databases. Ideally you are testing
# one class in isolation and no instantiating any other production classes.
# There are times where it is appropriate to bend these rules. However, in
# general, to meet these requirements you may need to refactor code and make it
# possible to insert test versions of the dependencies of the code under test.
#
# The benefit of writing unit tests is that 1) it makes the feedback loop
# faster because the tests are super quick to run 2) having tests surrounding a
# class allows a maintainer to make changes with confidence, as they now can see
# what the class is supposed to do and can be confident their changes don't
# break its behaviour 3) it gives more localised information compared to
# integration tests and 4) it forces you to decouple dependencies. One thing
# to be wary of is over-specifying the expected behaviour of a class in a unit
# test, tying it to a specific implementation.
#
# Tests that do not meet the criteria above and, e.g., test multiple PopART
# classes are still valuable, but they are not unit tests, they are
# integration tests. Integration tests belong in popart/tests/popart.
#
# Please prefix all test names in this folder with 'unittest_' so that
# developers can run `./test.sh popart -R unittest_` to run only these tests.

add_popart_cpp_unit_test(unittest_example unittest_example.cpp TEST_UTILS trompeloeil)
add_popart_cpp_unit_test(unittest_graphutils unittest_graphutils.cpp TEST_UTILS test-graphs-test-util)
add_popart_cpp_unit_test(unittest_utils unittest_utils.cpp TEST_UTILS trompeloeil)
add_popart_cpp_unit_test(unittest_Session_createFromIr session_createFromIr.cpp)

add_popart_cpp_unit_test(unittest_irquerier testutil/unittest_irquerier.cpp TEST_UTILS ir-query-test-util trompeloeil)

add_popart_cpp_unit_test(unittest_autodiff transforms/unittest_autodiff.cpp TEST_UTILS ir-query-test-util)
add_popart_cpp_unit_test(unittest_backwardsgraphcreatorhelper transforms/autodiff/unittest_backwardsgraphcreatorhelper.cpp)
add_popart_cpp_unit_test(unittest_calledgraphgradophelper transforms/autodiff/unittest_calledgraphgradophelper.cpp)
