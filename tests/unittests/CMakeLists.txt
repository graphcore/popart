# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Unit tests in this folder should be quick to run (<<0.1s) and should test one
# class in isolation. If the class has other classes it depends you should
# consider mocking those dependencies so you are only testing one class. By
# adding the `SUPPORT_LIBS trompeloeil` you should have access to the trompeloeil
# mocking framework. An example of a unit test that uses trompeloeil can be
# found in `unittest_example.cpp`.
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
# integration tests. Integration tests belong in tests/integration.
#
# For consistency, please follow the directory structure and naming of the
# source code.
#
# * Please put the c++ unittests for a file
#   'willow/src/somedir1/somedir2/file.cpp' in a file called
#   'tests/unittests/willow/somedir1/somedir2/test_file.cpp' and give it a ctest
#   name of 'unittest_willow_somedir1_somedir2_file'.
# * Please put the python unittests for a file
#   'python/somedir1/somedir2/file.py' in a file called
#   'tests/unittests/python/somedir1/somedir2/test_file.py' and use
#    `add_popart_py_unit_test(test_file)` to create a ctest instance.
#
# By doing this consistently it will be easy to match up tests with source
# code and it becomes easy for developers to run all unittests by running, e.g.,
# `./test.sh popart -R unittest`.

# C++ unittests for main codebase:
add_unit_test(unittest_example unittest_example.cpp SUPPORT_LIBS trompeloeil)
add_unit_test(unittest_profilecacher unittest_profilecacher.cpp)
add_unit_test(unittest_fileoperations unittest_fileoperations.cpp)
add_unit_test(unittest_gen_operators unittest_gen_operators.cpp)
add_unit_test(unittest_graphutils unittest_graphutils.cpp SUPPORT_LIBS test-graphs-test-util)
add_unit_test(unittest_tensor unittest_tensor.cpp SUPPORT_LIBS test-graphs-test-util)
add_unit_test(unittest_tensorid unittest_tensorid.cpp SUPPORT_LIBS test-graphs-test-util)
add_unit_test(unittest_vectorandset unittest_vectorandset.cpp)
add_unit_test(unittest_utils unittest_utils.cpp SUPPORT_LIBS trompeloeil)
add_unit_test(unittest_Session_createFromIr session_createFromIr.cpp)

add_unit_test(unittest_region unittest_region.cpp)
add_unit_test(unittest_pointercomparators unittest_pointercomparators.cpp)
add_unit_test(unittest_parsedtensorid unittest_parsedtensorid.cpp)
add_unit_test(unittest_replicatedtensorsharding unittest_replicatedtensorsharding.cpp SUPPORT_LIBS test-graphs-test-util ir-query-test-util)

add_unit_test(unittest_ReplicatedTensorMutableVoidInfoVerifier unittest_replicatedtensormutablevoidinfoverifier.cpp)

add_unit_test(unittest_pattern_optimizerdecompose patterns/optimizerdecompose.cpp)
add_unit_test(unittest_pattern_lambserialisedweight patterns/lambserialisedweight.cpp)
add_unit_test(unittest_pattern_sgd2decompose patterns/sgd2decompose.cpp SUPPORT_LIBS test-graphs-test-util ir-query-test-util)
add_unit_test(unittest_pattern_subtractarg1gradop patterns/subtractarg1gradop.cpp)
add_unit_test(unittest_pattern_tiedgather patterns/tiedgather.cpp)
add_unit_test(unittest_pattern_updateinplaceprioritiesforipu patterns/updateinplaceprioritiesforipu.cpp)
add_unit_test(unittest_pattern_convtranspose patterns/convtranspose.cpp)

add_unit_test(unittest_sgd_optimizer optimizer/sgd_optimizer.cpp)

add_unit_test(unittest_graphtestwrapper testutil/irquery/unittest_graphtestwrapper.cpp SUPPORT_LIBS ir-query-test-util trompeloeil)
add_unit_test(unittest_irtestwrapper testutil/irquery/unittest_irtestwrapper.cpp SUPPORT_LIBS ir-query-test-util trompeloeil)
add_unit_test(unittest_opstestwrapper testutil/irquery/unittest_opstestwrapper.cpp SUPPORT_LIBS ir-query-test-util trompeloeil)
add_unit_test(unittest_optestwrapper testutil/irquery/unittest_optestwrapper.cpp SUPPORT_LIBS ir-query-test-util trompeloeil)
add_unit_test(unittest_tensorindexmaptestwrapper testutil/irquery/unittest_tensorindexmaptestwrapper.cpp SUPPORT_LIBS ir-query-test-util trompeloeil)
add_unit_test(unittest_tensorindextestwrapper testutil/irquery/unittest_tensorindextestwrapper.cpp SUPPORT_LIBS ir-query-test-util trompeloeil)
add_unit_test(unittest_tensortestwrapper testutil/irquery/unittest_tensortestwrapper.cpp SUPPORT_LIBS ir-query-test-util trompeloeil)

add_unit_test(unittest_op_conv op/conv.cpp)
add_unit_test(unittest_op_convtranspose op/convtranspose.cpp)
add_unit_test(unittest_op_dynamicop op/dynamicop.cpp SUPPORT_LIBS ir-query-test-util)
add_unit_test(unittest_op_sgd2combo op/sgd2combo.cpp)
add_unit_test(unittest_op_varupdate op/varupdate.cpp)
add_unit_test(unittest_op_printtnesor op/printtensor.cpp)

add_unit_test(unittest_ir_deonnxing_regression_tests ir/deonnxing_regression_tests.cpp)
add_unit_test(unittest_ir_tensor_accessors ir/tensor_accessors.cpp)
add_unit_test(unittest_ir_clone_graph ir/clone_graph.cpp SUPPORT_LIBS test-graphs-test-util)
add_unit_test(unittest_ir_executeOpNTimesEveryMTimes ir/executeOpNTimesEveryMTimes.cpp)
add_unit_test(unittest_ir_remove_isolated_graphs ir/remove_isolated_graphs.cpp)

add_unit_test(unittest_stepio_deonnxing_regression_tests stepio/deonnxing_regression_tests.cpp)

add_unit_test(unittest_autodiff transforms/unittest_autodiff.cpp SUPPORT_LIBS ir-query-test-util test-graphs-test-util)
add_unit_test(unittest_overlapio transforms/unittest_overlapio.cpp SUPPORT_LIBS ir-query-test-util test-graphs-test-util)
add_unit_test(unittest_efficient_overlapio transforms/unittest_efficient_overlapio.cpp SUPPORT_LIBS ir-query-test-util test-graphs-test-util)
add_unit_test(unittest_mainloops transforms/unittest_mainloops.cpp VARIANTS Cpu)
add_unit_test(unittest_backwardsgraphcreator transforms/autodiff/unittest_backwardsgraphcreator.cpp)
add_unit_test(unittest_calledgraphgradophelper transforms/autodiff/unittest_calledgraphgradophelper.cpp)
add_unit_test(unittest_subgraph_autodiff transforms/autodiff/unittest_subgraph_autodiff.cpp SUPPORT_LIBS ir-query-test-util)
add_unit_test(unittest_streamingmemory transforms/unittest_streamingmemory.cpp SUPPORT_LIBS ir-query-test-util test-graphs-test-util)
add_unit_test(unittest_decomposeloops transforms/unittest_decomposeloops.cpp SUPPORT_LIBS test-graphs-test-util)
add_unit_test(unittest_explicitrecompute transforms/unittest_explicitrecompute.cpp SUPPORT_LIBS test-graphs-test-util)

add_unit_test(unittest_pipeline transforms/unittest_pipeline.cpp SUPPORT_LIBS test-graphs-test-util)
add_unit_test(unittest_recompute transforms/unittest_recompute_mode.cpp)

add_unit_test(unittest_preautomaticlossscaling transforms/unittest_preautomaticlossscaling.cpp)
add_unit_test(unittest_ensurefp32lossscale transforms/unittest_ensurefp32lossscale.cpp)

add_unit_test(unittest_opattributehelper unittest_opattributehelper.cpp)

add_popart_py_unit_test(test_error)

add_unit_test(unittest_reduceprodce ces/reduceprodce.cpp)

add_unit_test(unittest_complement unittest_complement.cpp)

add_subdirectory("compoundscalarhelper")

add_subdirectory("python")
add_subdirectory("util")
add_subdirectory("willow")
