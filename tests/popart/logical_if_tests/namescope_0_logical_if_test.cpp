#define BOOST_TEST_MODULE Namescope0LogicalIf

#include <../test_runner.hpp>
#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(LogicalIf_namescope0) {
  TensorInfo info{"FLOAT", std::vector<int64_t>{2, 2}};
  TensorInfo infoBool{"BOOL", std::vector<int64_t>{}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  TestRunner runner;
  runner.patterns.enableInPlace(false);

  runner.buildModel([&](Builder &builder) {
    auto aiOnnx       = builder.aiOnnxOpset9();
    auto in0          = builder.addInputTensor(info);
    auto in1          = builder.addInputTensor(info);
    auto in_condition = builder.addInputTensor(infoBool);

    // in0 + in1
    auto then_branch = [&info, in0, in1](Builder &parent_builder) {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromHigherScope(in0);
      builder.addInputTensorFromHigherScope(in1);

      // could get identical name as else_branch
      auto out0 = aiOnnx.add({in0, in1});
      builder.addOutputTensor(out0);
      return io::getModelFromString(builder.getModelProto()).graph();
    }(builder);

    // 2*(in0 + in1)
    auto else_branch = [&info, in0, in1](Builder &parent_builder) {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      auto aiGraphcore = builder.aiGraphcoreOpset1();
      builder.addInputTensorFromHigherScope(in0);
      builder.addInputTensorFromHigherScope(in1);

      // could get identical name as then_branch
      auto out0 = aiOnnx.add({in0, in1});
      auto out1 = aiGraphcore.scale({out0}, 2);
      builder.addOutputTensor(out1);
      return io::getModelFromString(builder.getModelProto()).graph();
    }(builder);

    auto out =
        aiOnnx.logical_if({in_condition}, 1, else_branch, then_branch)[0];
    inputs.push_back(
        TestTensor::create<float>(in0, {1, 2, 3, 4}, info.shape()));
    inputs.push_back(
        TestTensor::create<float>(in1, {2, 3, 4, 5}, info.shape()));
    inputs.push_back(TestTensor::create<bool>(in_condition, infoBool.shape()));
    outputs.push_back(TestTensor::create<float>(out, info.shape()));

    return out;
  });

  // Check true branch
  inputs.back().setData<char>({1});
  runner.checkResult(
      [](TestTensor &result) {
        auto data = result.getDataCopy<float>();
        std::vector<float> expected{3, 5, 7, 9};
        BOOST_CHECK(data == expected);
      },
      inputs,
      outputs);

  // Check false branch
  inputs.back().setData<char>({0});
  runner.checkResult(
      [](TestTensor &result) {
        auto data = result.getDataCopy<float>();
        std::vector<float> expected{6, 10, 14, 18};
        BOOST_CHECK(data == expected);
      },
      inputs,
      outputs);
}
