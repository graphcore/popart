#define BOOST_TEST_MODULE Basic0LogicalIf

#include <../test_runner.hpp>
#include <boost/test/unit_test.hpp>
#include <poponnx/builder.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/inputshapeinfo.hpp>
#include <poponnx/op/add.hpp>
#include <poponnx/op/if.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/tensornames.hpp>
#include <poponnx/tensors.hpp>

using namespace poponnx;

BOOST_AUTO_TEST_CASE(LogicalIf_basic0) {
  TensorInfo info{"FLOAT", std::vector<int64_t>{2, 2}};
  TensorInfo infoBool{"BOOL", std::vector<int64_t>{}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  TestRunner runner;
  runner.patterns.enableInPlace(false);

  runner.buildModel([&inputs, &outputs, info, infoBool](Builder &builder) {
    auto aiOnnx       = builder.aiOnnxOpset9();
    auto in0          = builder.addInputTensor(info);
    auto in1          = builder.addInputTensor(info);
    auto in_condition = builder.addInputTensor(infoBool);

    auto then_branch = [&info, in0, in1](Builder &parent_builder) {
      Builder &builder = parent_builder.createSubgraphBuilder();
      // auto builder = Builder::create();
      auto aiOnnx = builder.aiOnnxOpset9();
      builder.addInputTensorFromHigherScope(in0);
      builder.addInputTensorFromHigherScope(in1);
      auto out = aiOnnx.add({in0, in1});
      builder.addOutputTensor(out);
      return io::getModelFromString(builder.getModelProto()).graph();
    }(builder);

    auto else_branch = [&info, in0, in1](Builder &parent_builder) {
      Builder &builder = parent_builder.createSubgraphBuilder();
      // auto builder = Builder::create();
      auto aiOnnx = builder.aiOnnxOpset9();
      builder.addInputTensorFromHigherScope(in0);
      builder.addInputTensorFromHigherScope(in1);
      auto out = aiOnnx.sub({in0, in1});
      builder.addOutputTensor(out);
      return io::getModelFromString(builder.getModelProto()).graph();
    }(builder);

    auto out =
        aiOnnx.logical_if({in_condition}, 1, else_branch, then_branch)[0];
    builder.addOutputTensor(out);

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
        std::vector<float> expected{-1, -1, -1, -1};
        BOOST_CHECK(data == expected);
      },
      inputs,
      outputs);
}

BOOST_AUTO_TEST_CASE(LogicalIf_scopes0) {
  TensorInfo info{"FLOAT", std::vector<int64_t>{2, 2}};
  TensorInfo infoBool{"BOOL", std::vector<int64_t>{}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  TestRunner runner;
  runner.patterns.enableInPlace(false);

  runner.buildModel([&inputs, &outputs, info, infoBool](Builder &builder) {
    auto aiOnnx       = builder.aiOnnxOpset9();
    auto in0          = builder.addInputTensor(info);
    auto in1          = builder.addInputTensor(info);
    auto in_condition = builder.addInputTensor(infoBool);

    auto then_branch = [&info, in0, in1](Builder &parent_builder) {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromHigherScope(in0);
      builder.addInputTensorFromHigherScope(in1);
      auto out = aiOnnx.add({in0, in1});
      builder.addOutputTensor(out);
      return io::getModelFromString(builder.getModelProto()).graph();
    }(builder);

    auto else_branch = [&info, in0, in1](Builder &parent_builder) {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromHigherScope(in0);
      builder.addInputTensorFromHigherScope(in1);
      auto out = aiOnnx.add({in0, in1});
      builder.addOutputTensor(out);
      return io::getModelFromString(builder.getModelProto()).graph();
    }(builder);

    auto out =
        aiOnnx.logical_if({in_condition}, 1, else_branch, then_branch)[0];
    builder.addOutputTensor(out);

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
        logging::debug("data: {}", data);
        logging::debug("expected: {}", expected);
        BOOST_CHECK(data == expected);
      },
      inputs,
      outputs);

  // Check false branch
  inputs.back().setData<char>({0});
  runner.checkResult(
      [](TestTensor &result) {
        auto data = result.getDataCopy<float>();
        std::vector<float> expected{3, 5, 7, 9};
        logging::debug("data: {}", data);
        logging::debug("expected: {}", expected);
        BOOST_CHECK(data == expected);
      },
      inputs,
      outputs);
}

BOOST_AUTO_TEST_CASE(LogicalIf_scopes1) {
  TensorInfo info{"FLOAT", std::vector<int64_t>{2, 2}};
  TensorInfo infoBool{"BOOL", std::vector<int64_t>{}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  TestRunner runner;
  runner.patterns.enableInPlace(false);

  runner.buildModel([&inputs, &outputs, info, infoBool](Builder &builder) {
    auto aiOnnx       = builder.aiOnnxOpset9();
    auto in0          = builder.addInputTensor(info);
    auto in1          = builder.addInputTensor(info);
    auto in_condition = builder.addInputTensor(infoBool);

    auto then_branch = [&info, in0, in1](Builder &parent_builder) {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromHigherScope(in0);
      builder.addInputTensorFromHigherScope(in1);
      auto out = aiOnnx.add({in0, in1});
      out      = aiOnnx.add({out, out});
      builder.addOutputTensor(out);
      return io::getModelFromString(builder.getModelProto()).graph();
    }(builder);

    auto else_branch = [&info, in0, in1](Builder &parent_builder) {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromHigherScope(in0);
      builder.addInputTensorFromHigherScope(in1);
      auto out = aiOnnx.add({in0, in1});
      builder.addOutputTensor(out);
      return io::getModelFromString(builder.getModelProto()).graph();
    }(builder);

    auto out =
        aiOnnx.logical_if({in_condition}, 1, else_branch, then_branch)[0];
    builder.addOutputTensor(out);

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
        std::vector<float> expected{6, 10, 14, 18};
        logging::debug("data: {}", data);
        logging::debug("expected: {}", expected);
        BOOST_CHECK(data == expected);
      },
      inputs,
      outputs);

  // Check false branch
  inputs.back().setData<char>({0});
  runner.checkResult(
      [](TestTensor &result) {
        auto data = result.getDataCopy<float>();
        std::vector<float> expected{3, 5, 7, 9};
        logging::debug("data: {}", data);
        logging::debug("expected: {}", expected);
        BOOST_CHECK(data == expected);
      },
      inputs,
      outputs);
}
