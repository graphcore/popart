// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Basic0LogicalIf

#include <../test_runner.hpp>
#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
#include <popart/builder.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/op/add.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/if.hpp>
#include <popart/op/l1.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>

using namespace popart;

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

    auto &then_branch = [in0, in1](Builder &parent_builder) -> Builder & {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromParentGraph(in0);
      builder.addInputTensorFromParentGraph(in1);
      auto out = aiOnnx.add({in0, in1});
      builder.addOutputTensor(out);
      return builder;
    }(builder);

    auto &else_branch = [in0, in1](Builder &parent_builder) -> Builder & {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromParentGraph(in0);
      builder.addInputTensorFromParentGraph(in1);
      auto out = aiOnnx.sub({in0, in1});
      builder.addOutputTensor(out);
      return builder;
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

    auto &then_branch = [in0, in1](Builder &parent_builder) -> Builder & {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromParentGraph(in0);
      builder.addInputTensorFromParentGraph(in1);
      auto out = aiOnnx.add({in0, in1});
      builder.addOutputTensor(out);
      return builder;
    }(builder);

    auto &else_branch = [in0, in1](Builder &parent_builder) -> Builder & {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromParentGraph(in0);
      builder.addInputTensorFromParentGraph(in1);
      auto out = aiOnnx.add({in0, in1});
      builder.addOutputTensor(out);
      return builder;
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

    auto &then_branch = [in0, in1](Builder &parent_builder) -> Builder & {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromParentGraph(in0);
      builder.addInputTensorFromParentGraph(in1);
      auto out = aiOnnx.add({in0, in1});
      out      = aiOnnx.add({out, out});
      builder.addOutputTensor(out);
      return builder;
    }(builder);

    auto &else_branch = [in0, in1](Builder &parent_builder) -> Builder & {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromParentGraph(in0);
      builder.addInputTensorFromParentGraph(in1);
      auto out = aiOnnx.add({in0, in1});
      builder.addOutputTensor(out);
      return builder;
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

// Check that the generation of IfGradOp and backward graphs works
BOOST_AUTO_TEST_CASE(LogicalIf_train0) {
  TensorInfo info{"FLOAT", std::vector<int64_t>{2, 2}};
  TensorInfo infoBool{"BOOL", std::vector<int64_t>{}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  float in0Data[4]        = {1, 2, 3, 4};
  ConstVoidData in0CVData = {in0Data, info};

  float in1Data[4]        = {2, 3, 4, 5};
  ConstVoidData in1CVData = {in1Data, info};

  TestRunner runner;
  // Note: in D56156 it was identified this test was assuming the weight updates
  // were not happening and comparing "stale" outputs. The test was updated with
  // learning rate 0 to stop the weight update.
  // We are only interested in values of gradients and outputs, we don't want a
  // weight update to happen.
  runner.optimizer =
      std::unique_ptr<SGD>(new SGD({{"defaultLearningRate", {0.0f, false}}}));
  runner.patterns.enableInPlace(false);
  runner.isTraining = true;

  TensorId in0;
  TensorId in1;
  TensorId out;

  runner.buildModel([&inputs,
                     &outputs,
                     &runner,
                     &in0CVData,
                     &in1CVData,
                     &in0,
                     &in1,
                     &out,
                     info,
                     infoBool](Builder &builder) {
    auto aiOnnx       = builder.aiOnnxOpset9();
    in0               = builder.addInitializedInputTensor(in0CVData);
    in1               = builder.addInitializedInputTensor(in1CVData);
    auto in_condition = builder.addInputTensor(infoBool);

    auto &then_branch = [in0, in1](Builder &parent_builder) -> Builder & {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromParentGraph(in0);
      builder.addInputTensorFromParentGraph(in1);
      auto out = aiOnnx.add({in0, in1});
      builder.addOutputTensor(out);
      return builder;
    }(builder);

    auto &else_branch = [in0, in1](Builder &parent_builder) -> Builder & {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromParentGraph(in0);
      builder.addInputTensorFromParentGraph(in1);
      auto out = aiOnnx.add({in0, in1});
      builder.addOutputTensor(out);
      return builder;
    }(builder);

    out = aiOnnx.logical_if({in_condition}, 1, else_branch, then_branch)[0];
    auto l1 =
        builder.aiGraphcoreOpset1().l1loss({out}, 0.1, ReductionType::Sum);

    runner.anchors.insert({getGradId(in0), AnchorReturnType("All")});
    runner.anchors.insert({getGradId(in1), AnchorReturnType("All")});

    inputs.push_back(TestTensor::create<bool>(in_condition, infoBool.shape()));
    outputs.push_back(TestTensor::create<float>(out, info.shape()));
    outputs.push_back(TestTensor::create<float>(getGradId(in0), info.shape()));
    outputs.push_back(TestTensor::create<float>(getGradId(in1), info.shape()));

    runner.loss = l1;

    return out;
  });

  // Reference was generated with the below python
  // >>> a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32,
  // requires_grad=True)
  // >>> b = torch.tensor([[2, 3], [4, 5]], dtype=torch.float32,
  // requires_grad=True)
  // >>> o = a + b
  // >>> o
  // tensor([[3., 5.],
  //         [7., 9.]], grad_fn=<AddBackward0>)
  // >>> o.backward(torch.ones(2, 2) * 0.1)
  // >>> a.grad
  // tensor([[0.1000, 0.1000],
  //         [0.1000, 0.1000]])
  // >>> b.grad
  // tensor([[0.1000, 0.1000],
  //         [0.1000, 0.1000]])
  auto resultChecker = [&in0, &in1, &out](std::vector<TestTensor> &results) {
    BOOST_CHECK_EQUAL(results.size(), 3);
    std::map<TensorId, std::vector<float>> expected{
        // output is the same for both runs
        {out, {3, 5, 7, 9}},
        {getGradId(in0), {0.1, 0.1, 0.1, 0.1}},
        {getGradId(in1), {0.1, 0.1, 0.1, 0.1}}};

    for (auto &t : results) {
      auto data = t.getDataCopy<float>();
      auto e    = expected.at(t.id);
      BOOST_CHECK(data == e);
    }
  };

  // Check true branch
  inputs.back().setData<char>({1});
  runner.checkResults(resultChecker, inputs, outputs);

  // Check false branch
  inputs.back().setData<char>({0});
  runner.checkResults(resultChecker, inputs, outputs);
}

// Same as LogicalIf_train0, but then and else branchs are now different
BOOST_AUTO_TEST_CASE(LogicalIf_train1) {
  TensorInfo info{"FLOAT", std::vector<int64_t>{2, 2}};
  TensorInfo infoBool{"BOOL", std::vector<int64_t>{}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  float in0Data[4]        = {1, 2, 3, 4};
  ConstVoidData in0CVData = {in0Data, info};

  float in1Data[4]        = {2, 3, 4, 5};
  ConstVoidData in1CVData = {in1Data, info};

  TestRunner runner;
  // We are only interested in values of gradients and outputs, we don't want a
  // weight update to happen.
  runner.optimizer =
      std::unique_ptr<SGD>(new SGD({{"defaultLearningRate", {0.0f, false}}}));
  runner.opts.dotChecks.insert("Final");
  runner.opts.separateCallOpPdfs = false;
  runner.patterns.enableInPlace(false);
  runner.patterns.enableSubtractArg1GradOp(true);
  runner.isTraining = true;

  TensorId in0;
  TensorId in1;
  TensorId out;

  runner.buildModel([&inputs,
                     &outputs,
                     &runner,
                     &in0CVData,
                     &in1CVData,
                     &in0,
                     &in1,
                     &out,
                     &info,
                     infoBool](Builder &builder) {
    auto aiOnnx       = builder.aiOnnxOpset9();
    in0               = builder.addInitializedInputTensor(in0CVData);
    in1               = builder.addInitializedInputTensor(in1CVData);
    auto in_condition = builder.addInputTensor(infoBool);

    auto &then_branch = [in0, in1](Builder &parent_builder) -> Builder & {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromParentGraph(in0);
      builder.addInputTensorFromParentGraph(in1);
      auto out = aiOnnx.add({in0, in1});
      builder.addOutputTensor(out);
      return builder;
    }(builder);

    auto &else_branch = [in0, in1](Builder &parent_builder) -> Builder & {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromParentGraph(in0);
      builder.addInputTensorFromParentGraph(in1);
      auto out = aiOnnx.sub({in0, in1});
      builder.addOutputTensor(out);
      return builder;
    }(builder);

    out = aiOnnx.logical_if({in_condition}, 1, else_branch, then_branch)[0];
    auto l1 =
        builder.aiGraphcoreOpset1().l1loss({out}, 0.1, ReductionType::Sum);

    runner.anchors.insert({getGradId(in0), AnchorReturnType("All")});
    runner.anchors.insert({getGradId(in1), AnchorReturnType("All")});

    inputs.push_back(TestTensor::create<bool>(in_condition, infoBool.shape()));
    outputs.push_back(TestTensor::create<float>(out, info.shape()));
    outputs.push_back(TestTensor::create<float>(getGradId(in0), info.shape()));
    outputs.push_back(TestTensor::create<float>(getGradId(in1), info.shape()));

    runner.loss = l1;

    return out;
  });

  // Check true branch
  // Reference was generated with the below python
  // >>> a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32,
  // requires_grad=True)
  // >>> b = torch.tensor([[2, 3], [4, 5]], dtype=torch.float32,
  // requires_grad=True)
  // >>> o = a + b
  // >>> o
  // tensor([[3., 5.],
  //         [7., 9.]], grad_fn=<AddBackward0>)
  // >>> o.backward(torch.ones(2, 2) * 0.1)
  // >>> a.grad
  // tensor([[0.1000, 0.1000],
  //         [0.1000, 0.1000]])
  // >>> b.grad
  // tensor([[0.1000, 0.1000],
  //         [0.1000, 0.1000]])
  inputs.back().setData<char>({1});
  runner.checkResults(
      [&in0, &in1, &out](std::vector<TestTensor> &results) {
        BOOST_CHECK_EQUAL(results.size(), 3);
        std::map<TensorId, std::vector<float>> expected{
            {out, {3, 5, 7, 9}},
            {getGradId(in0), {0.1, 0.1, 0.1, 0.1}},
            {getGradId(in1), {0.1, 0.1, 0.1, 0.1}}};

        for (auto &t : results) {
          auto data = t.getDataCopy<float>();
          auto e    = expected.at(t.id);
          logging::debug("BOOST_CHECK({} == {});", data, e);
          BOOST_CHECK(data == e);
        }
      },
      inputs,
      outputs);

  // Check false branch
  // Reference was generated with the below python
  // >>> a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32,
  // requires_grad=True)
  // >>> b = torch.tensor([[2, 3], [4, 5]], dtype=torch.float32,
  // requires_grad=True)
  // >>> o = a - b
  // >>> o
  // tensor([[-1., -1.],
  //         [-1., -1.]], grad_fn=<SubBackward0>)
  // >>> o.backward(torch.ones(2, 2) * -0.1)
  // >>> a.grad
  // tensor([[-0.1000, -0.1000],
  //         [-0.1000, -0.1000]])
  // >>> b.grad
  // tensor([[0.1000, 0.1000],
  //         [0.1000, 0.1000]])
  inputs.back().setData<char>({0});
  runner.checkResults(
      [&in0, &in1, &out](std::vector<TestTensor> &results) {
        BOOST_CHECK_EQUAL(results.size(), 3);
        std::map<TensorId, std::vector<float>> expected{
            {out, {-1, -1, -1, -1}},
            {getGradId(in0), {-0.1, -0.1, -0.1, -0.1}},
            {getGradId(in1), {0.1, 0.1, 0.1, 0.1}}};

        for (auto &t : results) {
          auto data = t.getDataCopy<float>();
          auto e    = expected.at(t.id);
          logging::debug("Checking for t.id {}", t.id);
          logging::debug("BOOST_CHECK({} == {});", data, e);
          BOOST_CHECK(data == e);
        }
      },
      inputs,
      outputs);
}

// Same as LogicalIf_train0, but one branch requires inputs of forward branch
// for computing gradient.
BOOST_AUTO_TEST_CASE(LogicalIf_train2) {
  TensorInfo info{"FLOAT", std::vector<int64_t>{2, 2}};
  TensorInfo infoBool{"BOOL", std::vector<int64_t>{}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  float in0Data[4]        = {1, 2, 3, 4};
  ConstVoidData in0CVData = {in0Data, info};

  float in1Data[4]        = {2, 3, 4, 5};
  ConstVoidData in1CVData = {in1Data, info};

  TestRunner runner;
  // We are only interested in values of gradients and outputs, we don't want a
  // weight update to happen.
  runner.optimizer =
      std::unique_ptr<SGD>(new SGD({{"defaultLearningRate", {0.0f, false}}}));
  runner.patterns.enableInPlace(false);
  runner.patterns.enableMulArgGradOp(true);
  runner.isTraining = true;

  TensorId in0;
  TensorId in1;
  TensorId out;

  runner.buildModel([&inputs,
                     &outputs,
                     &runner,
                     &in0CVData,
                     &in1CVData,
                     &in0,
                     &in1,
                     &out,
                     info,
                     infoBool](Builder &builder) {
    auto aiOnnx       = builder.aiOnnxOpset9();
    in0               = builder.addInitializedInputTensor(in0CVData);
    in1               = builder.addInitializedInputTensor(in1CVData);
    auto in_condition = builder.addInputTensor(infoBool);

    auto &then_branch = [in0, in1](Builder &parent_builder) -> Builder & {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromParentGraph(in0);
      builder.addInputTensorFromParentGraph(in1);
      auto out = aiOnnx.add({in0, in1});
      builder.addOutputTensor(out);
      return builder;
    }(builder);

    auto &else_branch = [in0, in1](Builder &parent_builder) -> Builder & {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromParentGraph(in0);
      builder.addInputTensorFromParentGraph(in1);
      auto out = aiOnnx.mul({in0, in1});
      builder.addOutputTensor(out);
      return builder;
    }(builder);

    out = aiOnnx.logical_if({in_condition}, 1, else_branch, then_branch)[0];
    auto l1 =
        builder.aiGraphcoreOpset1().l1loss({out}, 0.1, ReductionType::Sum);

    runner.anchors.insert({getGradId(in0), AnchorReturnType("All")});
    runner.anchors.insert({getGradId(in1), AnchorReturnType("All")});

    inputs.push_back(TestTensor::create<bool>(in_condition, infoBool.shape()));
    outputs.push_back(TestTensor::create<float>(out, info.shape()));
    outputs.push_back(TestTensor::create<float>(getGradId(in0), info.shape()));
    outputs.push_back(TestTensor::create<float>(getGradId(in1), info.shape()));

    runner.loss = l1;

    return out;
  });

  // Check true branch
  // Reference was generated with the below python
  // >>> a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32,
  // requires_grad=True)
  // >>> b = torch.tensor([[2, 3], [4, 5]], dtype=torch.float32,
  // requires_grad=True)
  // >>> o = a + b
  // >>> o
  // tensor([[3., 5.],
  //         [7., 9.]], grad_fn=<AddBackward0>)
  // >>> o.backward(torch.ones(2, 2) * 0.1)
  // >>> a.grad
  // tensor([[0.1000, 0.1000],
  //         [0.1000, 0.1000]])
  // >>> b.grad
  // tensor([[0.1000, 0.1000],
  //         [0.1000, 0.1000]])
  inputs.back().setData<char>({1});
  runner.checkResults(
      [&in0, &in1, &out](std::vector<TestTensor> &results) {
        BOOST_CHECK_EQUAL(results.size(), 3);
        std::map<TensorId, std::vector<float>> expected{
            {out, {3, 5, 7, 9}},
            {getGradId(in0), {0.1, 0.1, 0.1, 0.1}},
            {getGradId(in1), {0.1, 0.1, 0.1, 0.1}}};

        for (auto &t : results) {
          auto data = t.getDataCopy<float>();
          auto e    = expected.at(t.id);
          logging::debug("BOOST_CHECK({} == {});", data, e);
          BOOST_CHECK(data == e);
        }
      },
      inputs,
      outputs);

  // Check false branch
  // Reference was generated with the below python
  // >>> a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32,
  // requires_grad=True)
  // >>> b = torch.tensor([[2, 3], [4, 5]], dtype=torch.float32,
  // requires_grad=True)
  // >>> o = a * b
  // >>> o
  // tensor([[ 2.,  6.],
  //         [12., 20.]], grad_fn=<MulBackward0>)
  // >>> o.backward(torch.ones(2, 2) * 0.1)
  // >>> a.grad
  // tensor([[0.2000, 0.3000],
  //         [0.4000, 0.5000]])
  // >>> b.grad
  // tensor([[0.1000, 0.2000],
  //         [0.3000, 0.4000]])
  inputs.back().setData<char>({0});
  runner.checkResults(
      [&in0, &in1, &out](std::vector<TestTensor> &results) {
        BOOST_CHECK_EQUAL(results.size(), 3);
        std::map<TensorId, std::vector<float>> expected{
            {out, {2, 6, 12, 20}},
            {getGradId(in0), {0.2, 0.3, 0.4, 0.5}},
            {getGradId(in1), {0.1, 0.2, 0.3, 0.4}}};

        for (auto &t : results) {
          auto data = t.getDataCopy<float>();
          auto e    = expected.at(t.id);
          logging::debug("BOOST_CHECK({} == {});", data, e);
          BOOST_CHECK(data == e);
        }
      },
      inputs,
      outputs);
}

// Same as LogicalIf_train0, but then and else branchs have different number of
// inputs
BOOST_AUTO_TEST_CASE(LogicalIf_train3) {
  TensorInfo info{"FLOAT", std::vector<int64_t>{1}};
  TensorInfo infoBool{"BOOL", std::vector<int64_t>{}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  float in0Data[1]        = {1};
  ConstVoidData in0CVData = {in0Data, info};

  float in1Data[1]        = {2};
  ConstVoidData in1CVData = {in1Data, info};

  TestRunner runner;
  // We are only interested in values of gradients and outputs, we don't want a
  // weight update to happen.
  runner.optimizer =
      std::unique_ptr<SGD>(new SGD({{"defaultLearningRate", {0.0f, false}}}));
  runner.patterns.enableInPlace(false);
  runner.isTraining = true;

  TensorId in0;
  TensorId in1;
  TensorId out;

  runner.buildModel([&inputs,
                     &outputs,
                     &runner,
                     &in0CVData,
                     &in1CVData,
                     &in0,
                     &in1,
                     &out,
                     info,
                     infoBool](Builder &builder) {
    auto aiOnnx       = builder.aiOnnxOpset9();
    in0               = builder.addInitializedInputTensor(in0CVData);
    in1               = builder.addInitializedInputTensor(in1CVData);
    auto in_condition = builder.addInputTensor(infoBool);

    auto &then_branch = [in0, in1](Builder &parent_builder) -> Builder & {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromParentGraph(in0);
      builder.addInputTensorFromParentGraph(in1);
      auto out = aiOnnx.add({in0, in1});
      builder.addOutputTensor(out);
      return builder;
    }(builder);

    auto &else_branch = [in0, in1](Builder &parent_builder) -> Builder & {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromParentGraph(in0);
      auto out = aiOnnx.add({in0, in0});
      builder.addOutputTensor(out);
      return builder;
    }(builder);

    out = aiOnnx.logical_if({in_condition}, 1, else_branch, then_branch)[0];
    auto l1 =
        builder.aiGraphcoreOpset1().l1loss({out}, 0.1, ReductionType::Sum);

    runner.anchors.insert({getGradId(in0), AnchorReturnType("All")});
    runner.anchors.insert({getGradId(in1), AnchorReturnType("All")});

    inputs.push_back(TestTensor::create<bool>(in_condition, infoBool.shape()));
    outputs.push_back(TestTensor::create<float>(out, info.shape()));
    outputs.push_back(TestTensor::create<float>(getGradId(in0), info.shape()));
    outputs.push_back(TestTensor::create<float>(getGradId(in1), info.shape()));

    runner.loss = l1;

    return out;
  });

  auto resultChecker = [](std::vector<TestTensor> &results) {
    logging::debug("Results:");
    for (auto &t : results) {
      auto data = t.getDataCopy<float>();
      logging::debug("  {}: {}", t.id, data);
    }
  };

  (void)resultChecker;

  // Check true branch
  inputs.back().setData<char>({1});
  runner.checkResults(
      [&in0, &in1, &out](std::vector<TestTensor> &results) {
        BOOST_CHECK_EQUAL(results.size(), 3);
        std::map<TensorId, std::vector<float>> expected{
            {out, {3}}, {getGradId(in0), {0.1}}, {getGradId(in1), {0.1}}};

        for (auto &t : results) {
          auto data = t.getDataCopy<float>();
          auto e    = expected.at(t.id);
          logging::debug("BOOST_CHECK({} == {});", data, e);
          BOOST_CHECK(data == e);
        }
      },
      inputs,
      outputs);

  // Check false branch
  inputs.back().setData<char>({0});
  runner.checkResults(
      [&in0, &in1, &out](std::vector<TestTensor> &results) {
        BOOST_CHECK_EQUAL(results.size(), 3);
        std::map<TensorId, std::vector<float>> expected{
            {out, {2}}, {getGradId(in0), {0.2}}, {getGradId(in1), {0.0}}};

        for (auto &t : results) {
          auto data = t.getDataCopy<float>();
          auto e    = expected.at(t.id);
          logging::debug("BOOST_CHECK({} == {});", data, e);
          BOOST_CHECK(data == e);
        }
      },
      inputs,
      outputs);
}

BOOST_AUTO_TEST_CASE(LogicalIf_inputs_differ0) {
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

    auto &then_branch = [in0, in1](Builder &parent_builder) -> Builder & {
      Builder &builder = parent_builder.createSubgraphBuilder();
      // auto builder = Builder::create();
      auto aiOnnx = builder.aiOnnxOpset9();
      builder.addInputTensorFromParentGraph(in0);
      auto out = aiOnnx.add({in0, in0});
      builder.addOutputTensor(out);
      return builder;
    }(builder);

    auto &else_branch = [in0, in1](Builder &parent_builder) -> Builder & {
      Builder &builder = parent_builder.createSubgraphBuilder();
      // auto builder = Builder::create();
      auto aiOnnx = builder.aiOnnxOpset9();
      builder.addInputTensorFromParentGraph(in0);
      builder.addInputTensorFromParentGraph(in1);
      auto out = aiOnnx.add({in0, in1});
      builder.addOutputTensor(out);
      return builder;
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
        std::vector<float> expected{2, 4, 6, 8};
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
        BOOST_CHECK(data == expected);
      },
      inputs,
      outputs);
}

BOOST_AUTO_TEST_CASE(LogicalIf_inputs_differ_train0) {
  TensorInfo info{"FLOAT", std::vector<int64_t>{2, 2}};
  TensorInfo infoBool{"BOOL", std::vector<int64_t>{}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  float in0Data[4]        = {1, 2, 3, 4};
  ConstVoidData in0CVData = {in0Data, info};

  float in1Data[4]        = {2, 3, 4, 5};
  ConstVoidData in1CVData = {in1Data, info};

  TestRunner runner;
  // We are only interested in values of gradients and outputs, we don't want a
  // weight update to happen.
  runner.optimizer =
      std::unique_ptr<SGD>(new SGD({{"defaultLearningRate", {0.0f, false}}}));
  runner.patterns.enableInPlace(false);
  runner.patterns.enableSubtractArg1GradOp(true);
  runner.isTraining = true;

  runner.buildModel(
      [&inputs, &outputs, &runner, &in0CVData, &in1CVData, info, infoBool](
          Builder &builder) {
        auto aiOnnx       = builder.aiOnnxOpset9();
        auto in0          = builder.addInitializedInputTensor(in0CVData);
        auto in1          = builder.addInitializedInputTensor(in1CVData);
        auto in_condition = builder.addInputTensor(infoBool);

        auto &then_branch = [in0, in1](Builder &parent_builder) -> Builder & {
          Builder &builder = parent_builder.createSubgraphBuilder();
          auto aiOnnx      = builder.aiOnnxOpset9();
          builder.addInputTensorFromParentGraph(in0);
          auto out = aiOnnx.add({in0, in0});
          builder.addOutputTensor(out);
          return builder;
        }(builder);

        auto &else_branch = [in0, in1](Builder &parent_builder) -> Builder & {
          Builder &builder = parent_builder.createSubgraphBuilder();
          auto aiOnnx      = builder.aiOnnxOpset9();
          builder.addInputTensorFromParentGraph(in0);
          builder.addInputTensorFromParentGraph(in1);
          auto out = aiOnnx.add({in0, in1});
          builder.addOutputTensor(out);
          return builder;
        }(builder);

        auto out =
            aiOnnx.logical_if({in_condition}, 1, else_branch, then_branch)[0];
        auto l1 =
            builder.aiGraphcoreOpset1().l1loss({out}, 0.1, ReductionType::Sum);

        inputs.push_back(
            TestTensor::create<bool>(in_condition, infoBool.shape()));
        outputs.push_back(TestTensor::create<float>(out, info.shape()));

        runner.loss = l1;

        return out;
      });

  // Check true branch
  inputs.back().setData<char>({1});
  runner.checkResult(
      [](TestTensor &result) {
        auto data = result.getDataCopy<float>();
        std::vector<float> expected{2, 4, 6, 8};
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
        BOOST_CHECK(data == expected);
      },
      inputs,
      outputs);
}
