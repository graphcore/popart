#define BOOST_TEST_MODULE Basic0LogicalIf

#include <../test_runner.hpp>
#include <boost/test/unit_test.hpp>
#include <poponnx/builder.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/inputshapeinfo.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/tensornames.hpp>

using namespace poponnx;

BOOST_AUTO_TEST_CASE(LogicalIf_basic0) {

  // A ---|
  //      |---- (Add) ----------|
  //      |                     |
  //      |                     |
  // B ===|---- (Relu) -----|   |
  //                        |   |
  //                    ------- If ----- output
  //                    |
  //                    |
  // C (bool) ----------|

  TensorInfo infoData{"FLOAT", std::vector<int64_t>{4, 4}};
  TensorInfo infoBool{"BOOL", std::vector<int64_t>{}};

  // Graph, level 0, (top level)
  auto builder0 = Builder::create();
  auto aiOnnx0  = builder0->aiOnnxOpset9();
  auto A        = builder0->addInputTensor(infoData);
  auto B        = builder0->addInputTensor(infoData);
  auto C        = builder0->addInputTensor(infoBool);

  // Graph, level 1, false branch : A + B
  auto builder10 = Builder::create();
  auto aiOnnx10  = builder10->aiOnnxOpset9();
  // the name comes from the parent graph
  builder10->addInputTensorFromParentGraph(infoData, A);
  builder10->addInputTensorFromParentGraph(infoData, B);
  auto out10 = aiOnnx10.add({A, B});
  builder10->addOutputTensor(out10);

  // Graph, level 1, true branch : relu(B)
  auto builder11 = Builder::create();
  auto aiOnnx11  = builder11->aiOnnxOpset9();
  builder11->addInputTensorFromParentGraph(infoData, B);
  auto out11 = aiOnnx11.relu({B});
  builder11->addOutputTensor(out11);

  auto out_if = aiOnnx0.logical_if(
      {C},
      // number of outputs (must be same along true and false branches)
      1,
      // GraphProto for false branch
      io::getModelFromString(builder10->getModelProto()).graph(),
      // GraphProto for true branch
      io::getModelFromString(builder11->getModelProto()).graph());

  auto proto      = builder0->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  auto finalGraph = modelProto.graph();

  // A single "If" NodeProto:
  BOOST_CHECK(modelProto.graph().node_size() == 1);
  BOOST_CHECK(modelProto.graph().node(0).op_type() == "If");
  // The "If" NodeProto has 2 (sub-graph) attributes:
  BOOST_CHECK(modelProto.graph().node(0).attribute_size() == 2);
}

BOOST_AUTO_TEST_CASE(LogicalIf_basic1) {
  TensorInfo info{"FLOAT", std::vector<int64_t>{2, 2}};
  TensorInfo infoBool{"BOOL", std::vector<int64_t>{}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  TestRunner runner;
  runner.enableInPlace = false;

  runner.buildModel([&](Builder &builder) {
    auto aiOnnx       = builder.aiOnnxOpset9();
    auto in0          = builder.addInputTensor(info);
    auto in1          = builder.addInputTensor(info);
    auto in_condition = builder.addInputTensor(infoBool);

    auto then_branch = [&]() {
      auto builder = Builder::create();
      auto aiOnnx  = builder->aiOnnxOpset9();
      builder->addInputTensorFromParentGraph(info, in0);
      builder->addInputTensorFromParentGraph(info, in1);
      auto out = aiOnnx.add({in0, in1});
      builder->addOutputTensor(out);
      return io::getModelFromString(builder->getModelProto()).graph();
    }();

    auto else_branch = [&]() {
      auto builder = Builder::create();
      auto aiOnnx  = builder->aiOnnxOpset9();
      builder->addInputTensorFromParentGraph(info, in0);
      builder->addInputTensorFromParentGraph(info, in1);
      auto out = aiOnnx.sub({in0, in1});
      builder->addOutputTensor(out);
      return io::getModelFromString(builder->getModelProto()).graph();
    }();

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
