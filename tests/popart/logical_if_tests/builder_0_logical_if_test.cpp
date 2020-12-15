// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Builder0LogicalIf

#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(LogicalIf_builder0) {

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
  Builder &builder10 = builder0->createSubgraphBuilder();
  auto aiOnnx10      = builder10.aiOnnxOpset9();
  // the name comes from the parent graph
  builder10.addInputTensorFromParentGraph(A);
  builder10.addInputTensorFromParentGraph(B);
  auto out10 = aiOnnx10.add({A, B});
  builder10.addOutputTensor(out10);

  // Graph, level 1, true branch : relu(B)
  Builder &builder11 = builder0->createSubgraphBuilder();
  auto aiOnnx11      = builder11.aiOnnxOpset9();
  builder11.addInputTensorFromParentGraph(B);
  auto out11 = aiOnnx11.relu({B});
  builder11.addOutputTensor(out11);

  auto out_if = aiOnnx0.logical_if(
      {C},
      // number of outputs (must be same along true and false branches)
      1,
      // Builder for false branch
      builder10,
      // Builder for true branch
      builder11);

  auto proto      = builder0->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  auto finalGraph = modelProto.graph();

  // A single "If" NodeProto:
  BOOST_CHECK(modelProto.graph().node_size() == 1);
  BOOST_CHECK(modelProto.graph().node(0).op_type() == "If");
  // The "If" NodeProto has 2 (sub-graph) attributes + 1 for debugIf:
  BOOST_CHECK(modelProto.graph().node(0).attribute_size() == 2 + 1);
}
