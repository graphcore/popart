// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ConvTransposePatternTests
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <map>
#include <memory>
#include <sys/types.h>
#include <utility>

#include "popart/graph.hpp"
#include "popart/ir.hpp"
#include "popart/onnxoperators.gen.hpp"
#include "popart/op/conv.hpp"
#include "popart/op/convtranspose.hpp"
#include "popart/tensorindex.hpp"

#include "popart/op/convbase.hpp"
#include <popart/patterns/convtransposepattern.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(TestConvTransposeConnectsInputs) {
  // Set up a basic graph with one conv transpose op
  Ir ir;
  Graph &graph = ir.getMainGraph();

  auto convAttr = Attributes();
  auto convOpts =
      MultiConvOptions(ir.getSessionOptions().convolutionOptions, convAttr);

  Op::Settings settings(graph, "test_convtranspose");
  auto autoPad = AutoPad::NOTSET;

  Tensor dataTensor("data", popart::TensorType::ActGrad, graph);
  dataTensor.info.set(popart::DataType::FLOAT8_143, {1, 1, 2, 2});

  Tensor weightTensor("weight", popart::TensorType::ActGrad, graph);
  weightTensor.info.set(popart::DataType::FLOAT8_152, {1, 1, 2, 2});

  Tensor log2Scale("log2Scale", popart::TensorType::ActGrad, graph);
  log2Scale.info.set(popart::DataType::INT32, {});

  Tensor out("output", popart::TensorType::ActGrad, graph);

  std::vector<u_int8_t> data(dataTensor.info.nelms(), static_cast<u_int8_t>(0));
  std::vector<u_int8_t> weight(weightTensor.info.nelms(),
                               static_cast<u_int8_t>(0));
  std::vector<u_int8_t> l2s(log2Scale.info.nelms(), static_cast<u_int8_t>(0));

  graph.getTensors().addVarInit(dataTensor.id, dataTensor.info, data.data());
  graph.getTensors().addVarInit(
      weightTensor.id, weightTensor.info, weight.data());
  graph.getTensors().addVarInit(log2Scale.id, log2Scale.info, l2s.data());

  auto inputs = std::map<InIndex, TensorId>{
      {ConvTransposeOp::getInIndex(), dataTensor.id},
      {ConvTransposeOp::getWeightsInIndex(), weightTensor.id},
      {ConvTransposeOp::getLog2ScaleInIndex(), log2Scale.id}};

  // create the conv transpose op
  auto createdOp = graph.createConnectedOp<ConvTransposeOp>(
      inputs,
      {{ConvTransposeOp::getOutIndex(), out.id}},
      Onnx::Operators::Conv_1,
      settings,
      std::vector<int64_t>(),
      std::vector<int64_t>(),
      std::vector<int64_t>(),
      1,
      autoPad,
      std::vector<int64_t>(),
      std::vector<int64_t>(),
      convOpts);

  BOOST_REQUIRE(createdOp->isPow2ScaledConvTranspose());

  // apply the pattern to transform the conv transpose to a ConvWeightsFlipOp
  // Followed by a conv.
  ConvTransposePattern pattern;
  pattern.apply(createdOp);

  // Find the new conv op in the graph
  ConvOp *newOp;
  auto opIds = graph.getOpIds();
  for (auto &pair : graph.getOps()) {
    if (pair.second->isConvertibleTo<ConvOp>()) {
      newOp = dynamic_cast<ConvOp *>(pair.second.get());
      break;
    }
  }

  // check the conv op received the inputs from the conv transpose op
  BOOST_CHECK(newOp->isPow2ScaledConv());
  BOOST_CHECK(newOp->hasInput(ConvOp::getDataInIndex()));
  BOOST_CHECK(newOp->hasInput(ConvOp::getWeightsInIndex()));
  BOOST_CHECK(newOp->hasInput(ConvOp::getLog2ScaleInIndex()));

  // don't check the weight tensor's id because it was produced by
  // convweightsflip op, so its id will be different
  BOOST_CHECK(newOp->input->tensor(ConvOp::getDataInIndex())->id ==
              dataTensor.id);
  BOOST_CHECK(newOp->input->tensor(ConvOp::getLog2ScaleInIndex())->id ==
              log2Scale.id);
}
