// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE TestPatternsOptimizerDecompose
#include <boost/test/unit_test.hpp>

// To access gradUnscale
#define protected public
#include <popart/patterns/adamdecompose.hpp>
#undef protected

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/adamcombo.hpp>
#include <popart/op/cast.hpp>
#include <popart/optimizervalue.hpp>
#include <popart/patterns/patterns.hpp>

using namespace popart;

namespace {
void addTensor(Graph &g, TensorId tid, DataType type, Shape shape) {
  TensorInfo info{type, shape};
  if (type == DataType::FLOAT) {
    std::vector<float> d(info.nelms(), static_cast<float>(0));
    g.getTensors().addVarInit(tid, info, d.data());
  } else if (type == DataType::FLOAT16) {
    std::vector<float16_t> d(info.nelms(), static_cast<float16_t>(0));
    g.getTensors().addVarInit(tid, info, d.data());
  }
}
} // namespace

BOOST_AUTO_TEST_CASE(TestGradUnscaleTypeMismatch) {
  // In the case where any of the OptimizerStateTensors in Adam are not fp32 and
  // the weight is fp16, the gradient to be unscaled will be fp16. All
  // OptimizerTensors are stored as fp32. This test checks a cast is added when
  // gradUnscale multiplies the fp32 optimizerTensor with the fp16 grad. It also
  // checks the output of the cast is not an OptimizerTensor to avoid any
  // interaction with optimizerFromHost.
  Ir ir;
  Graph &graph = ir.getMainGraph();

  AdamDecompose pattern;

  TensorId wId = "w";
  addTensor(graph, wId, DataType::FLOAT16, {4});
  TensorId gradId = "grad";
  addTensor(graph, gradId, DataType::FLOAT16, {4});
  TensorId gsId = reservedDefaultAdamGradientScalingPrefix();
  addTensor(graph, gsId, DataType::FLOAT, {1});

  const auto adam = graph.createConnectedOp<AdamComboOp>(
      {{AdamComboOp::getVarToUpdateInIndex(), wId},
       {AdamComboOp::getUpdaterInIndex(), gradId},
       {AdamComboOp::getGsInIndex(), gsId}},
      {{AdamComboOp::getUpdatedVarOutIndex(),
        ir.createIntermediateTensorId(wId)}},
      OptimizerValue{},
      OptimizerValue{},
      OptimizerValue{},
      OptimizerValue{},
      OptimizerValue{},
      OptimizerValue{},
      OptimizerValue{},
      OptimizerValue{0.0f, false}, // Variable Gradient Scaling
      AdamMode::Adam,
      WeightDecayMode::Decay,
      false,
      OptimizerReductionType::AccumReduce,
      DataType::FLOAT,
      DataType::FLOAT,
      DataType::FLOAT,
      false,
      Op::Settings(graph, "adamcombo"));

  pattern.gradUnscale(graph,
                      adam,
                      adam->initGs,
                      adam->inId(AdamComboOp::getGsInIndex()),
                      adam->inId(AdamComboOp::getUpdaterInIndex()),
                      false);

  Op *castOp = nullptr;
  for (auto op : ir.getAllOps()) {
    if (op->isConvertibleTo<CastOp>()) {
      castOp = op;
      break;
    }
  }
  BOOST_REQUIRE(castOp != nullptr);
  auto *castedGs = castOp->outTensor(CastOp::getOutIndex());
  BOOST_REQUIRE(!castedGs->isOptimizerTensor());
}
