// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE TestEnsureFp32LossScaleTransform
#include <boost/test/unit_test.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/cast.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/nll.hpp>
#include <popart/op/reshape.hpp>
#include <popart/op/scale.hpp>
#include <popart/opidentifier.hpp>
#include <popart/sgd.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensors.hpp>
#include <popart/transforms/ensurefp32lossscale.hpp>

using namespace popart;

using PassThroughOps            = std::vector<Op *>;
using TerminalOps               = std::vector<Op *>;
using FromLossScaleTraversalOps = std::pair<PassThroughOps, TerminalOps>;

BOOST_AUTO_TEST_CASE(TestEnsureFp32LossScaleTransformSessionOption) {
  auto opts = SessionOptions();
  BOOST_CHECK(opts.ensureFp32LossScaleTensor == false);
}

/*
  lossScale -- NllGradOp -- gradOut
  label --------'  |
  probs -----------
 */
auto nllGradOp = [](Graph &g,
                    const TensorId probs,
                    const TensorId label,
                    const TensorId lossScale) {
  auto nllGradOp =
      g.createConnectedOp<NllGradOp>({{NllGradOp::getProbsInIndex(), probs},
                                      {NllGradOp::getLabelInIndex(), label},
                                      {NllGradOp::getGradInIndex(), lossScale}},
                                     {{NllGradOp::getOutIndex(), "gradOut"}},
                                     "tmp_loss",
                                     nonstd::optional<int>(),
                                     ReductionType::Mean,
                                     false,
                                     Op::Settings(g, ""));

  FromLossScaleTraversalOps ops{{}, {nllGradOp}};
  return ops;
};

/*
  lossScale - Reshape -- reshape0 -- Reshape -- reshape1 -- NllGradOp -- gradOut
                                                label --------'  |
                                                probs -----------
 */
auto reshapeChain = [](Graph &g,
                       const TensorId probs,
                       const TensorId label,
                       const TensorId lossScale) {
  TensorId reshape0 = "reshape0";
  auto reshapeOp0 =
      g.createConnectedOp<ReshapeOp>({{ReshapeOp::getInIndex(), lossScale}},
                                     {{ReshapeOp::getOutIndex(), reshape0}},
                                     Onnx::Operators::Reshape_5,
                                     Shape{1, 1},
                                     Op::Settings(g, ""));

  TensorId reshape1 = "reshape1";
  auto reshapeOp1 =
      g.createConnectedOp<ReshapeOp>({{ReshapeOp::getInIndex(), reshape0}},
                                     {{ReshapeOp::getOutIndex(), reshape1}},
                                     Onnx::Operators::Reshape_5,
                                     Shape{},
                                     Op::Settings(g, ""));

  auto nllGradOp =
      g.createConnectedOp<NllGradOp>({{NllGradOp::getProbsInIndex(), probs},
                                      {NllGradOp::getLabelInIndex(), label},
                                      {NllGradOp::getGradInIndex(), reshape1}},
                                     {{NllGradOp::getOutIndex(), "gradOut"}},
                                     "tmp_loss",
                                     nonstd::optional<int>(),
                                     ReductionType::Mean,
                                     false,
                                     Op::Settings(g, ""));

  FromLossScaleTraversalOps ops{{reshapeOp0, reshapeOp1}, {nllGradOp}};
  return ops;
};

/*
       .----- ScaleOp -- scaleOut -- NllGradOp -- gradOut1
      |
  lossScale ------------- NllGradOp -- gradOut0
              label --------'  |
              probs -----------
 */
auto nllGradOps = [](Graph &g,
                     const TensorId probs,
                     const TensorId label,
                     const TensorId lossScale) {
  auto nllGradOp0 =
      g.createConnectedOp<NllGradOp>({{NllGradOp::getProbsInIndex(), probs},
                                      {NllGradOp::getLabelInIndex(), label},
                                      {NllGradOp::getGradInIndex(), lossScale}},
                                     {{NllGradOp::getOutIndex(), "gradOut0"}},
                                     "tmp_loss",
                                     nonstd::optional<int>(),
                                     ReductionType::Mean,
                                     false,
                                     Op::Settings(g, ""));

  TensorId scaleOut = "scaleOut";
  auto scaleOp =
      g.createConnectedOp<ScaleOp>({{ScaleOp::getInIndex(), lossScale}},
                                   {{ScaleOp::getOutIndex(), scaleOut}},
                                   Onnx::AiGraphcore::OpSet1::Scale,
                                   0.3,
                                   Op::Settings(g, ""));

  auto nllGradOp1 =
      g.createConnectedOp<NllGradOp>({{NllGradOp::getProbsInIndex(), probs},
                                      {NllGradOp::getLabelInIndex(), label},
                                      {NllGradOp::getGradInIndex(), scaleOut}},
                                     {{NllGradOp::getOutIndex(), "gradOut1"}},
                                     "tmp_loss",
                                     nonstd::optional<int>(),
                                     ReductionType::Sum,
                                     false,
                                     Op::Settings(g, ""));

  FromLossScaleTraversalOps ops{{scaleOp}, {nllGradOp0, nllGradOp1}};
  return ops;
};

/*
  lossScale -- L1GradOp -- gradOut
  probs ---------'

 */
auto l1GradOp = [](Graph &g,
                   const TensorId probs,
                   const TensorId label,
                   const TensorId lossScale) {
  auto l1GradOp =
      g.createConnectedOp<L1GradOp>({{L1GradOp::getFwdActInIndex(), probs},
                                     {L1GradOp::getGradInIndex(), lossScale}},
                                    {{L1GradOp::getOutIndex(), "gradOut"}},
                                    0.1,
                                    ReductionType::Mean,
                                    Op::Settings(g, ""));

  FromLossScaleTraversalOps ops{{}, {l1GradOp}};
  return ops;
};

/*
  lossScale(fp16) = stream
  probs(fp16), label(fp16) = const

  Construct the beginning of a backwards pass from these inputs, e.g.

  gradOut = nllgrad(lossScale, probs, label)

  Apply the transform to the graph, and check:
    - lossScale has become fp32.
    - pass-through ops have fp32 output
    - terminal ops have fp16 output
 */
void testReplacesFp16LossScaleWithFp32(
    std::function<FromLossScaleTraversalOps(Graph &g,
                                            const TensorId probs,
                                            const TensorId label,
                                            const TensorId lossScale)>
        constructGraph) {
  Ir ir;
  Graph &g = ir.getMainGraph();

  auto optimizer = SGD({{"lossScaling", {2.0, false}}});
  ir.setOptimizer(optimizer);

  const TensorId lossScale =
      optimizer.getLossScalingTensorId(DataType::FLOAT16);
  BOOST_CHECK(g.getTensors().contains(lossScale));

  const TensorId probs = "probs";
  const TensorId label = "label";

  const TensorInfo ls_ti{DataType::FLOAT16, Shape{}};
  const std::vector<float> lsData(1);

  const TensorInfo pr_ti{DataType::FLOAT16, Shape{2, 10}};
  const std::vector<float> prData(20);

  const TensorInfo lb_ti{DataType::FLOAT16, Shape{2}};
  const std::vector<float> lbData(2);

  g.getTensors().addConstInit(probs, pr_ti, prData.data());
  g.getTensors().addConstInit(label, lb_ti, lbData.data());

  // Construct the graph
  auto ops            = constructGraph(g, probs, label, lossScale);
  auto passThroughOps = ops.first;
  auto terminalOps    = ops.second;

  ir.removeIsolatedTensors(true); // remove fp32 loss scale tensor from graph

  // Transform graph
  ir.applyTransform(EnsureFp32LossScale::id(), ir.getMainGraph());

  // Run checks:
  // The loss scale tensor is fp32
  BOOST_CHECK(g.getTensors().get(lossScale)->info.dataType() ==
              DataType::FLOAT);

  // All pass-through op outputs are fp32
  for (Op *op : passThroughOps) {
    for (Tensor *output : op->output->tensors()) {
      BOOST_CHECK(output->info.dataType() == DataType::FLOAT);
    }
  }

  // All loss grad outputs are fp16
  for (Op *op : terminalOps) {
    for (Tensor *output : op->output->tensors()) {
      BOOST_CHECK(output->info.dataType() == DataType::FLOAT16);
    }
  }
}

BOOST_AUTO_TEST_CASE(TestReplacesFp16LossScaleWithFp32) {
  testReplacesFp16LossScaleWithFp32(nllGradOp);
  testReplacesFp16LossScaleWithFp32(reshapeChain);
  testReplacesFp16LossScaleWithFp32(nllGradOps);
  testReplacesFp16LossScaleWithFp32(l1GradOp);
}

BOOST_AUTO_TEST_CASE(TestDoNotReplaceCastedFp16LossScaleWithFp32) {
  Ir ir;
  Graph &g = ir.getMainGraph();

  auto optimizer = SGD({{"lossScaling", {2.0, false}}});
  ir.setOptimizer(optimizer);

  const TensorId lossScale =
      optimizer.getLossScalingTensorId(DataType::FLOAT16);
  BOOST_CHECK(g.getTensors().contains(lossScale));

  // Construct the graph:
  //
  // lossScale_fp16 -- CastOp -- lossScale_fp32
  g.createConnectedOp<CastOp>({{ScaleOp::getInIndex(), lossScale}},
                              {{ScaleOp::getOutIndex(), "cast"}},
                              Onnx::Operators::Cast_6,
                              DataType::FLOAT,
                              Op::Settings(g, ""));

  ir.removeIsolatedTensors(true); // remove fp32 loss scale tensor from graph

  // Graph transformation fails on this graph. Graph traversal from the loss
  // scale tensor terminates at the CastOp. A CastOp consumer of the loss scale
  // tensor prevets us from being able to convert the tensor to fp32.
  BOOST_CHECK_THROW(
      ir.applyTransform(EnsureFp32LossScale::id(), ir.getMainGraph()), error);

  // Check loss scale tensor is still fp16
  BOOST_CHECK(g.getTensors().get(lossScale)->info.dataType() ==
              DataType::FLOAT16);
}
