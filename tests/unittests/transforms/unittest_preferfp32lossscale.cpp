// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE TestPatternsUpdateInplacePrioritiesForIpu
#include <boost/test/unit_test.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/nll.hpp>
#include <popart/opidentifier.hpp>
#include <popart/sgd.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensors.hpp>
#include <popart/transforms/preferfp32lossscale.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(TestReplacesConstFp16LossScaleWithFp32) {
  Ir ir;
  Graph &g = ir.getMainGraph();

  auto optimizer = SGD({{"lossScaling", {2.0, false}}});
  ir.setOptimizer(optimizer);

  const TensorId lossScale =
      optimizer.getLossScalingTensorId(DataType::FLOAT16);
  BOOST_CHECK(g.getTensors().contains(lossScale));

  /*
    lossScale(fp16), probs(fp16), label(fp16) = const
    gradOut = nllgrad(lossScale, probs, label)

    Apply pattern to nllgrad, check lossScale has become fp32.
   */

  const TensorId probs   = "probs";
  const TensorId label   = "label";
  const TensorId gradOut = "gradOut";

  const TensorInfo ls_ti{DataType::FLOAT16, Shape{}};
  const std::vector<float> lsData(1);

  const TensorInfo pr_ti{DataType::FLOAT16, Shape{2, 10}};
  const std::vector<float> prData(20);

  const TensorInfo lb_ti{DataType::FLOAT16, Shape{2}};
  const std::vector<float> lbData(2);

  g.getTensors().addConstInit(probs, pr_ti, prData.data());
  g.getTensors().addConstInit(label, lb_ti, lbData.data());

  auto nllgrad =
      g.createConnectedOp<NllGradOp>({{NllGradOp::getProbsInIndex(), probs},
                                      {NllGradOp::getLabelInIndex(), label},
                                      {NllGradOp::getGradInIndex(), lossScale}},
                                     {{NllGradOp::getOutIndex(), gradOut}},
                                     "tmp_loss",
                                     nonstd::optional<int>(),
                                     ReductionType::Mean,
                                     false,
                                     Op::Settings(g, ""));

  ir.removeIsolatedTensors(true); // remove fp32 loss scale tensor from graph

  ir.applyTransform(PreferFp32LossScale::id(), ir.getMainGraph());

  BOOST_CHECK(g.getTensors().get(lossScale)->info.dataType() ==
              DataType::FLOAT);
}
