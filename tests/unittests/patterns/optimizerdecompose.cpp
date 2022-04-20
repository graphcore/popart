// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE TestPatternsOptimizerDecompose
#include <boost/test/unit_test.hpp>

// To access gradUnscale
#ifdef __clang__
#pragma clang diagnostic ignored "-Wkeyword-macro"
#endif
#define protected public
#include <popart/patterns/adamdecompose.hpp>

#undef protected

#include <limits>
#include <map>
#include <string>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/op/adamcombo.hpp>
#include <popart/op/mul.hpp>
#include <popart/optimizervalue.hpp>

#include "popart/adam.hpp"
#include "popart/datatype.hpp"
#include "popart/half.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/optimizer.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensornames.hpp"
#include "popart/tensors.hpp"

using namespace popart;

BOOST_AUTO_TEST_CASE(TestGradUnscaleTypeMismatch) {
  // In the case where any of the OptimizerStateTensors in Adam are not fp32 and
  // the weight is fp16, the gradient to be unscaled will be fp16. All
  // OptimizerTensors are stored as fp32. The pattern's gradUnscale function
  // multiplies the fp32 optimizerTensor with the fp16 grad. This test checks
  // that the MulOp with mixed precision inputs and fp16 output is added in the
  // Ir. It also checks the output of the mulyiplication is not an
  // OptimizerTensor to avoid any interaction with optimizerFromHost.
  Ir ir;
  Graph &graph = ir.getMainGraph();

  AdamDecompose pattern;

  const TensorInfo wInfo{DataType::FLOAT16, Shape{4}};
  const TensorInfo gsInfo{DataType::FLOAT, Shape{1}};

  TensorId wId = "w";
  std::vector<float16_t> w_h(wInfo.nelms(), static_cast<float16_t>(0));
  graph.getTensors().addVarInit(wId, wInfo, w_h.data());

  TensorId gradId = "grad";
  std::vector<float16_t> g_h(wInfo.nelms(), static_cast<float16_t>(0));
  graph.getTensors().addVarInit(gradId, wInfo, g_h.data());

  TensorId gsId = reservedDefaultAdamGradientScalingPrefix();
  std::vector<float16_t> gs_h(wInfo.nelms(), static_cast<float>(0));
  graph.getTensors().addVarInit(gsId, gsInfo, gs_h.data());

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

  // A single instance of MulOp in the Ir:
  std::vector<Op *> mulOps = ir.opsOfType(Onnx::AiOnnx::OpSet7::Mul);
  BOOST_REQUIRE(mulOps.size() == 1);
  auto mulOp = mulOps.back();

  auto *castedGs = mulOp->outTensor(MulOp::getOutIndex());
  BOOST_REQUIRE(castedGs->info.dataType() == DataType::FLOAT16);
  BOOST_REQUIRE(!castedGs->isOptimizerTensor());
}

BOOST_AUTO_TEST_CASE(
    TestGradAccumSchedulesAccumulateEarlyWhenDelayingVarUpdates) {
  /*
    If withEarlyGradientAccumulations = true && explicitIr = false:
      - Turn on scheduleNonWeightUpdateGradientConsumersEarly option.
      - Expect AccumulateOp schedule priority of max.
    If withEarlyGradientAccumulations = true && explicitIr = true:
      - Turn on scheduleNonWeightUpdateGradientConsumersEarly option.
      - Turn on explicit main loop, host io, and recomputation.
      - Expect AccumulateOp schedule priority equal to combo op's. Should be
        0, but that is tested in the VarUpdateOp unit tests.
    If withEarlyGradientAccumulations = false:
      - Turn off scheduleNonWeightUpdateGradientConsumersEarly options.
      - Expect AccumulateOp schedule priority equal to combo op's. Should be
        -inf, but that is tested in the VarUpdateOp unit tests.
  */
  const auto test = [](const bool withEarlyGradientAccumulations,
                       const bool explicitIr) {
    Ir ir;
    Graph &graph = ir.getMainGraph();

    auto &opts                      = ir.getSessionOptions();
    opts.enableGradientAccumulation = true;
    opts.accumulationFactor         = 2;
    opts.delayVarUpdates            = true;
    opts.scheduleNonWeightUpdateGradientConsumersEarly =
        withEarlyGradientAccumulations;
    opts.enableExplicitMainLoops = explicitIr;
    opts.useHostCopyOps          = explicitIr;

    // Make any concrete subclass of OptimizerDecompose, we only want to test
    // the helper function OptimizerDecompose::gradAccum.
    AdamDecompose pat;

    const TensorInfo wInfo{DataType::FLOAT, Shape{4}};

    TensorId wId = "w";
    std::vector<float16_t> w_h(wInfo.nelms(), static_cast<float>(0));
    graph.getTensors().addVarInit(wId, wInfo, w_h.data());

    TensorId gradId = "grad";
    std::vector<float16_t> g_h(wInfo.nelms(), static_cast<float>(0));
    graph.getTensors().addVarInit(gradId, wInfo, g_h.data());

    const auto adam = graph.createConnectedOp<AdamComboOp>(
        {{AdamComboOp::getVarToUpdateInIndex(), wId},
         {AdamComboOp::getUpdaterInIndex(), gradId}},
        {{AdamComboOp::getUpdatedVarOutIndex(),
          ir.createIntermediateTensorId(wId)}},
        OptimizerValue{},
        OptimizerValue{},
        OptimizerValue{},
        OptimizerValue{},
        OptimizerValue{},
        OptimizerValue{},
        OptimizerValue{},
        OptimizerValue{0.0f, true},
        AdamMode::Adam,
        WeightDecayMode::Decay,
        false,
        OptimizerReductionType::None,
        DataType::FLOAT,
        DataType::FLOAT,
        DataType::FLOAT,
        false,
        Op::Settings(graph, "adamcombo"));

    TensorId accumId = "accum";
    std::vector<float16_t> accum_h(wInfo.nelms(), static_cast<float>(0));
    graph.getTensors().addVarInit(accumId, wInfo, accum_h.data());

    // Test gradAccum helper results in an AccumulateOp with 0 schedule priority

    pat.gradAccum(graph,
                  adam,
                  wId,
                  accumId,
                  gradId,
                  false,
                  ir.createIntermediateTensorId(accumId));

    Op *accumOp = nullptr;
    for (auto op : ir.getAllOps()) {
      if (op->isConvertibleTo<AccumulateOp>()) {
        accumOp = op;
        break;
      }
    }

    BOOST_REQUIRE(accumOp);

    double expectedSchedulePriority =
        withEarlyGradientAccumulations && !explicitIr
            ? std::numeric_limits<double>::max()
            : adam->settings.schedulePriority;

    const auto tol = boost::test_tools::tolerance(1e-10);
    BOOST_TEST(accumOp->settings.schedulePriority == expectedSchedulePriority,
               tol);
  };

  test(true, true);
  test(true, false);
  test(false, true);
  test(false, false);
}
