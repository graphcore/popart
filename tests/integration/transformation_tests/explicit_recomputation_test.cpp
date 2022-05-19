// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ExplicitRecomputationTest

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <filereader.hpp>
#include <memory>
#include <string>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/sgd.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/testdevice.hpp>

#include "popart/builder.gen.hpp"
#include "popart/graphutils.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/batchnorm.hpp"
#include "popart/op/dynamic/dynamicslice.hpp"
#include "popart/op/dynamic/dynamicupdate.hpp"
#include "popart/op/identity.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/scheduler_requireoptimal.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/vendored/any.hpp"
#include "popart/vertex.hpp"
#include "popart/voiddata.hpp"

using namespace popart;

// Detailed test description:
//
// Compute the forward pass which is made of MatMuls and Adds
// Normal Ops or Ops marked to not be recomputed do not perform recompute
// In the backward pass, we have MatMul and Add that are in fact from the
// forward pass. They have therefore marked as recomputed
// Finally, we have gradient computations and var updates as normal

TensorId gemm(Builder *builder, TensorId ip, TensorId w0, TensorId b0) {
  return builder->customOp(Onnx::AiOnnx::OpSet10::Gemm,
                           10,
                           {ip, w0, b0},
                           1,
                           {{"__recompute_output_in_backward_pass",
                             static_cast<int>(RecomputeType::Recompute)}},
                           "CustomGEMM")[0];
}

BOOST_AUTO_TEST_CASE(ExplicitRecomputation_Case) {

  // Build the following three layered NN:
  // (in0) -> [GEMM0] -> [GEMM1] -> [GEMM2] -> (out)
  // turn on recomputation and check if the IR contains recomputation
  // The purpose of any recomputation is to trade computation for memory

  auto test = [](bool recomputation) {
    auto builder = Builder::create();
    auto aiOnnx  = builder->aiOnnxOpset10();

    TensorInfo wshape{"FLOAT", std::vector<int64_t>{10, 10}};
    TensorInfo bshape{"FLOAT", std::vector<int64_t>{10, 1}};

    std::vector<float> ones(100, 1.0f);
    ConstVoidData ones_data = {ones.data(), wshape};

    std::vector<float> zeros(10, 0.0f);
    ConstVoidData zeros_data = {zeros.data(), bshape};

    auto ip0 = builder->addInputTensor(wshape);

    auto w0 = builder->addInitializedInputTensor(ones_data);
    auto b0 = builder->addInitializedInputTensor(zeros_data);

    auto m1 = gemm(builder.get(), ip0, w0, b0);
    auto m2 = gemm(builder.get(), m1, w0, b0);
    auto m3 = gemm(builder.get(), m2, w0, b0);

    auto out      = m3;
    auto l1       = builder->aiGraphcoreOpset1().l1loss({out}, 0.1);
    auto nMatMuls = 3;

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    auto dataFlow = DataFlow(1, {out}, AnchorReturnType("All"));

    auto optimizer = ConstSGD(0.01);

    auto device = createTestDevice(TEST_TARGET);
    SessionOptions opts;
    opts.enableOutlining = false;

    if (recomputation) {
      opts.explicitRecomputation = true;
    } else {
      opts.explicitRecomputation = false;
    }

    Ir ir;
    ir.prepare({modelProto,
                InputShapeInfo(),
                dataFlow,
                l1,
                &optimizer,
                *device,
                opts,
                Patterns(PatternsLevel::All)});

    auto nRecompute = 0;
    for (auto op : ir.getOpSchedule({}, RequireOptimalSchedule::Yes)) {
      if (recomputation) {
        // All Ops except those that have their type set to Recomputed
        // must have their recompute type set to Checkpoint
        if (op->settings.recomputeType != RecomputeType::Recomputed) {
          BOOST_CHECK(op->settings.recomputeType == RecomputeType::Checkpoint);
        }

        // Count the number of MatMuls with a recompute flag
        if ((op->settings.recomputeType == RecomputeType::Recomputed) &&
            (!op->opid.type.compare("MatMul"))) {
          nRecompute++;
        }
      } else {
        // All Forward pass Ops must have recomputeType set to Recompute
        // if explicit recomputation is turned off and path is to the loss
        if (op->toLoss == PathToLoss::Yes) {
          BOOST_CHECK(op->settings.recomputeType == RecomputeType::Recompute);
        }

        // All Backwards pass Ops must have recomputeType set to Checkpoint
        // if explicit recomputation is turned off and path is from Loss
        if (op->fromLoss == PathFromLoss::Yes) {
          BOOST_CHECK(op->settings.recomputeType == RecomputeType::Checkpoint);
        }
      }
    }
    // Verify that the number of recomputation equals to the num ops
    if (recomputation) {
      BOOST_CHECK(nRecompute == nMatMuls);
    }
  };

  test(true);
  test(false);
}

BOOST_AUTO_TEST_CASE(ExplicitRecomputation_Case1) {
  // Build the graph:
  //
  // in0 -- Add -- Relu -- L1Loss -- out
  //       /     (recomp)
  // in1 -
  //
  // and verify that the recomputed Relu tensor is scheduled as expected:
  // after the loss gradient has been computed
  auto builder = Builder::create();
  auto ip0     = builder->addInputTensor("FLOAT", std::vector<int64_t>{10});
  auto ip1     = builder->addInputTensor("FLOAT", std::vector<int64_t>{10, 10});

  auto rel = builder->aiOnnxOpset10().relu({ip0});
  auto out = builder->aiOnnxOpset10().add({rel, ip1});
  auto l1  = builder->aiGraphcoreOpset1().l1loss({out}, 0.1);
  auto id  = builder->aiGraphcoreOpset1().identityloss({l1});

  builder->recomputeOutputInBackwardPass(rel);
  auto modelProto = io::getModelFromString(builder->getModelProto());

  auto device    = createTestDevice(TEST_TARGET);
  auto optimizer = ConstSGD(0.01);
  SessionOptions opts;
  opts.explicitRecomputation      = true;
  std::vector<TensorId> anchorIds = {reservedGradientPrefix() + ip0, out};
  auto patterns                   = Patterns(PatternsLevel::All);
  // So that the identity loss is not pruned
  patterns.enableInPlace(false);
  patterns.enableOpToIdentity(false);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              DataFlow(1, anchorIds),
              id,
              &optimizer,
              *device,
              opts,
              patterns});

  auto opSchedule = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
  std::vector<size_t> recomputedIdx;
  std::vector<size_t> lossGradIdx;

  for (size_t idx = 0; idx < opSchedule.size(); idx++) {
    auto op = opSchedule.at(idx);

    // Register the position in the schedule of the recomputed Relu op
    if ((op->settings.recomputeType == RecomputeType::Recomputed) &&
        (!op->opid.type.compare("Relu"))) {
      recomputedIdx.push_back(idx);
    }

    // Register the position in the schedule of the loss gradient op
    if (!op->opid.type.compare("L1Grad")) {
      lossGradIdx.push_back(idx);
    }
  }
  // Verify that the recomputed Relu and the loss grad have been
  // found in the schedule
  BOOST_CHECK(recomputedIdx.size() == 1);
  BOOST_CHECK(lossGradIdx.size() == 1);

  // Verify that the recomputed Relu appears after the loss grad op
  // in the schedule
  BOOST_CHECK(recomputedIdx[0] > lossGradIdx[0]);
}

std::vector<TensorId>
batchnormalization(Builder *b, TensorId act, ConstVoidData bndata) {
  auto aiOnnx = b->aiOnnxOpset10();
  auto scale  = b->addInitializedInputTensor(bndata);
  auto bias   = b->addInitializedInputTensor(bndata);
  auto mean   = b->addInitializedInputTensor(bndata);
  auto var    = b->addInitializedInputTensor(bndata);
  // Batchnorm version 10 creates 5 outputs, from which the first one is the
  // data output
  auto bn_out = aiOnnx.batchnormalization({act, scale, bias, mean, var}, 5);
  return bn_out;
}

BOOST_AUTO_TEST_CASE(ExplicitRecomputation_BatchNormTest) {
  // Build the graph:
  //
  //        ps0    ps0          ps1       ps2       ps2        (pipeline stage)
  // in0 -- Add -- BatchNorm -- MatMul -- L1Loss -- IdentityLoss -- id
  //       /      (recomp)     /
  // in1 -               in2 -
  //
  // and verify that the recomputed BatchNorm tensor is scheduled as expected,
  // and that the (inplace modified) batchnorm statistics have been backed up

  auto builder = Builder::create();
  auto ip0 = builder->addInputTensor("FLOAT", std::vector<int64_t>{4, 10, 10});
  auto ip1 = builder->addInputTensor("FLOAT", std::vector<int64_t>{4, 10, 10});
  auto ip2 = builder->addInputTensor("FLOAT", std::vector<int64_t>{4, 10, 10});

  TensorInfo bn_shape{"FLOAT", std::vector<int64_t>{10}};
  float bn_vals[4]      = {0};
  ConstVoidData bn_data = {bn_vals, bn_shape};

  auto out = builder->aiOnnxOpset10().add({ip0, ip1});
  auto bn  = batchnormalization(builder.get(), out, bn_data);
  auto mul = builder->aiOnnxOpset10().matmul({bn.front(), ip2});
  auto l1  = builder->aiGraphcoreOpset1().l1loss({mul}, 0.1);
  auto id  = builder->aiGraphcoreOpset1().identityloss({l1});

  std::set<TensorId> bnTensorIds;
  bnTensorIds.insert(bn.begin(), bn.end());

  builder->pipelineStage(out, 0);
  builder->pipelineStage(bnTensorIds, 0);
  builder->pipelineStage(mul, 1);
  builder->pipelineStage(l1, 2);
  builder->pipelineStage(id, 2);

  builder->virtualGraph(out, 0);
  builder->virtualGraph(bnTensorIds, 0);
  builder->virtualGraph(mul, 1);
  builder->virtualGraph(l1, 2);
  builder->virtualGraph(id, 2);

  int numStages = 3;

  builder->recomputeOutputInBackwardPass(bnTensorIds);
  auto modelProto = io::getModelFromString(builder->getModelProto());

  auto device    = createTestDevice(TEST_TARGET, numStages);
  auto optimizer = ConstSGD(0.01);
  SessionOptions opts;
  opts.enableExplicitIR(true);
  opts.enablePipelining = true;
  opts.virtualGraphMode = VirtualGraphMode::Manual;
  opts.enableOutlining  = false;

  std::vector<TensorId> anchorIds = {reservedGradientPrefix() + ip0, out};
  auto patterns                   = Patterns(PatternsLevel::All);
  // So that the identity loss is not pruned
  patterns.enableInPlace(false);
  patterns.enableOpToIdentity(false);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              DataFlow(2 * numStages - 1, anchorIds),
              id,
              &optimizer,
              *device,
              opts,
              patterns});

  auto opSchedule = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);

  std::vector<size_t> batchNormIdx;
  std::vector<size_t> identityIdx;

  for (size_t idx = 0; idx < opSchedule.size(); idx++) {
    auto op0 = opSchedule.at(idx);

    logging::trace("Op: {} {}", op0->debugName(), op0->settings.recomputeType);

    // Check that the graph around the recomputed BatchNormOp is wired correctly
    if (op0->isConvertibleTo<BatchNormOp>()) {
      graphutils::OpPredMap preds;

      // One of the two batchnorms consumes stashed & restored var & mean values
      preds[0] = [](const Op *op1) {
        return op1->isConvertibleTo<DynamicSliceOp>();
      };
      preds[1] = [](const Op *op1) {
        return op1->isConvertibleTo<DynamicSliceOp>();
      };
      preds[2] = [&op0](const Op *op1) {
        return op0 == op1 && op1->isConvertibleTo<BatchNormOp>();
      };

      graphutils::Edges edges{{0, 2}, {1, 2}};

      auto matches = graphutils::findMatchingOps(op0->getGraph(), preds, edges);

      if (!matches.empty()) {
        batchNormIdx.push_back(idx);
      }
    }

    // Check that the graph around the IdentityOp used to backup the tensors
    // before they are inplace modified is wired correctly
    if (op0->isConvertibleTo<IdentityOp>()) {
      graphutils::OpPredMap preds;

      // Stashed & restored var & mean values
      preds[0] = [&op0](const Op *op1) {
        return op0 == op1 && op1->isConvertibleTo<IdentityOp>();
      };
      preds[1] = [](const Op *op1) {
        return op1->isConvertibleTo<DynamicUpdateInplaceOp>();
      };

      graphutils::Edges edges{{0, 1}};

      auto matches = graphutils::findMatchingOps(op0->getGraph(), preds, edges);

      if (!matches.empty()) {
        identityIdx.push_back(idx);
      }
    }
  }

  // Verify recomputed BatchNorm exists in the schedule
  BOOST_CHECK(batchNormIdx.size() == 1);

  // Verify IdentityOp backup of the batchnorm mean & var exist in the schedule
  BOOST_CHECK(identityIdx.size() == 2);
}
