#define BOOST_TEST_MODULE ExplicitRecomputationTest

#include <memory>
#include <vector>

#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/onnxutil.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/nll.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

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

    std::vector<float> ones(1.0f, 1.0f);
    ConstVoidData ones_data = {ones.data(), wshape};

    std::vector<float> zeros(0.0f, 1);
    ConstVoidData zeros_data = {ones.data(), bshape};

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
        // if explicit recomputation is turned off and path is not from loss
        if (op->fromLoss == PathFromLoss::No) {
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

  // test(true);
  // test(false);
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
