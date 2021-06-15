// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE RecomputeTestIrNormOnlyAnnotation0

#include <boost/test/unit_test.hpp>
#include <memory>
#include <vector>
#include <popart/testdevice.hpp>

#include <filereader.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensordata.hpp>

using namespace popart;

TensorId conv(Builder *b, TensorId act, ConstVoidData wdata) {
  auto aiOnnx  = b->aiOnnxOpset9();
  auto weights = b->addInitializedInputTensor(wdata);
  act = aiOnnx.conv({act, weights}, {1, 1}, 1, {}, {1, 1, 1, 1}, {1, 1});
  return act;
}

TensorId batchnormalization(Builder *b, TensorId act, ConstVoidData bndata) {
  auto aiOnnx = b->aiOnnxOpset9();
  auto scale  = b->addInitializedInputTensor(bndata);
  auto bias   = b->addInitializedInputTensor(bndata);
  auto mean   = b->addInitializedInputTensor(bndata);
  auto var    = b->addInitializedInputTensor(bndata);
  auto bn_out = aiOnnx.batchnormalization({act, scale, bias, mean, var}, 5);
  act         = bn_out.at(0);
  return act;
}

BOOST_AUTO_TEST_CASE(NormOnlyRecomputeTest) {

  // Test that norms are Recompute

  // The model:
  //
  // In -> Conv -> BN -> Relu -> Conv -> Relu -> Conv -> BN -> Out
  //
  auto test = [](bool explicitRecomputation) {
    // Build an onnx model
    auto builder = Builder::create();
    auto aiOnnx  = builder->aiOnnxOpset9();

    TensorInfo input_shape{"FLOAT", std::vector<int64_t>{1, 4, 32, 32}};

    TensorInfo weights_shape{"FLOAT", std::vector<int64_t>{4, 4, 3, 3}};
    float weight_vals[4 * 4 * 3 * 3] = {0};
    ConstVoidData weight_data        = {weight_vals, weights_shape};

    TensorInfo bn_shape{"FLOAT", std::vector<int64_t>{4}};
    float bn_vals[4]      = {0};
    ConstVoidData bn_data = {bn_vals, bn_shape};

    auto act = builder->addInputTensor(input_shape);

    act     = conv(builder.get(), act, weight_data);
    act     = batchnormalization(builder.get(), act, bn_data);
    act     = aiOnnx.relu({act});
    act     = conv(builder.get(), act, weight_data);
    act     = aiOnnx.relu({act});
    act     = conv(builder.get(), act, weight_data);
    act     = batchnormalization(builder.get(), act, bn_data);
    auto l1 = builder->aiGraphcoreOpset1().l1loss({act}, 0.1);

    int nBNs = 2;

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    // Add the last tensor, and the 3rd tensor as anchors
    auto dataFlow  = DataFlow(1, {{act, AnchorReturnType("All")}});
    auto optimizer = ConstSGD(0.01);
    auto device    = createTestDevice(TEST_TARGET);

    SessionOptions opts;
    if (explicitRecomputation) {
      opts.explicitRecomputation = true;
    } else {
      opts.explicitRecomputation = false;
    }
    opts.autoRecomputation = RecomputationType::NormOnly;
    opts.enableOutlining   = false;
    opts.mergeVarUpdate    = MergeVarUpdateType::None;

    Ir ir;
    ir.prepare({modelProto,
                InputShapeInfo(),
                dataFlow,
                l1,
                &optimizer,
                *device,
                opts,
                Patterns({PreAliasPatternType::OptoIdentity,
                          PreAliasPatternType::PostNRepl})
                    .enableRuntimeAsserts(false)});

    int nRecompute = 0;
    for (auto op : ir.getOpSchedule({}, RequireOptimalSchedule::Yes)) {
      if (explicitRecomputation) {
        if ((op->settings.recomputeType == RecomputeType::Recomputed) &&
            (!op->opid.type.compare("BatchNormalization"))) {
          nRecompute++;
        } else {
          BOOST_CHECK(op->settings.recomputeType == RecomputeType::Checkpoint);
        }
      } else {
        // When explicit recomputation is switched OFF, only the BNs should
        // have their flags set to Recompute
        if (op->isNorm()) {
          BOOST_CHECK(op->settings.recomputeType == RecomputeType::Recompute);
        } else {
          BOOST_CHECK(op->settings.recomputeType == RecomputeType::Checkpoint);
        }
      }
    }

    // Verify that the number of recomputation equals to ops
    if (explicitRecomputation) {
      BOOST_CHECK(nRecompute == nBNs);
    }
  };

  test(false);
  test(true);
}
