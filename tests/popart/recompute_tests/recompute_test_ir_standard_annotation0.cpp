// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE RecomputeTestIrStandardAnnotation0

#include <boost/test/unit_test.hpp>
#include <memory>
#include <vector>
#include <popart/testdevice.hpp>

#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
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

BOOST_AUTO_TEST_CASE(StandardRecomputeTest) {
  auto run_test = [](bool recomputation, bool enableOutlining) {
    // Build an onnnx model
    auto builder = Builder::create();
    auto aiOnnx  = builder->aiOnnxOpset9();

    TensorInfo input_shape{"FLOAT", std::vector<int64_t>{1, 4, 32, 32}};

    TensorInfo weights_shape{"FLOAT", std::vector<int64_t>{4, 4, 3, 3}};
    float weight_vals[4 * 4 * 3 * 3] = {0};
    ConstVoidData weight_data        = {weight_vals, weights_shape};

    auto act = builder->addInputTensor(input_shape);

    for (int i = 0; i < 16; ++i) {
      act = conv(builder.get(), act, weight_data);
      act = aiOnnx.relu({act});
    }

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    // Add the last tensor as an anchor
    auto dataFlow  = DataFlow(1, {{act, AnchorReturnType("ALL")}});
    auto optimizer = ConstSGD(0.01);
    std::vector<std::shared_ptr<Loss>> losses{
        std::make_shared<L1Loss>(act, "l1LossVal", 0.1, ReductionType::SUM)};
    auto device = createTestDevice(TEST_TARGET);

    SessionOptions opts;
    opts.autoRecomputation = RecomputationType::Standard;

    if (recomputation && enableOutlining) {
      opts.explicitRecomputation = true;
      opts.enableOutlining       = true;
    } else if (!recomputation && enableOutlining) {
      opts.explicitRecomputation = false;
      opts.enableOutlining       = true;
    } else if (recomputation && !enableOutlining) {
      opts.explicitRecomputation = true;
      opts.enableOutlining       = false;
    } else {
      opts.explicitRecomputation = false;
      opts.enableOutlining       = false;
    }

    opts.mergeVarUpdate = MergeVarUpdateType::None;

    Ir ir;
    ir.prepare({modelProto,
                InputShapeInfo(),
                dataFlow,
                losses,
                &optimizer,
                *device,
                opts,
                Patterns({PreAliasPatternType::OPTOIDENTITY,
                          PreAliasPatternType::POSTNREPL})});

    // RECOMPUTE and RECOMPUTED counters
    int nRecompute  = 0;
    int nRecomputed = 0;

    for (auto op : ir.getOpSchedule({})) {

      if (recomputation && enableOutlining) {
        // Enabling outlining means the CallOp replaces the other Ops
        // No Op cannot be RECOMPUTE
        BOOST_CHECK(op->settings.recomputeType != RecomputeType::RECOMPUTE);

        // All other Ops except Relu are replaced by CallOp
        // Relu and Call should have at least type set to RECOMPUTED
        if (op->settings.recomputeType == RecomputeType::RECOMPUTED) {
          if ((op->opid.type == "Relu") || (op->opid.type == "Call")) {
            nRecomputed++;
          }
        }

        // All Grads should have recomputeType set to CHECKPOINT
        if (op->fromLoss == PathFromLoss::Yes) {
          BOOST_CHECK(op->settings.recomputeType == RecomputeType::CHECKPOINT);
        }
      }

      if (!recomputation && enableOutlining) {
        // Recomputation has been disabled. RECOMPUTED must nost exist
        BOOST_CHECK(op->settings.recomputeType != RecomputeType::RECOMPUTED);

        if (op->settings.recomputeType == RecomputeType::RECOMPUTE) {
          if (op->opid.type == "Call") {
            nRecompute++;
          }
        }

        // All Grads should have recompute type set to CHECKPOINT
        if (op->fromLoss == PathFromLoss::Yes) {
          BOOST_CHECK(op->settings.recomputeType == RecomputeType::CHECKPOINT);
        }
      }

      if (recomputation && !enableOutlining) {
        // We can accept CHECKPOINT or UNDEFINED or non RECOMPUTE types
        BOOST_CHECK(op->settings.recomputeType == RecomputeType::CHECKPOINT ||
                    op->settings.recomputeType == RecomputeType::UNDEFINED ||
                    op->settings.recomputeType != RecomputeType::RECOMPUTE);

        // Outlining is disabled, CallOp must not exists
        BOOST_CHECK(op->opid.type != "Call");

        // Same as when both recomputation and outlining is enabled
        // Except Conv replaces CallOp
        if (op->settings.recomputeType == RecomputeType::RECOMPUTED) {
          if ((op->opid.type == "Relu") || (op->opid.type == "Call")) {
            nRecomputed++;
          }
        }

        // All Grads must have recompute type set to CHECKPOINT
        if (op->fromLoss == PathFromLoss::Yes) {
          BOOST_CHECK(op->settings.recomputeType == RecomputeType::CHECKPOINT);
        }
      }

      if (!recomputation && !enableOutlining) {

        // Recomputation has been disabled. RECOMPUTED must nost exist
        BOOST_CHECK(op->settings.recomputeType != RecomputeType::RECOMPUTED);

        // Outlining is disabled, CallOp must not exists
        BOOST_CHECK(op->opid.type != "Call");

        if (op->settings.recomputeType == RecomputeType::RECOMPUTE) {
          ++nRecompute;
        }

        // All Grads must have recompute type set to CHECKPOINT
        if (op->fromLoss == PathFromLoss::Yes) {
          BOOST_CHECK(op->settings.recomputeType == RecomputeType::CHECKPOINT);
        }
      }
    }
    std::cout << "recomputation = " << recomputation
              << ", outlining = " << enableOutlining
              << ", nRecompute = " << nRecompute
              << ", nRecomputed = " << nRecomputed << std::endl;

    if (recomputation && enableOutlining) {
      BOOST_CHECK(nRecompute == 0);
      BOOST_CHECK(nRecomputed > 0);
    }
    if (!recomputation && enableOutlining) {
      BOOST_CHECK(nRecompute > 0);
      BOOST_CHECK(nRecomputed == 0);
    }
    if (recomputation && !enableOutlining) {
      BOOST_CHECK(nRecompute == 0);
      BOOST_CHECK(nRecomputed > 0);
    }
    if (!recomputation && !enableOutlining) {
      BOOST_CHECK(nRecompute > 0);
      BOOST_CHECK(nRecomputed == 0);
    }
  };

  run_test(true, true);
  run_test(false, true);
  run_test(true, false);
  run_test(false, false);
}
