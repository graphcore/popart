// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE RecomputeTestIrNoneAnnotation0

#include <memory>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensordata.hpp>
#include <popart/testdevice.hpp>

using namespace popart;

TensorId conv(Builder *b, TensorId act, ConstVoidData wdata) {
  auto aiOnnx  = b->aiOnnxOpset9();
  auto weights = b->addInitializedInputTensor(wdata);
  act = aiOnnx.conv({act, weights}, {1, 1}, 1, {}, {1, 1, 1, 1}, {1, 1});
  return act;
}

BOOST_AUTO_TEST_CASE(NoRecomputeTest) {
  auto getOpSchedule = [](bool enableOutlining) {
    // Build an onnnx model
    auto builder = Builder::create();
    auto aiOnnx  = builder->aiOnnxOpset9();

    TensorInfo input_shape{"FLOAT", std::vector<int64_t>{1, 4, 32, 32}};

    TensorInfo weights_shape{"FLOAT", std::vector<int64_t>{4, 4, 3, 3}};
    float weight_vals[4 * 4 * 3 * 3] = {0};
    ConstVoidData weight_data        = {weight_vals, weights_shape};

    auto act = builder->addInputTensor(input_shape);

    for (int i = 0; i < 7; ++i) {
      act = conv(builder.get(), act, weight_data);
      act = aiOnnx.relu({act});
    }
    auto l1 = builder->aiGraphcoreOpset1().l1loss({act}, 0.1);

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    // Add the last tensor, and the 3rd tensor as anchors
    auto dataFlow  = DataFlow(1, {{act, AnchorReturnType("All")}});
    auto optimizer = ConstSGD(0.01);
    auto device    = createTestDevice(TEST_TARGET);

    SessionOptions opts;
    opts.autoRecomputation              = RecomputationType::None;
    opts.explicitRecomputation          = false;
    opts.enableOutlining                = enableOutlining;
    opts.enableOutliningCopyCostPruning = false;
    opts.mergeVarUpdate                 = MergeVarUpdateType::None;

    Ir ir;
    ir.prepare({modelProto,
                InputShapeInfo(),
                dataFlow,
                l1,
                &optimizer,
                *device,
                opts,
                Patterns({PreAliasPatternType::OptoIdentity,
                          PreAliasPatternType::PostNRepl})});

    auto opSchedule = ir.getOpSchedule({});

    // Check that there is no recomputation
    BOOST_CHECK(
        std::all_of(opSchedule.cbegin(), opSchedule.cend(), [](const Op *op) {
          return op->settings.recomputeType != RecomputeType::Recompute;
        }));
    return opSchedule.size();
  };
  auto schedNoOutliningSize   = getOpSchedule(false);
  auto schedWithOutliningSize = getOpSchedule(true);

  // Check that outlining grouped multiple ops at least once:
  BOOST_CHECK(schedWithOutliningSize < schedNoOutliningSize);
}
