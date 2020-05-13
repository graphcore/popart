// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PingPongGradSumDecTest

#include <../test_runner.hpp>
#include <boost/test/unit_test.hpp>
#include <string>
#include <popart/builder.hpp>
#include <popart/ir.hpp>
#include <popart/op/add.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/init.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/reshape.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

using namespace popart;

// Model: N parallel matmuls, sharing weights, separate pingpong phases
// _________________________________________
// phase 0:
// in0 -
//       \
//        Matmul0 -- ReLU ----.
//       /                    |
//  w0 -                      |
// ___________________________|_____________
// phase 2:                   |
// in1 -                      |
//       \                    |
//        Matmul1 -- ReLU --. |
//       /                  | |
//  w0 -                    | |
// _________________________|_|_____________
// phase 4:                 | |
// in2 -                    | |
//       \                  | |
//        Matmul2 -- ReLU - Sum -- out
//       /
//  w0 -
// _________________________________________

// Note that the number of ops from the loss to the original grad-sum is the
// same for all matmuls.
// We then verify that the grad of w0 is correctly added in the bwd phases:
// 1. Add to w0-grad in phase 4 (Matmul2)
// 2. Add to w0-grad in phase 6 (Matmul1)
// 3. Add to w0-grad in phase 8 (Matmul0)

BOOST_AUTO_TEST_CASE(TestDecomposeAcrossPingPongPhases) {
  TestRunner runner;
  runner.isTraining = true;
  int numLayers     = 3;
  int size          = 100;

  // Weights are [size, size], Input and Acts are [1, size]
  TensorInfo wInfo{"FLOAT", std::vector<int64_t>{size, size}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;
  std::vector<float> w0Data(wInfo.nelms(), 0);
  ConstVoidData w0CVData{w0Data.data(), wInfo};

  runner.buildModel([&](auto &builder) {
    auto aiOnnx = builder.aiOnnxOpset9();
    TensorInfo inInfo{"FLOAT", std::vector<int64_t>{1, size}};
    auto w0 = builder.addInitializedInputTensor(w0CVData);
    std::vector<std::string> outVec;
    for (int layer = 0; layer < numLayers; layer++) {
      auto input = builder.addInputTensor(inInfo);
      auto out = aiOnnx.matmul({input, w0}, "mm_layer" + std::to_string(layer));
      builder.pingPongPhase(out, layer * 2);
      out = aiOnnx.relu({out}, "relu_layer" + std::to_string(layer));
      builder.pingPongPhase(out, layer * 2);
      outVec.push_back(out);
    }
    auto sum = aiOnnx.sum(outVec);
    auto l1  = builder.aiGraphcoreOpset1().l1loss({sum}, 0.1);
    builder.pingPongPhase(l1, (numLayers - 1) * 2);

    // To make introspecting the IR easy
    runner.opts.enableOutlining = false;
    // Enable feature-under-test
    runner.opts.decomposeGradSum = true;
    // Use every second pingpong phase only (maps to one IPU)
    runner.opts.pingPongPhases   = numLayers * 2 - 1;
    runner.opts.virtualGraphMode = VirtualGraphMode::PingPong;
    runner.patterns              = Patterns(PatternsLevel::Default);
    runner.losses.push_back(
        new IdentityLoss(l1, "l1Loss", ReductionType::Mean));

    return sum;
  });

  runner.checkIr([&](Ir &ir) {
    std::vector<Op *> schedule = ir.getOpSchedule({});
    std::vector<Op *> gradPartialAddsOrder;
    size_t addOpNumber = 0;
    for (size_t i = 0; i < schedule.size(); i++) {
      Op *op = schedule.at(i);
      if (op->isConvertibleTo<AddLhsInplaceOp>() ||
          op->isConvertibleTo<AddRhsInplaceOp>()) {

        gradPartialAddsOrder.push_back(op);
        auto addLhs = AddOp::getArg0InIndex();
        auto addRhs = AddOp::getArg1InIndex();

        size_t mmLayer         = numLayers - 1 - addOpNumber;
        std::string mmLayerStr = "mm_layer" + std::to_string(mmLayer);

        PingPongPhase expectedPhase = numLayers + 1 + addOpNumber * 2;
        // Check that the add occurs in the expected phase (4, 6, 8)
        BOOST_CHECK(op->getPingPongPhase() == expectedPhase);

        // Check that the rhs input is produced in the same phase
        BOOST_CHECK(
            op->input->tensor(addRhs)->getProducer()->getPingPongPhase() ==
            expectedPhase);

        // Check that the add comes as soon as possible
        // after the weight partial tensor becomes live
        auto checkForMatMulRhsReshape = [&](Op *op) {
          BOOST_CHECK(op->getName().find("MatMulOp_RhsReshape") !=
                      std::string::npos);
          BOOST_CHECK(op->getName().find(mmLayerStr) != std::string::npos);
          BOOST_CHECK(op->isConvertibleTo<ReshapeInplaceOp>());
        };

        auto checkForMatMulRhsGrad = [&](Op *op) {
          BOOST_CHECK(op->getName().find("MatMulRhsGradOp") !=
                      std::string::npos);
          BOOST_CHECK(op->getName().find(mmLayerStr) != std::string::npos);
          BOOST_CHECK(op->isConvertibleTo<MatMulOp>());
        };

        if (addOpNumber == 0) {
          BOOST_CHECK(schedule.at(i - 1)->isConvertibleTo<InitOp>());
          checkForMatMulRhsReshape(schedule.at(i - 2));
          checkForMatMulRhsGrad(schedule.at(i - 3));
        } else {
          checkForMatMulRhsReshape(schedule.at(i - 1));
          checkForMatMulRhsGrad(schedule.at(i - 2));
        }
        addOpNumber++;
      }
    }

    BOOST_CHECK_EQUAL(gradPartialAddsOrder.size(), numLayers);
  });
}
