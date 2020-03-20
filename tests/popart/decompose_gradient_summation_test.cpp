// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE DecomposeGradientSummationTest

#include <boost/test/unit_test.hpp>
#include <string>
#include <test_runner.hpp>
#include <popart/builder.hpp>
#include <popart/ir.hpp>
#include <popart/op/add.hpp>
#include <popart/op/init.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/reshape.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

using namespace popart;

// Model: N matmuls in series, sharing weights
//
// in0 -
//       \
//        Matmul0 - Relu0 - Matmul1 - Relu1 -  ... - MatmulN-1 - ReluN-1 - out
//       /                   /                  /        /
//  w0 -------------------------------------------------
//
// In the below test, N = 4
// 1. Verify that the partial gradients of w0 are added:
//                       Init - .
//    Reshape(Matmul3RhsGrad) -- + - .
//    Reshape(Matmul2RhsGrad) ------ + - .
//    Reshape(Matmul1RhsGrad) ---------- + - .
//    Reshape(Matmul0RhsGrad) -------------- + - VarUpdate
// 2. Verify that that the adds are scheduled as early as possible, minimising
//    the sum liveness of the gradient partials
// 3. Verify that they are in-place adds
BOOST_AUTO_TEST_CASE(Test0) {
  TestRunner runner;
  runner.isTraining = true;
  int numLayers     = 4;
  int size          = 100;
  // Weights are [size, size], Input and Acts are [1, size]

  TensorInfo wInfo{"FLOAT", std::vector<int64_t>{size, size}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;
  std::vector<float> w0Data(4 * 4, 0);
  ConstVoidData w0CVData{w0Data.data(), wInfo};

  runner.buildModel([&](auto &builder) {
    auto aiOnnx = builder.aiOnnxOpset9();
    TensorInfo inInfo{"FLOAT", std::vector<int64_t>{1, size}};
    auto input   = builder.addInputTensor(inInfo);
    auto w0      = builder.addInitializedInputTensor(w0CVData);
    TensorId out = input;
    for (int layer = 0; layer < numLayers; layer++) {
      out = aiOnnx.matmul({out, w0}, "mm_layer" + std::to_string(layer));
      out = aiOnnx.relu({out}, "relu_layer" + std::to_string(layer));
    }

    runner.anchors.emplace(getGradId(input), AnchorReturnType("ALL"));
    runner.opts.enableOutlining  = false; // to make introspecting the IR easy
    runner.opts.decomposeGradSum = true;
    runner.patterns              = Patterns(PatternsLevel::DEFAULT);
    runner.losses.push_back(new L1Loss(out, "l1Loss", 0.1, ReductionType::SUM));

    return out;
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

        // (1)
        // Check rhs input comes from the (numLayers-1-i)'th layer's
        // MatMulGradRhs
        Op *addRhsProducer     = op->input->tensor(addRhs)->getProducer();
        size_t mmLayer         = numLayers - 1 - addOpNumber;
        std::string mmLayerStr = "mm_layer" + std::to_string(mmLayer);
        BOOST_CHECK(addRhsProducer->getName().find(mmLayerStr) !=
                    std::string::npos);
        BOOST_CHECK(addRhsProducer->getName().find("MatMulOp_RhsReshape") !=
                    std::string::npos);
        // Trace back one op to find the MatMul
        Op *matmulGradRhs = addRhsProducer->input->tensor(0)->getProducer();
        BOOST_CHECK(matmulGradRhs->isConvertibleTo<MatMulOp>());

        // Check lhs input comes from previous add
        if (addOpNumber > 0) {
          auto prevAddOp = gradPartialAddsOrder.at(addOpNumber - 1);
          BOOST_CHECK(op->input->tensor(addLhs)->getProducer() == prevAddOp);
        }

        // (2)
        // Check that the i'th add comes just after the (numLayers-1-i)'th
        // layer's MatMulRhsGradOp in the schedule (i.e. as soon as possible
        // after the large weight partial tensor becomes live)
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

    // (3)
    BOOST_CHECK_EQUAL(gradPartialAddsOrder.size(), numLayers);
  });
}
