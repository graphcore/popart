// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE DecomposeGradientSummationTest

#include <boost/test/unit_test.hpp>

#include <test_runner.hpp>

#include <popart/builder.hpp>
#include <popart/ir.hpp>
#include <popart/op/add.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/init.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/mean.hpp>
#include <popart/op/mul.hpp>
#include <popart/op/reshape.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

#include <string>

using namespace popart;

namespace {
// Call with `ShouldLog` = true to get `logging::debug` messages that log
// the schedule.
template <bool ShouldLog>
void debugLogSchedule(const std::vector<Op *> &schedule,
                      const std::string &testName);
template <>
void debugLogSchedule<false>(const std::vector<Op *> &schedule,
                             const std::string &testName);
template <>
void debugLogSchedule<true>(const std::vector<Op *> &schedule,
                            const std::string &testName);

using OpPredicate = bool (*)(const Op *);

void CHECK_commonAddOpChecks(const int numLayers,
                             std::vector<Op *> &schedule,
                             OpPredicate shouldTestAddOp,
                             const bool shouldCheckPartialsMergedImmediately);
} // namespace

// [test-0]
//
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
BOOST_AUTO_TEST_CASE(TestWeightSharingWhereWeightsMuchLargerThanActivations) {
  TestRunner runner;
  runner.isTraining = true;
  int numLayers     = 4;
  int size          = 100;
  // Weights are [size, size], Input and Acts are [1, size]

  TensorInfo wInfo{"FLOAT", std::vector<int64_t>{size, size}};
  std::vector<float> w0Data(wInfo.nelms(), 0);
  ConstVoidData w0CVData{w0Data.data(), wInfo};

  runner.buildModel([&](Builder &builder) {
    auto aiOnnx = builder.aiOnnxOpset10();
    TensorInfo inInfo{"FLOAT", std::vector<int64_t>{1, size}};
    auto input   = builder.addInputTensor(inInfo);
    auto w0      = builder.addInitializedInputTensor(w0CVData);
    TensorId out = input;
    for (int layer = 0; layer < numLayers; layer++) {
      out = aiOnnx.matmul({out, w0}, "mm_layer" + std::to_string(layer));
      out = aiOnnx.relu({out}, "relu_layer" + std::to_string(layer));
    }
    out = builder.aiGraphcoreOpset1().l1loss(
        {out}, 0.1, ReductionType::Mean, "loss");

    runner.anchors.emplace(getGradId(input), AnchorReturnType("All"));
    runner.opts.enableOutlining  = false; // to make introspecting the IR easy
    runner.opts.decomposeGradSum = true;
    runner.patterns              = Patterns(PatternsLevel::Default);
    runner.loss                  = out;

    return out;
  });

  runner.checkIr([&](Ir &ir) {
    std::vector<Op *> schedule =
        ir.getOpSchedule({}, RequireOptimalSchedule::Yes);

    debugLogSchedule<false>(
        schedule, "TestWeightSharingWhereWeightsMuchLargerThanActivations");

    const auto isGradAdd = [](const Op *op) {
      return op->isConvertibleTo<AddLhsInplaceOp>() ||
             op->isConvertibleTo<AddRhsInplaceOp>();
    };

    // Because the weights are much larger than the activations, to minimise
    // liveness, it is imperative to merge the partial gradients immediately
    // after they are produced.
    const bool checkPartialsMergedImmediately = true;

    CHECK_commonAddOpChecks(
        numLayers, schedule, isGradAdd, checkPartialsMergedImmediately);
  });
}

// Same as [test-0], but with skip connections.
// We check the tree order is still correct: in the order that the partial
// weight gradients are created.
BOOST_AUTO_TEST_CASE(
    TestWeightSharingWithSkipConnectionsWhereWeightsMuchSmallerThanActivations) {
  TestRunner runner;
  runner.isTraining = true;
  int numLayers     = 4;
  int size          = 100;
  // Weights are [size, size], Input and Acts are [size*numLayers, size]

  TensorInfo wInfo{"FLOAT", std::vector<int64_t>{size, size}};
  std::vector<float> w0Data(wInfo.nelms(), 0);
  ConstVoidData w0CVData{w0Data.data(), wInfo};

  runner.buildModel([&](auto &builder) {
    auto aiOnnx = builder.aiOnnxOpset10();
    TensorInfo inInfo{"FLOAT", std::vector<int64_t>{size * numLayers, size}};
    auto input   = builder.addInputTensor(inInfo);
    auto w0      = builder.addInitializedInputTensor(w0CVData);
    TensorId out = input;
    TensorId temp;
    for (int layer = 0; layer < numLayers; layer++) {
      temp = aiOnnx.matmul({out, w0}, "mm_layer" + std::to_string(layer));
      temp = aiOnnx.relu({temp}, "relu_layer" + std::to_string(layer));
      out =
          aiOnnx.add({out, temp}, "skip_connect_layer" + std::to_string(layer));
    }

    out = builder.aiGraphcoreOpset1().l1loss({out}, 0.1);

    runner.anchors.emplace(getGradId(input), AnchorReturnType("ALL"));
    runner.opts.enableOutlining  = false; // to make introspecting the IR easy
    runner.opts.decomposeGradSum = true;
    runner.patterns              = Patterns(PatternsLevel::Default);
    runner.loss                  = out;

    return out;
  });

  runner.checkIr([&](Ir &ir) {
    std::vector<Op *> schedule =
        ir.getOpSchedule({}, RequireOptimalSchedule::Yes);

    debugLogSchedule<false>(schedule,
                            "TestWeightSharingWithSkipConnectionsWhereWeightsMu"
                            "chSmallerThanActivations");

    // This will match only the adds from the grad sum decomposition of the
    // weight gradient, not from the skip connections.
    const auto isGradAdd = [](const Op *op) {
      return op->getName().find("GradAdd") != std::string::npos;
    };

    // Because the activations are much larger than the weights, peak liveness
    // is reached at the start of the backward pass, when all activations and
    // the first gradient activation are live. After this, as we proceed
    // backwards, we free the forward and gradient activations of a layer.
    //
    // Thus, as the weights are much smaller than the activations and we will
    // have already gone past the point of peak liveness, it does not matter
    // whether or not we merge the weight gradient partials immediately after
    // they are produced.
    const auto checkPartialsMergedImmediately = false;

    CHECK_commonAddOpChecks(
        numLayers, schedule, isGradAdd, checkPartialsMergedImmediately);
  });
}

/*
 Test the transform can handle a split-merge whose branch is itself a
 split-merge, resulting in grad sum whose output is itself an input to another
 grad sum.

 That is, the network (of tensors) looks like:

 t0 -> t1 -> t2 -> t3 ---
  |     |                 \
  |     |--> t4 -> t5 ------> t8 -----------------> t11
  |     |                 /                   /
  |     |--> t6 -> t7 ---                    /
  |                                         /
  |-------------------------> t9 ----------/
  |                                       /
  |-------------------------> t10 --------

 Resulting in the backwards of t0 and t1 looking like:
 (where ops are in capital case and tensors in all lower case)

         --<-- t0                                 --<-- t1
         |                                        |
 t0' <- T0' <- sum0 <- Sum0 <------------ t1' <- T1' <- sum1 <- Sum1
                        |                                       |
                        |--<-- t9', t10'                        |- t2', t4', t6'


 You can see that the inner sum operator, Sum1, goes into the actual grad
 operator of the forward op, T1', and not directly into the outer sum
 operator, Sum0.

 This means that nested split-merges result in grad sums that are not directly
 nested one-into-the-other, so this case is no different from two regular,
 separate grad sums.

 Therefore, as inspecting the Ir and schedule is pretty brittle (given all the
 patterns, transforms and more that could occur), we do not rigorously inspect
 them in this test. In fact, this test will simply be that no error is thrown;
 and to give documentation (and explanation) of the fact that nested grad sums
 work.

 There are already other tests that establish the correctness of the transform,
 so introducing extra brittleness here is unneccesary.
 */
BOOST_AUTO_TEST_CASE(TestNestedSums) {
  TestRunner runner;
  runner.isTraining = true;
  int size          = 32;

  TensorInfo tInfo{"FLOAT", std::vector<int64_t>{10, 10}};
  std::vector<float> wData(tInfo.nelms(), 0);
  ConstVoidData wCVData{wData.data(), tInfo};

  const auto modelBuilder = [&](Builder &builder) {
    auto aiOnnx      = builder.aiOnnxOpset10();
    auto aiGraphcore = builder.aiGraphcoreOpset1();

    auto w     = builder.addInitializedInputTensor(wCVData, "w");
    auto input = builder.addInputTensor(tInfo, "input");

    auto t0  = aiOnnx.mul({w, input}, "0-mul");
    auto t1  = aiOnnx.relu({t0}, "1-relu");
    auto t2  = aiOnnx.relu({t1}, "2-relu");
    auto t3  = aiOnnx.relu({t2}, "3-relu");
    auto t4  = aiOnnx.relu({t1}, "4-relu");
    auto t5  = aiOnnx.relu({t4}, "5-relu");
    auto t6  = aiOnnx.relu({t1}, "6-relu");
    auto t7  = aiOnnx.relu({t6}, "7-relu");
    auto t8  = aiOnnx.mean({t3, t5, t7}, "8-mean");
    auto t9  = aiOnnx.relu({t0}, "9-relu");
    auto t10 = aiOnnx.relu({t0}, "10-relu");
    auto t11 = aiOnnx.mean({t8, t9, t10}, "11-mean");

    auto tLoss = aiGraphcore.l1loss({t11}, 0.1, ReductionType::Mean, "loss");

    runner.anchors.emplace(getGradId(w), AnchorReturnType("All"));

    runner.opts.enableOutlining  = false; // to make introspecting the IR easy
    runner.opts.decomposeGradSum = true;
    runner.patterns              = Patterns(PatternsLevel::Default);
    runner.loss                  = tLoss;

    return tLoss;
  };

  BOOST_CHECK_NO_THROW(runner.buildModel(modelBuilder));

  runner.checkIr([&](Ir &ir) {
    std::vector<Op *> schedule =
        ir.getOpSchedule({}, RequireOptimalSchedule::Yes);

    debugLogSchedule<false>(schedule, "TestNestedSums");
  });
}

namespace {

template <>
void debugLogSchedule<false>(const std::vector<Op *> &schedule,
                             const std::string &testName) {
  (void)schedule;
  (void)testName;
}

template <>
void debugLogSchedule<true>(const std::vector<Op *> &schedule,
                            const std::string &testName) {
  const auto oldLevel = logging::getLogLevel(logging::Module::popart);
  logging::setLogLevel(logging::Module::popart, logging::Level::Debug);

  std::ostringstream log;

  log << "Op schedule for: " << testName << std::endl;

  for (Op *p : schedule) {
    log << "-> " << p->getName() << ": (" << p->id << ")" << std::endl
        << "      " << p->debugName() << std::endl;
  }
  logging::debug(log.str());

  logging::setLogLevel(logging::Module::popart, oldLevel);
}

// Performs certain assertions on the schedule using BOOST_CHECK. These are
// as follows:
//
// For every op in the `schedule` s.t. `shouldTestAddOp(op)`:
//   let i = schedule idx of op.
//
//   (1): Check rhs input comes from the (`numLayers`-i-1)'th layer's
//        MatMulGradRhs.
//        That is, check the tree order chosen by the decomposition is correct:
//        merge the partials in the order they are produced, which is the
//        reverse order the corresponding fwd ops appear in the fwd pass.
//
//   if (`shouldCheckPartialsMergedImmediately`):
//     (2): Check that the i'th add comes just after the (`numLayers`-1-i)'th
//          layer's MatMulRhsGradOp in the schedule (i.e. as soon as possible
//          after the large weight partial tensor becomes live)
//
// (3): Check the number of grad partial updates found == `numLayers`.
void CHECK_commonAddOpChecks(const int numLayers,
                             std::vector<Op *> &schedule,
                             OpPredicate shouldTestAddOp,
                             const bool shouldCheckPartialsMergedImmediately) {
  std::vector<Op *> gradPartialAddsOrder;
  size_t addOpNumber = 0;

  for (size_t i = 0; i < schedule.size(); i++) {
    Op *op = schedule.at(i);

    if (!shouldTestAddOp(op)) {
      continue;
    }

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

    if (shouldCheckPartialsMergedImmediately) {
      // (2)
      // Check that the i'th add comes just after the (numLayers-1-i)'th
      // layer's MatMulRhsGradOp in the schedule (i.e. as soon as possible
      // after the large weight partial tensor becomes live)

      // clang-format off
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
      // clang-format on

      if (addOpNumber == 0) {
        BOOST_CHECK(schedule.at(i - 1)->isConvertibleTo<InitOp>());
        checkForMatMulRhsReshape(schedule.at(i - 2));
        checkForMatMulRhsGrad(schedule.at(i - 3));
      } else {
        checkForMatMulRhsReshape(schedule.at(i - 1));
        checkForMatMulRhsGrad(schedule.at(i - 2));
      }
    }

    addOpNumber++;
  }

  // (3)
  BOOST_CHECK_EQUAL(gradPartialAddsOrder.size(), numLayers);
}

} // namespace

/*
  A NOTE ON RECOMPUTATION

  The interplay of the decomposegradsum transform and the scheduler does not
  work optimally under certain conditions when using recomputation, due to the
  implementation of recomputation in popart. We will explain this below.

  The specific case is when layers share weights and the activations are much
  larger than those weights.

  ------------------------------------------------------------------------------

  As the above tests assert, in a model where the layers share weights
  and the activations are much larger than the weights, the scheduler will
  (correclty) minimise sum liveness by pushing the weight grad adds to the end
  of the schedule.

  To clarify why:

  The scheduler is trying to minimise `memory allocated X time its live`, and as
  the activations are bigger, it wants to proceed further backwards first so it
  can keep clearing activations and their gradients as it does so - so the
  partial weight gradient additions get pushed back.

  That is, the scheduler decides having the weight grads live for an extra step,
  as long as it gets to clear the activations one step earlier, is better than
  immediately clearing the weight grad and having to keep the activation for an
  extra step.

  More precisely (though with some simplification), it calculates:

    size(activation) * t + size(weights) * (t'+1)

  is less than

    size(activation) * (t+1) + size(weights) *t'

  In the above test case, at every layer, it keeps deciding that doing this and
  delaying the add another step is better, so they get pushed back all the way
  to the end.

  ------------------------------------------------------------------------------

  The next preliminary is understanding how recomputation is implemented in
  popart.

  Recomputation is not explicitly in the graph. The ops to be recomputed are
  annotated, then we add them to poplar twice. That is not the best explanation,
  but the important point is that there is no recomputation in the graph, so the
  scheduler just sees a regular graph; it doesn't know about recomputation!

  ------------------------------------------------------------------------------

  Finally, we again consider the case where we are using recomputation, the
  layers share weights, and activations are much larger than weights. We will
  show why the scheduler fails to find the optimal strategy.

  Using the same reasoning as above, once we've done backpropagation on a
  segment, we will still need to merge all the partial weight grads, because we
  chose to delay them to the end as activations are much larger.

  Now, because we're doing recomputation, none of the activations from the next
  segment have been allocated yet, thus they are not contributing to sum
  liveness. The still live weight partials are though, so we should merge them
  immediately!

  However, because the scheduler doesn't know that next segment's activations
  aren't live, it uses the usual logic to decide it needs to keep proceeding
  backwards to minimise sum liveness, rather than merging the weight partials
  first.

  Thus, the scheduler has failed to do the right thing under recomputation
  because it is not aware of recomputation and its effects on liveness.

  Note, if the weights were much larger than the activations, we are always
  merging the weight partials immediately anyway. Whether or not the next
  segment's activations really are live has no effect, so the scheduler still
  does the right thing without knowing about recomputation.
*/
