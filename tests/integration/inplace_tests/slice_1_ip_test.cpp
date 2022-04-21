// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Slice1InplaceTest

#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/nll.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>
#include <popart/topocons.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(Inplace_SlicesOverlap) {

  // In this test, slice regions overlap:
  //
  //             |-(A)- Slice [0->4, 0->4] -- Relu --|
  //             |                                   |
  // in0 (4x6) --|                                   |-- Matmul --
  //             |                                   |
  //             |-(B)- Slice [0->4, 2->6] --- Exp --|
  //
  //
  // we test with various priorities for the four ops
  // 1) Relu > Exp > Slice(B) > Slice(A)
  // 2) ...
  //    etc etc etc.
  //
  // The rules of what should happen with different priorities:
  // - the last op to be attempted can never be inplaced.
  // - if both slices are in-place, then there is a constraint
  //   on the ordering. Example, if A-slice, B-slice and Relu are
  //   inplace, then Exp must run before Relu-inplace.

  struct Priorities {
    float relu;
    float exp;
    float sliceA;
    float sliceB;
  };

  auto test = [](Priorities priorities) {
    // Build an onnx model
    auto builder = Builder::create();
    auto aiOnnx  = builder->aiOnnxOpset9();

    TensorInfo shape0{"FLOAT", std::vector<int64_t>{4, 6}};

    auto in0 = builder->addInputTensor(shape0);

    std::string asliceName = "slice_name_alice";
    auto slA = aiOnnx.slice({in0}, {4, 4}, {0, 0}, {0, 1}, asliceName);
    builder->setInplacePreferences(slA, {{"SliceInplace", priorities.sliceA}});

    std::string bsliceName = "slice_name_bob";
    auto slB = aiOnnx.slice({in0}, {4, 6}, {0, 2}, {0, 1}, bsliceName);
    builder->setInplacePreferences(slB, {{"SliceInplace", priorities.sliceB}});

    auto r0 = aiOnnx.relu({slA});
    builder->setInplacePreferences(r0, {{"ReluInplace", priorities.relu}});

    auto e0 = aiOnnx.exp({slB});
    builder->setInplacePreferences(e0, {{"ExpInplace", priorities.exp}});

    auto out = aiOnnx.matmul({r0, e0});
    builder->addOutputTensor(out);

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    // Create the IR
    auto dataFlow = DataFlow(1, {{out, AnchorReturnType("All")}});

    auto device = createTestDevice(TEST_TARGET);

    Ir ir;
    ir.prepare({modelProto,
                InputShapeInfo(),
                dataFlow,
                {},
                nullptr,
                *device,
                {},
                Patterns(PatternsLevel::NoPatterns)
                    .enableRuntimeAsserts(false)
                    .enableInPlace(true)});

    // get the index of the first op of type "identifier"
    auto getFirstPtrIndex = [](OperatorIdentifier identifier,
                               const std::vector<Op *> &schOps) {
      auto found = std::find_if(
          schOps.cbegin(), schOps.cend(), [identifier](const Op *op) {
            return op->opid == identifier;
          });
      if (found == schOps.cend()) {
        throw error("Failed to find op of type {} in schedule", identifier);
      }
      auto dist = std::distance(schOps.cbegin(), found);
      return dist;
    };

    auto oneSliceInplacedBoostCheck = [&ir] {
      BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu).size() == 0);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ReluInplace).size() == 1);
      BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Exp).size() == 0);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ExpInplace).size() == 1);
      BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Slice).size() == 1);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::SliceInplace).size() ==
                  1);
    };

    // slice B will not be in-placed if it has the lowest priority
    if ((priorities.sliceA > priorities.sliceB) &&
        (priorities.relu > priorities.sliceB) &&
        (priorities.exp > priorities.sliceB)) {
      oneSliceInplacedBoostCheck();
      auto notInplaceSlices = ir.opsOfType(Onnx::AiOnnx::OpSet9::Slice);
      // we have already confirmed that there is 1 not-inplace slice:
      if (notInplaceSlices.size() != 0) {
        auto op = notInplaceSlices.back();
        BOOST_CHECK(op->settings.name == bsliceName);
      }
    }

    // slice A will not be in-placed
    else if ((priorities.sliceB > priorities.sliceA) &&
             (priorities.exp > priorities.sliceA) &&
             (priorities.relu > priorities.sliceA)) {
      oneSliceInplacedBoostCheck();
      auto notInplaceSlices = ir.opsOfType(Onnx::AiOnnx::OpSet9::Slice);
      // we have already confirmed that there is 1 not-inplace slice:
      if (notInplaceSlices.size() != 0) {
        auto op = notInplaceSlices.back();
        BOOST_CHECK(op->settings.name == asliceName);
      }
    }

    // relu will not be in-placed
    else if ((priorities.sliceB > priorities.relu) &&
             (priorities.exp > priorities.relu) &&
             (priorities.sliceA > priorities.relu)) {
      BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu).size() == 1);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ReluInplace).size() == 0);
      BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Exp).size() == 0);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ExpInplace).size() == 1);
      BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Slice).size() == 0);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::SliceInplace).size() ==
                  2);

      // an explicit check for the constraint Relu --> ExpInplace
      auto reluOp = ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu).back();
      auto expInplaceOp =
          ir.opsOfType(Onnx::CustomOperators::ExpInplace).back();
      auto afterRelu = ir.getMainGraph().topoCons->getAfters(reluOp);
      BOOST_CHECK(std::find(afterRelu.cbegin(),
                            afterRelu.cend(),
                            expInplaceOp) != afterRelu.cend());

      // a secondary test that the exp-inplace runs after the relu
      auto schOps = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
      BOOST_CHECK(
          (getFirstPtrIndex(Onnx::AiOnnx::OpSet9::Relu, schOps) <
           getFirstPtrIndex(Onnx::CustomOperators::ExpInplace, schOps)));

    }

    // exp will not be in-placed
    else {
      BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu).size() == 0);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ReluInplace).size() == 1);
      BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Exp).size() == 1);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ExpInplace).size() == 0);
      BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Slice).size() == 0);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::SliceInplace).size() ==
                  2);

      // an explicit check for the constraint Exp --> ReluInplace
      auto expOp = ir.opsOfType(Onnx::AiOnnx::OpSet9::Exp).back();
      auto reluInplaceOp =
          ir.opsOfType(Onnx::CustomOperators::ReluInplace).back();
      auto afterExp = ir.getMainGraph().topoCons->getAfters(expOp);
      BOOST_CHECK(std::find(afterExp.cbegin(),
                            afterExp.cend(),
                            reluInplaceOp) != afterExp.cend());

      // a secondary test that the relu-inplace runs after the exp
      auto schOps = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
    }
  };

  Priorities priorities;

  // where NL : non-linearity and SL : a slice
  // NL, NL, SL, SL
  logging::ir::debug("Relu > Exp > SliceA > SliceB");
  priorities.relu   = 1e5;
  priorities.exp    = 1e4;
  priorities.sliceA = 1e3;
  priorities.sliceB = 1e2;
  test(priorities);

  // NL, SL, NL, SL
  logging::ir::debug("Relu > SliceA > Exp > SliceB");
  priorities.relu   = 1e5;
  priorities.sliceA = 1e4;
  priorities.exp    = 1e3;
  priorities.sliceB = 1e2;
  test(priorities);

  // NL, SL, SL, NL
  logging::ir::debug("Relu > SliceA > SliceB > Exp");
  priorities.relu   = 1e5;
  priorities.sliceA = 1e4;
  priorities.sliceB = 1e3;
  priorities.exp    = 1e2;
  test(priorities);

  // SL, SL, NL, NL
  logging::ir::debug("SliceA > SliceB > Exp > Relu");
  priorities.sliceA = 1e5;
  priorities.sliceB = 1e4;
  priorities.exp    = 1e3;
  priorities.relu   = 1e2;
  test(priorities);

  logging::ir::debug("SliceA > SliceB > Relu > Exp");
  priorities.sliceA = 1e5;
  priorities.sliceB = 1e4;
  priorities.relu   = 1e3;
  priorities.exp    = 1e2;
  test(priorities);

  logging::ir::debug("SliceB > SliceA > Relu > Exp");
  priorities.sliceB = 1e5;
  priorities.sliceA = 1e4;
  priorities.relu   = 1e3;
  priorities.exp    = 1e2;
  test(priorities);

  // SL, NL, SL, NL
  logging::ir::debug("SliceA > Exp > SliceB > Relu");
  priorities.sliceA = 1e5;
  priorities.exp    = 1e4;
  priorities.sliceB = 1e3;
  priorities.relu   = 1e2;
  test(priorities);

  // SL, NL, NL, SL
  logging::ir::debug("SliceA > Exp > Relu > SliceB");
  priorities.sliceA = 1e5;
  priorities.exp    = 1e4;
  priorities.relu   = 1e3;
  priorities.sliceB = 1e2;
  test(priorities);
}

// A future test:
//
// in0 : [3xJ] --------------|            |--- [Slice] (3->3, 3->6) ----|
//                           |            |                             |
//                           |- [Concat] -|                             |
//                           |            |                             |
// in1 : [3x(6-J)] - [Relu] -|            |--- [Slice] (0->3, 0->3) -|  |
//                                                                  /   |
//                                                                 /    |
//                                                                /     |
//                                                            [Exp]   [Exp]
//                                                              |       |
//                                                              ---------
//                                                                  |
//                                                               [Matmul]
