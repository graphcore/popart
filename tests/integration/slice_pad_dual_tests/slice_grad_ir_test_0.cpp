// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ConverSliceGradsTest

#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <filereader.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/pad.hpp>
#include <popart/op/slice.hpp>
#include <popart/sgd.hpp>
#include <popart/testdevice.hpp>

#include "popart/builder.gen.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/scheduler_requireoptimal.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensornames.hpp"

using namespace popart;

BOOST_AUTO_TEST_CASE(ConverSliceGrads0) {

  //                 input         .
  //                /  |              .
  //              relu |      .
  //                \  |            .
  //                 add        .
  //                 /||\       .
  //       slice slice slice slice       .
  //           \  /     |      |     .
  //             mul    |      |   .
  //                \  /       |       .
  //                 mul       |   .
  //                    \     /     .
  //                      mul
  //                       |             .
  //                     l1 loss
  //
  //
  // In this test, we answer:
  //
  // In the backwards pass, what happens to the slices?
  //
  // With inplacing enabled and the padsum pattern enabled, we expect
  // the slice-gradients to be absorbed into a concat inplace.
  //
  // This test checks all combinations of inplacing and the padsum pattern.
  //

  auto test = [](bool inplace, bool padsum) {
    // Build an onnx model (for training)
    auto builder = Builder::create();
    auto aiOnnx  = builder->aiOnnxOpset9();

    TensorInfo shape{"FLOAT", std::vector<int64_t>{5, 4}};
    auto in0    = builder->addInputTensor(shape);
    auto h0     = aiOnnx.relu({in0});
    auto h3     = aiOnnx.add({h0, in0});
    auto slice0 = aiOnnx.slice({h3}, {5, 1}, {0, 0});
    auto slice1 = aiOnnx.slice({h3}, {5, 2}, {0, 1});
    auto slice2 = aiOnnx.slice({h3}, {5, 3}, {0, 2});
    auto slice3 = aiOnnx.slice({h3}, {5, 4}, {0, 3});

    auto out = aiOnnx.mul({slice0, slice1});
    out      = aiOnnx.mul({slice2, out});
    out      = aiOnnx.mul({slice3, out});

    auto l1 = builder->aiGraphcoreOpset1().l1loss({out}, 0.1);

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    // Create the IR
    auto dataFlow =
        DataFlow(1,
                 {{out, AnchorReturnType("All")},
                  {reservedGradientPrefix() + in0, AnchorReturnType("All")}});
    auto optimizer = ConstSGD(0.01);
    auto device    = createTestDevice(TEST_TARGET);

    Ir ir;
    ir.prepare({modelProto,
                InputShapeInfo(),
                dataFlow,
                l1,
                &optimizer,
                *device,
                {},
                Patterns(PatternsLevel::Default)
                    .enableInPlace(inplace)
                    .enablePattern("PadSum", padsum)});

    std::cout << "\n\n\nwith inplace = " << inplace
              << ", and with padsum = " << padsum << std::endl;
    auto schedule = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
    int nConcatInplace{0};
    int nConcat{0};
    int nPadInplace{0};
    int nSliceGrad{0};

    for (auto op : schedule) {

      nConcatInplace += (dynamic_cast<const ConcatInplaceOp *>(op) != nullptr);

      nConcat += ((dynamic_cast<const ConcatInplaceOp *>(op) == nullptr) &&
                  (dynamic_cast<const ConcatOp *>(op) != nullptr));

      // padinplace
      nPadInplace += (dynamic_cast<const PadInplaceOp *>(op) != nullptr);

      nSliceGrad += (dynamic_cast<const SliceGradOp *>(op) != nullptr);
    }

    std::ostringstream oss;
    oss << "Failed in test of inplacing and padsum reduction to concat. ";
    oss << "In this test, we assume that mul does not have an inplace op, and "
           "that"
        << " inplacing priorities are set appropriately - this might need "
        << " reconsidering. ";
    if (inplace && padsum && nConcatInplace == 0) {
      oss << "With inplacing and the padsum pattern enabled, expected "
          << "a concat inplace. ";
      throw error(oss.str());
    }

    if (!inplace && padsum && nConcat == 0) {
      oss << "With no inplacing and the padsum pattern enabled, expected a "
             "concat ";
      throw error(oss.str());
    }

    if (inplace && !padsum && nPadInplace == 0) {
      oss << "With inplacing and no padsum pattern, expected "
          << "at least 1 pad inplace (in backwards pass). ";
      throw error(oss.str());
    }

    if (!inplace && !padsum && nSliceGrad == 0) {
      oss << "With no inplacing and no padsum pattern, expected "
          << "at least 1 slicegrad (in backwards pass). ";
      throw error(oss.str());
    }
  };

  test(true, true);
  test(true, false);
  test(false, true);
  test(false, false);
}
