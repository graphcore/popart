// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PrePlanConvolutionsTest

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <filereader.hpp>
#include <memory>
#include <onnx/onnx_pb.h>
#include <string>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/executablex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/sgd.hpp>
#include <popart/tensornames.hpp>
#include <popart/testdevice.hpp>

#include "popart/builder.gen.hpp"
#include "popart/inputshapeinfo.hpp"
#include "popart/names.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensordebuginfo.hpp"

using namespace popart;

bool prePlanningFailed(const internal_error &ex) {
  return std::string(ex.what()).find("Pre-planning failed for ") !=
         std::string::npos;
}

BOOST_AUTO_TEST_CASE(PrePlanConvolutions_0) {
  // Build the graph:
  //
  // in0 -- Conv0 - ... - ConvN - IdLoss - out
  //  w0 ----'         wN --'
  //
  // and verify that the 'pre-plan conv' option is turned off that
  // the run-time check throws an exception
  auto builder = Builder::create();
  std::vector<int64_t> input_shape;
  int numLayers;

  input_shape = {1, 1, 20, 20};
  numLayers   = 3;

  auto ip0     = builder->addInputTensor("FLOAT", input_shape);
  TensorId out = ip0;
  TensorId wN;
  for (int layer = 0; layer < numLayers; layer++) {
    wN  = builder->addInputTensor("FLOAT", std::vector<int64_t>{1, 1, 5, 5});
    out = builder->aiOnnxOpset10().conv({out, wN});
  }
  out = builder->aiGraphcoreOpset1().identityloss({out});

  auto modelProto = io::getModelFromString(builder->getModelProto());

  SessionOptions opts;

  std::vector<TensorId> anchorIds = {out};
  anchorIds.push_back(reservedGradientPrefix() + ip0);
  anchorIds.push_back(reservedGradientPrefix() + wN);
  auto optimizer = ConstSGD(0.1);

  auto compileModel = [&](bool prePlanConvolutions) {
    auto device = createTestDevice(TEST_TARGET);

    Ir ir;
    ir.prepare({modelProto,
                InputShapeInfo(),
                DataFlow(1, anchorIds),
                out,
                &optimizer,
                *device,
                opts,
                Patterns(PatternsLevel::All)});

    std::unique_ptr<popx::IrLowering> lowering;
    lowering.reset(new popx::IrLowering(ir, device));

    std::unique_ptr<popx::Executablex> executable =
        popx::Executablex::createFromLoweredIr(*lowering);

    const auto devicex = std::make_unique<popx::Devicex>(*executable, device);

    devicex->prePlanConvolutions = prePlanConvolutions;

    devicex->prepare();
  };

  compileModel(true); // No run-time exception; pre-planning works
  BOOST_CHECK_EXCEPTION(compileModel(false), internal_error, prePlanningFailed);
}
