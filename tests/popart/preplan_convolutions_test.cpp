#define BOOST_TEST_MODULE PrePlanConvolutionsTest

#include <chrono>
#include <memory>
#include <vector>

#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/ir.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

#include <popart/popx/devicex.hpp>
#include <popart/popx/executablex.hpp>
#include <popart/popx/irlowering.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(PrePlanConvolutions_0) {
  // Build the graph:
  //
  // in0 -- Conv0 - ... - ConvN - IdLoss - out
  //  w0 ----'         wN --'
  //
  // and verify that poplar compilation is shorter when
  // convolutions are preplanned
  auto test = [&](bool training) {
    auto builder = Builder::create();
    std::vector<int64_t> input_shape;
    int numLayers;

    if (training) {
      input_shape = {1, 1, 50, 50};
      numLayers   = 5;
    } else {
      input_shape = {1, 1, 100, 100};
      numLayers   = 8;
    }

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
    TensorId loss                   = "";
    auto optimizer                  = ConstSGD(0.1);
    Optimizer *optimizer_ptr        = nullptr;

    if (training) {
      anchorIds.push_back(reservedGradientPrefix() + ip0);
      anchorIds.push_back(reservedGradientPrefix() + wN);
      loss          = out;
      optimizer_ptr = &optimizer;
    }

    auto getCompileTime = [&](bool prePlanConvolutions) {
      auto device = createTestDevice(TEST_TARGET);

      Ir ir;
      ir.prepare({modelProto,
                  InputShapeInfo(),
                  DataFlow(1, anchorIds),
                  loss,
                  optimizer_ptr,
                  *device,
                  opts,
                  Patterns(PatternsLevel::All)});

      std::unique_ptr<popx::IrLowering> lowering;
      lowering.reset(new popx::IrLowering(ir, device));

      std::unique_ptr<popx::Executablex> executable =
          std::move(popx::Executablex::createFromLoweredIr(*lowering));

      const auto devicex = std::make_unique<popx::Devicex>(*executable, device);

      devicex->prePlanConvolutions = prePlanConvolutions;

      auto start = std::chrono::high_resolution_clock::now();
      devicex->prepare();
      auto finish = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> compileTime = finish - start;

      return compileTime.count();
    };

    // Verify that compilation is faster when pre-planning convs
    // by at least 15%
    // TODO T31376: The feature being tested relies on multi-processing.
    // It is too flaky to compare wall-clock times when running on buildbots
    // (different hardware, under different loads). Leaving this check
    // commented out for now.
    // BOOST_CHECK(getCompileTime(false) > getCompileTime(true) * 1.15);
  };

  test(false); // inference
  test(true);  // training
}