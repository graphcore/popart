#define BOOST_TEST_MODULE PrePlanMatMulsTest

#include <chrono>
#include <cmath>
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

BOOST_AUTO_TEST_CASE(PrePlanMatMuls_0) {
  // Build the graph:
  //         w0 --.
  // in0 ------- MatMul0 -- Slice ----.
  //      |  w1 --.                    \
  //      |----- MatMul1 -- Slice ----. \
  //     ... wN --.                    \ \
  //      '----- MatMulN -- Slice ------ Sum - out

  // and verify that poplar compilation is shorter when
  // matmuls are preplanned
  auto test = [&](bool training) {
    auto builder                    = Builder::create();
    std::vector<int64_t> inputShape = {1, 2000};
    int numLayers;

    if (training) {
      numLayers = 3;
    } else {
      numLayers = 5;
    }

    auto ip0 = builder->addInputTensor("FLOAT", inputShape);
    std::vector<TensorId> outs;
    for (int layer = 0; layer < numLayers; layer++) {
      std::vector<int64_t> wShape = {2000, layer + 1};
      auto wN                     = builder->addInputTensor("FLOAT", wShape);
      auto matmul                 = builder->aiOnnxOpset6().matmul({ip0, wN});
      auto out =
          builder->aiOnnxOpset6().slice({matmul}, {1, 1}, {0, 0}, {0, 1});
      outs.push_back(out);
    }
    auto sum = builder->aiOnnxOpset6().sum(outs);
    auto out = builder->aiGraphcoreOpset1().identityloss({sum});

    auto modelProto = io::getModelFromString(builder->getModelProto());

    auto getCompileTime = [&](bool prePlanMatMuls) {
      auto device = createTestDevice(TEST_TARGET);
      SessionOptions opts;

      std::vector<TensorId> anchorIds = outs;
      TensorId loss                   = "";
      auto optimizer                  = ConstSGD(0.1);
      Optimizer *optimizer_ptr        = nullptr;

      if (training) {
        anchorIds.push_back(reservedGradientPrefix() + ip0);
        loss          = out;
        optimizer_ptr = &optimizer;
      }

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

      devicex->prePlanMatMuls = prePlanMatMuls;

      auto start = std::chrono::high_resolution_clock::now();
      devicex->prepare();
      auto finish = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> compileTime = finish - start;

      return compileTime.count();
    };

    // Verify that compilation is faster when pre-planning matmuls
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