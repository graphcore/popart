// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PrePlanMatMulsTest

#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/sgd.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

#include <popart/popx/devicex.hpp>
#include <popart/popx/executablex.hpp>
#include <popart/popx/irlowering.hpp>

using namespace popart;

bool prePlanningFailed(const internal_error &ex) {
  return std::string(ex.what()).find("Pre-planning failed for ") !=
         std::string::npos;
}

BOOST_AUTO_TEST_CASE(PrePlanMatMuls_0) {
  // Build the graph:
  //         w0 -- TopK -.
  // in0 ------------ MatMul0 -- Slice ----.
  //      |  w1 -- TopK -.                  \
  //      |---------- MatMul1 -- Slice ----. \
  //     ... wN -- TopK -.                    \
  //      '---------- MatMulN -- Slice ------ Sum - out

  // and verify that the 'pre-plan matmul' option is turned off that
  // the run-time check throws an exception
  auto builder                    = Builder::create();
  std::vector<int64_t> inputShape = {1, 20};
  int numLayers                   = 3;

  auto ip0 = builder->addInputTensor("FLOAT", inputShape);
  std::vector<TensorId> outs;
  for (int layer = 0; layer < numLayers; layer++) {
    std::vector<int64_t> wShape = {23, layer + 1};
    auto wN                     = builder->addInputTensor("FLOAT", wShape);
    // Require a 'can-create-input' op before the matmul, such that the
    // subsequent matmul doesn't create it's input. Why?:
    // - we check at runtime (inside MatMulOpx::grow) if the plan has been
    //   cached from pre-planning.
    // - if it has not, we know that pre-planning has failed.
    // - but if pre-planning is not run, a plan is still created before the grow
    //   function when createWeights is called.
    // - By inserting a topk op, we make sure that the matmul isn't creating
    //   its weight input, making it easy to test that no plan is cached
    //   when pre-planning is turned off.
    auto topk   = builder->aiOnnxOpset6().topk({wN}, 20, 0)[0];
    auto matmul = builder->aiOnnxOpset6().matmul({ip0, topk});
    auto out = builder->aiOnnxOpset6().slice({matmul}, {1, 1}, {0, 0}, {0, 1});
    outs.push_back(out);
  }
  auto sum = builder->aiOnnxOpset6().sum(outs);
  auto out = builder->aiGraphcoreOpset1().identityloss({sum});

  auto modelProto = io::getModelFromString(builder->getModelProto());

  auto compileModel = [&](bool prePlanMatMuls) {
    auto device = createTestDevice(TEST_TARGET);
    SessionOptions opts;

    std::vector<TensorId> anchorIds = outs;
    anchorIds.push_back(reservedGradientPrefix() + ip0);
    TensorId loss  = out;
    auto optimizer = ConstSGD(0.1);

    Ir ir;
    ir.prepare({modelProto,
                InputShapeInfo(),
                DataFlow(1, anchorIds),
                loss,
                &optimizer,
                *device,
                opts,
                Patterns(PatternsLevel::All)});

    std::unique_ptr<popx::IrLowering> lowering;
    lowering.reset(new popx::IrLowering(ir, device));

    std::unique_ptr<popx::Executablex> executable =
        std::move(popx::Executablex::createFromLoweredIr(*lowering));

    const auto devicex = std::make_unique<popx::Devicex>(*executable, device);

    devicex->prePlanMatMuls = prePlanMatMuls;

    devicex->prepare();
  };

  compileModel(true); // No run-time exception; pre-planning works
  BOOST_CHECK_EXCEPTION(compileModel(false), internal_error, prePlanningFailed);
}