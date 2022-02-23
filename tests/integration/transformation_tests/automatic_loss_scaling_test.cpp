// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE AutomaticLossScalingTest
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <vector>

#ifdef __clang__
#pragma clang diagnostic ignored "-Wkeyword-macro"
#endif
#define protected public
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/op.hpp>
#include <popart/op/autolossscaleproxy.hpp>
#include <popart/op/histogram.hpp>
#include <popart/op/mul.hpp>
#include <popart/op/scale.hpp>
#include <popart/op/sigmoid.hpp>
#include <popart/optimizer.hpp>
#include <popart/session.hpp>
#include <popart/sgd.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/testdevice.hpp>
#undef protected

using namespace popart;

template <class T> std::vector<T *> getOpsOfType(const Ir &ir) {
  std::vector<T *> ops;
  for (auto &id_op : ir.getMainGraphOps()) {
    auto op = id_op.second.get();
    if (op->isConvertibleTo<T>()) {
      ops.push_back(dynamic_cast<T *>(op));
    }
  }
  return ops;
}

BOOST_AUTO_TEST_CASE(AutomaticLossScalingTest0) {
  // Model :
  //
  //  Input           Weights
  //     |              |
  //     +--- [mul] ----+
  //            |
  //          Mul:0
  //            |
  //         [sigmoid]
  //            |
  //          [scale]
  //            |
  //          [L1 loss]
  //
  // If user select forward tensor Mul:0 to be als tracked we temporarily add
  // [AutoLossScaleProxy] op and due to auto grad [AutoLossScaleProxyGrad]
  // is added in backward pass.
  //
  //  Input           Weights      Input           Weights
  //     |              |             |                |
  //     +--- [mul] ----+             +--- [G mul] ----+
  //            |                             |
  //           Mul:0                   Gradient___Mul:0
  //            |                             |
  //      [AutoLossScaleProxy]       [AutoLossScaleProxyGrad]
  //            |                             |
  //        Mul:0_AlsProxy           Gradient___Mul:0_AlsProxy
  //            |                             |
  //         [sigmoid]                   [G sigmoid]
  //            |                             |
  //   |      [scale]             ^       [G scale]
  //   |        |                 |           |
  //   V     [L1 loss]            |       [L1 loss]
  //
  //
  // We test that Proxy ops and tensors are removed
  // and that Gradient___Mul:0 tensor is tracked.

  // Build model to run.
  TensorInfo weightInfo{"FLOAT16", std::vector<int64_t>{2, 2}};
  TensorInfo inputInfo{"FLOAT16", std::vector<int64_t>{2, 2}};

  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();
  auto input0      = builder->addInputTensor(inputInfo, "input");

  auto weights_init         = std::vector<float>(4, 0);
  ConstVoidData weights_cvd = {weights_init.data(), weightInfo};
  TensorId weights_id       = builder->addInitializedInputTensor(weights_cvd);

  auto act0         = aiOnnx.mul({weights_id, input0});
  act0              = aiOnnx.sigmoid({act0});
  act0              = aiGraphcore.scale({act0}, 2.0f);
  TensorId actFinal = act0;

  SessionOptions userOptions;
  userOptions.automaticLossScalingSettings.enabled        = true;
  userOptions.automaticLossScalingSettings.toTrackTensors = {"Mul:0"};
  userOptions.automaticLossScalingSettings.gradientTensorTrackingMethod =
      GradientTensorTrackingMethod::GradientsOfUserSpecifiedTensors;

  auto optimizer = SGD({{"lossScaling", {0.2f, false}}});

  float lambda  = 0.159;
  actFinal      = builder->aiGraphcoreOpset1().l1loss({actFinal}, lambda);
  auto proto    = builder->getModelProto();
  auto dataFlow = DataFlow(1);

  auto device = createTestDevice(TEST_TARGET);

  auto session = popart::TrainingSession::createFromOnnxModel(
      proto,
      dataFlow,
      actFinal,
      optimizer,
      device,
      InputShapeInfo(),
      userOptions,
      popart::Patterns(PatternsLevel::Default));

  session->prepareDevice();

  // Testing.
  auto &ir = session->getIr();

  auto alsOpPs  = getOpsOfType<AutoLossScaleProxyOp>(ir);
  auto alsOpGPs = getOpsOfType<AutoLossScaleProxyGradOp>(ir);
  auto histOps  = getOpsOfType<HistogramOp>(ir);

  // Check AutoLossScaleProxy, AutoLossScaleProxyGrad ops removed.
  BOOST_CHECK(alsOpPs.size() == 0);
  BOOST_CHECK(alsOpGPs.size() == 0);
  BOOST_CHECK(histOps.size() == 1);
  // Check HistogramOp has correct tensor.
  BOOST_CHECK(histOps[0]->input->tensors().size() == 1);
  BOOST_CHECK(histOps[0]->input->tensors()[0]->str() == "Gradient___Mul:0");

  // Check Proxy tensors removed.
  std::vector<TensorId> tensorsIds = ir.getMainGraphTensors().getAllTensorIds();
  bool alsProxyTensorsExist =
      std::any_of(tensorsIds.begin(), tensorsIds.end(), [](const TensorId &x) {
        return x == "Mul:0_AlsProxy" || x == "Gradient___Mul:0_AlsProxy";
      });

  BOOST_CHECK(alsProxyTensorsExist == false);
}
