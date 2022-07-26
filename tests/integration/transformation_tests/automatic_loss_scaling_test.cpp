// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE AutomaticLossScalingTest
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "popart/builder.gen.hpp"
#include "popart/debugcontext.hpp"
#include "popart/inputshapeinfo.hpp"
#include "popart/ir.hpp"
#include "popart/names.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensornames.hpp"
#include "popart/tensors.hpp"
#include "popart/util.hpp"
#include "popart/vendored/optional.hpp"
#include "popart/voiddata.hpp"

#ifdef __clang__
#pragma clang diagnostic ignored "-Wkeyword-macro"
#endif
#define protected public
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/op/autolossscaleproxy.hpp>
#include <popart/op/histogram.hpp>
#include <popart/optimizer.hpp>
#include <popart/session.hpp>
#include <popart/sgd.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
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

BOOST_AUTO_TEST_CASE(TestGetInverseLossScaleTensors) {
  // Model :
  //
  //  Input           Weights1
  //     |              |
  //     +--- [mul] ----+
  //            |
  //         [detach]  Weights2
  //            |        |
  //          [mul] -----+
  //            |
  //         [batchNorm]
  //            |
  //         [sigmoid]
  //            |
  //          [scale]
  //            |
  //          [L1 loss]
  //
  // there's detach & batchNorm in the graph,
  // "getInverseLossScaleTensors" function should only pick the
  // variables which use a optimizer to do update.
  // detached tensors is not connected to optimizer
  // batch norm's running mean/var's VariableUpdateType==Copy
  // these tensors don't have related inverseLossScaleTensors

  // Build model to run.

  auto builder     = popart::Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();

  std::vector<std::vector<float>> inits;
  auto createWeight = [](popart::Builder *builder,
                         std::vector<std::vector<float>> inits,
                         std::vector<int64_t> shape) -> popart::TensorId {
    int64_t num_elements = 1;
    for (auto i : shape) {
      num_elements *= i;
    }
    inits.emplace_back(num_elements, 0);
    popart::ConstVoidData weights_cvd = {inits.back().data(),
                                         popart::TensorInfo("FLOAT16", shape)};
    popart::TensorId weights_id =
        builder->addInitializedInputTensor(weights_cvd);
    return weights_id;
  };
  auto batchNorm =
      [createWeight](popart::Builder *builder,
                     popart::AiOnnxOpset9 aiOnnx,
                     std::vector<std::vector<float>> inits,
                     popart::TensorId input_x,
                     int64_t num_features) -> std::vector<popart::TensorId> {
    popart::TensorId init_scale = createWeight(builder, inits, {num_features});

    popart::TensorId init_biases = createWeight(builder, inits, {num_features});
    popart::TensorId mean        = createWeight(builder, inits, {num_features});
    popart::TensorId var         = createWeight(builder, inits, {num_features});

    auto outs = aiOnnx.batchnormalization(
        {input_x, init_scale, init_biases, mean, var}, 5, 1e-5, 0.99);
    return {outs[0], init_scale, init_biases};
  };
  popart::TensorId weights1_id =
      createWeight(builder.get(), inits, std::vector<int64_t>{4, 4});
  popart::TensorId weights2_id =
      createWeight(builder.get(), inits, std::vector<int64_t>{4, 4});
  popart::TensorInfo input_info{"FLOAT16", std::vector<int64_t>{1, 8, 4, 4}};
  auto input0 = builder->addInputTensor(input_info, "input");

  auto act     = aiOnnx.mul({weights1_id, input0});
  act          = aiGraphcore.detach({act});
  act          = aiOnnx.mul({weights2_id, act});
  auto bn_outs = batchNorm(builder.get(), aiOnnx, inits, {act}, 8);
  act          = bn_outs[0];
  act          = aiOnnx.sigmoid({act});
  act          = aiGraphcore.scale({act}, 2.0f);

  popart::SessionOptions options;

  auto optimizer = popart::SGD({{"lossScaling", {0.2f, false}}});
  // when weights one specific hyper param,
  // all hyper params(like weightDecay or lossScaling) for those weights will be
  // specific tensors, so we only need to specify a specific learningRate.
  optimizer.insertSpecific(weights2_id,
                           std::map<std::string, std::pair<float, bool>>{
                               {"learningRate", {0.3f, false}}});
  // bn_outs[1] is the scale tensor of batchnorm
  // bn_outs[2] is the bias tensor of batchnorm
  // these two tensors are updated by optimzer using their gradients
  optimizer.insertSpecific(bn_outs[1],
                           std::map<std::string, std::pair<float, bool>>{
                               {"learningRate", {0.3f, false}}});
  optimizer.insertSpecific(bn_outs[2],
                           std::map<std::string, std::pair<float, bool>>{
                               {"learningRate", {0.3f, false}}});

  float lambda   = 0.159;
  auto act_final = builder->aiGraphcoreOpset1().l1loss({act}, lambda);
  auto proto     = builder->getModelProto();
  auto dataFlow  = popart::DataFlow(1);

  auto device  = popart::createTestDevice(popart::TEST_TARGET);
  auto session = popart::TrainingSession::createFromOnnxModel(
      proto, dataFlow, act_final, optimizer, device);

  session->prepareDevice();
  // a set of all tensors that used loss scaling.
  std::set<popart::TensorId> tensors_use_optimizer = {
      popart::reservedSpecificScaledLearningRate0Prefix() + weights2_id,
      popart::reservedSpecificScaledLearningRate0Prefix() + bn_outs[1],
      popart::reservedSpecificScaledLearningRate0Prefix() + bn_outs[2],
  };
  auto &ir = session->getIr();
  auto inverseLossScaleTensors =
      popart::getInverseLossScaleTensors(ir.getMainGraph());

  // check returned  inverseLossScaleTensors is the same as expected
  BOOST_CHECK(inverseLossScaleTensors.size() == 3);
  for (popart::Tensor *tensor : inverseLossScaleTensors) {
    BOOST_CHECK(tensors_use_optimizer.find(tensor->id) !=
                tensors_use_optimizer.end());
  }
}
