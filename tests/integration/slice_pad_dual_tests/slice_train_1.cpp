// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE slice_train_1

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <map>
#include <vector>

// This trick is required to access the Devicex's poplar::Tensors.
#define protected public
#define private public

#include <popart/builder.hpp>
#include <popart/error.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/session.hpp>
#include <popart/sgd.hpp>
#include <popart/testdevice.hpp>

#undef private
#undef protected

BOOST_AUTO_TEST_CASE(SliceTrain1) {

  // model to train:
  //
  // Weight -> slice -> relu -> slice -> l1_loss
  //                                       |
  //    L       M        M        S        |       (sizes)
  //                                       |
  // dWeight  dSlice   dRelu    dSlice <- dLoss
  //
  // In this test, we check that the layouts (tile mappings) of gradient
  // tensors are the same as the corresponding forward tensors.
  //
  // In particular, we are testing the logic in "getPropitiousPadLayout".
  //
  // The gradient of a slice is a Pad, how to layout the zeros? If
  // getPropitiousPadLayout is working correctly, the padding zeros should
  // derive their tile mapping from the tensor whose gradient they correspond
  // to.
  //
  // In this test, we confirm that Tensors of the same shapes have the same tile
  // mapping.
  //
  // Note : I have confirmed that disabling getPropitiousPadLayout results in
  // this no longer holding.
  //

  using namespace popart;

  std::map<std::string, std::pair<float, bool>> optParams;
  optParams.insert({"defaultLearningRate", {1.0f, true}});
  auto opt0 = SGD(optParams);

  // names of  weights used in model
  std::string w0name = "__w0__";

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  // weigh shape:  [weightDim0, weightDim1]
  int64_t weightDim0 = 100;
  int64_t weightDim1 = 100;
  auto nWeightVals   = weightDim0 * weightDim1;
  std::vector<int64_t> weightShape{weightDim0, weightDim1};
  TensorInfo weightInfo("FLOAT", weightShape);

  // the initial weight values
  std::vector<float> weight0(nWeightVals, 13.0f);

  // insert initialized weight Tensor into the ONNX model
  ConstVoidData cvd0({weight0.data(), weightInfo});
  auto w0Id = builder->addInitializedInputTensor(cvd0, w0name);

  // slice the weights into 2 parts, each the shape of a sample:
  auto slice0 = aiOnnx.slice({w0Id}, {75, 75}, {25, 25});
  auto relu0  = aiOnnx.relu({slice0});
  auto slice1 = aiOnnx.slice({w0Id}, {63, 63}, {35, 35});

  auto l1 = builder->aiGraphcoreOpset1().l1loss({slice1}, 1.0);
  builder->addOutputTensor(relu0);

  auto proto    = builder->getModelProto();
  auto dataFlow = DataFlow(1);

  SessionOptions userOptions;

  auto device = createTestDevice(TestDeviceType::IpuModel, 1, 20);

  auto session = popart::TrainingSession::createFromOnnxModel(
      proto,
      dataFlow,
      l1,
      opt0,
      device,
      InputShapeInfo(),
      SessionOptions(),
      popart::Patterns(PatternsLevel::Default));

  session->prepareDevice();

  using Mapping = poplar::Graph::TileToTensorMapping;

  // Tensor shape: tile mapping
  std::map<std::vector<size_t>, std::vector<Mapping>> mappings;
  const auto &graph = session->getDevice().lowering().graph();
  for (auto id : session->getDevice().lowering().tensors().tensors_) {
    auto shape = id.second->getPoplarTensor().shape();
    const auto tm =
        graph.getPoplarGraph().getTileMapping(id.second->getPoplarTensor());
    mappings[shape].push_back(tm);
    std::cout << tm << std::endl;
  }

  for (const auto &shape_mappings : mappings) {

    auto maps = shape_mappings.second;
    for (auto m : maps) {
      if (m != maps[0]) {
        std::ostringstream oss;
        oss << "In this test, we expect all Tensors of the same shape to have "
            << "the same tile mappings, It is a test of the slice grad (pad) "
               "to "
            << "correctly locate and clone a corresponding forward Tensor. "
            << "If correctly cloned, the tile mapping should be identical. ";
        throw popart::error(oss.str());
      }
    }
  }
}
