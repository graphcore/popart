// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE NumericsInplaceVsNot0Test

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/session.hpp>
#include <popart/tensordata.hpp>
#include <popart/testdevice.hpp>

#include <chrono>
#include <complex>
#include <iostream>
#include <random>

using namespace popart;

//           input
//         /   |   \
//   scale1 scale2 scale3
//
//   x = (scale1 + scale2) + scale3
//   y = (scale3 * scale2) * scale1
//
//   output = x - y

BOOST_AUTO_TEST_CASE(Inplace_numericsIpNip0) {
  auto getValue = [](bool outline, bool inplace) {
    Shape inShape{2, 2};
    TensorInfo inInfo{"FLOAT", inShape};

    std::vector<float> vInData{2.0, 0.5, 7.0, 3.125};

    // Build an onnx model
    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();

    auto in0  = builder->addInputTensor(inInfo);
    auto s1   = aiGraphcore.scale({in0}, 1.0);
    auto s2   = aiGraphcore.scale({in0}, 2.0);
    auto s3   = aiGraphcore.scale({in0}, 3.0);
    auto add0 = aiOnnx.add({s1, s2});
    auto x    = aiOnnx.add({add0, s3});
    auto mul0 = aiOnnx.mul({s3, s2});
    auto y    = aiOnnx.mul({mul0, s1});
    auto sub  = aiOnnx.sub({x, y});
    auto out  = aiOnnx.reducesum({sub}, std::vector<int64_t>{{0, 1}}, false);
    builder->addOutputTensor(out);

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    // Create the IR, adding outId as an anchor
    auto art      = AnchorReturnType("All");
    auto dataFlow = DataFlow(1, {{out, art}});

    auto opts = SessionOptions();
    opts.dotChecks.insert(DotCheck::Bwd0);
    opts.dotChecks.insert(DotCheck::Fwd0);
    opts.dotChecks.insert(DotCheck::Final);
    opts.dotOpNames       = false;
    opts.logDir           = "./dotfiles_ip_with_outlining";
    opts.enableOutlining  = outline;
    opts.outlineThreshold = 0.0f;
    if (!boost::filesystem::exists(opts.logDir)) {
      boost::filesystem::create_directory(opts.logDir);
    }

    auto device = popart::createTestDevice(TEST_TARGET);

    auto session = popart::InferenceSession::createFromOnnxModel(
        proto,
        dataFlow,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::NoPatterns).enableInPlace(inplace));

    // prepare the anchors
    float rawOutputData;
    Shape outShape{};
    popart::NDArrayWrapper<float> outData(&rawOutputData, outShape);

    std::map<popart::TensorId, popart::IArray &> anchors = {
        {out, outData},
    };

    session->prepareDevice();

    popart::NDArrayWrapper<float> inData(vInData.data(), inShape);
    std::map<popart::TensorId, popart::IArray &> inputs = {{in0, inData}};

    popart::StepIO stepio(inputs, anchors);

    session->run(stepio);

    return rawOutputData;
  };

  // no outlining, no inplacing
  auto vBase = getValue(false, false);
  // outlining, no inplacing
  auto vOutline = getValue(true, false);
  // no outlining, inplacing
  auto vInplace = getValue(false, true);
  // outlining, inplacing
  auto vAll = getValue(true, true);

  std::cout << vBase << "  == " << vOutline << " == " << vInplace
            << " == " << vAll << " ?" << std::endl;
  BOOST_CHECK(vBase - vOutline == 0.0);
  BOOST_CHECK(vOutline - vInplace == 0.0);
  BOOST_CHECK(vInplace - vAll == 0.0);
}
