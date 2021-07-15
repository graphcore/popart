// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE NumericsInplaceVsNot0Test

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
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

BOOST_AUTO_TEST_CASE(Inplace_numericsIpNip0) {

  // A 2x2 Tensor of zeros, denoted as
  //
  // . .
  // . .
  //
  // gets padded on different sides, then the padded
  // Tensors are added together:
  //
  // The padding constants are 1, 2, 4, 8:
  //
  //  . . 1       4 4 4       2 . .       8 8 8
  //  . . 1       . . 4       2 . .       8 . .
  //  1 1 1       . . 4       2 2 2       8 . .
  //    \          /             \          /.
  //      \      /                 \      /.
  //        \  /                     \  /.
  //        add                      add
  //            .                 .
  //                 .       .
  //                    add
  //
  //
  //
  // Expected output :
  //
  // 14 10 11
  // 12 .  3
  // 13 5  7
  //
  // Tested with inplace on, inplacing off, and pad inplace taking higher or
  // lower priority than usual.
  //
  auto test = [](bool withInplace, bool padInplaceHighPriority) {
    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();

    std::vector<int64_t> inShape{{2, 2}};
    auto nInElms = 2 * 2;
    TensorInfo inInfo("FLOAT", inShape);

    std::vector<int64_t> outShape{{3, 3}};
    assert(outShape.size() == 2);
    auto nOutElms = 3 * 3;
    TensorInfo outInfo("FLOAT", outShape);

    auto inId = builder->addInputTensor(inInfo);

    // add minimal padding in each dimension, in all 4 possible directions
    auto pad0 = aiOnnx.pad({inId}, {1, 1, 0, 0}, "constant", 8.f);
    auto pad1 = aiOnnx.pad({inId}, {1, 0, 0, 1}, "constant", 2.f);
    auto pad2 = aiOnnx.pad({inId}, {0, 1, 1, 0}, "constant", 4.f);
    auto pad3 = aiOnnx.pad({inId}, {0, 0, 1, 1}, "constant", 1.f);

    float padPriority = padInplaceHighPriority ? 1000.f : 0.1;
    for (auto padId : {pad0, pad1, pad2, pad3}) {
      builder->setInplacePreferences(padId, {{"PadInplace", padPriority}});
    }

    auto add0   = aiOnnx.add({pad0, pad1});
    auto add1   = aiOnnx.add({pad2, pad3});
    auto summed = aiOnnx.add({add0, add1});

    builder->addOutputTensor(summed);

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    // Create the IR, adding outId as an anchor
    auto art      = AnchorReturnType("All");
    auto dataFlow = DataFlow(1, {{summed, art}});

    auto opts   = SessionOptions();
    auto device = popart::createTestDevice(TEST_TARGET);

    auto session = popart::InferenceSession::createFromOnnxModel(
        proto,
        dataFlow,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Minimal).enableInPlace(withInplace));

    // prepare the anchors
    std::vector<float> outputData(nOutElms, -777.);
    auto outputDataPtr = outputData.data();
    popart::NDArrayWrapper<float> outData(outputDataPtr, outShape);

    std::map<popart::TensorId, popart::IArray &> anchors = {
        {summed, outData},
    };

    session->prepareDevice();

    std::vector<float> vInData(nInElms, 0.0f);

    popart::NDArrayWrapper<float> inData(vInData.data(), inShape);
    std::map<popart::TensorId, popart::IArray &> inputs = {{inId, inData}};

    popart::StepIO stepio(inputs, anchors);

    session->run(stepio);

    std::vector<float> expected{14, 10, 11, 12, 0, 3, 13, 5, 7};

    for (uint64_t i = 0; i < nOutElms; ++i) {
      if (outputData[i] != expected[i]) {
        std::ostringstream oss;
        oss << "Failure in pad_outplace_0 test. "
            << "At index i = " << i
            << ", the expected value of the jitter-pad-add output is "
            << expected[i] << ", but the observed value was " << outputData[i]
            << '.' << " In this test, inplace = " << withInplace
            << ", and padInplaceHighPriority = " << padInplaceHighPriority;
        throw error(oss.str());
      }
    }
  };

  //  (inplace, pad-high-priority)
  test(true, true);
  test(true, false);
  test(false, true);

  // Note that the Ir does not take into account that Pad outputs cannot be
  // modified, as they contain constants. This is opaque to the Ir, and is
  // resolved at the last moment when the codelet is generated for the IPU,
  // where the Op calls poplar::Tensor::isModifiable().
  //
  // In the future, we will hopefully have more accurate information available
  // in the Ir w.r.t. constants and self-aliasing, and we will be able to
  // resolve these cases earlier and thereby have more effective inplacing.
}
