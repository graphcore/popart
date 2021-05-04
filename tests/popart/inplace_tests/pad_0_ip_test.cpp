// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Pad0InplaceTest

#include <boost/test/unit_test.hpp>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/session.hpp>
#include <popart/stepio.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(Inplace_pad0) {

  auto test = [](float padInplacePriority) {
    //           in [1,1]
    //          /  \
    //         /    \
    //       pad    scale-3.0 (s1)
    //        |         |
    // (s0) scale-2.0  pad
    //         \       /
    //          \     /
    //            add   [5,5]
    //             |
    //
    //  where pad above is null padding

    // Build an onnx model
    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();

    // input and output:
    int64_t nElmsIn   = 2;
    int64_t nPadLeft  = 3;
    int64_t nPadRight = 6;
    TensorInfo info0{"FLOAT", std::vector<int64_t>{nElmsIn}};
    TensorInfo infoOut{"FLOAT",
                       std::vector<int64_t>{nElmsIn + nPadLeft + nPadRight}};

    auto in0 = builder->addInputTensor(info0);

    auto p0 = aiOnnx.pad({in0}, {nPadLeft, nPadRight});
    auto s0 = aiGraphcore.scale({p0}, 2.0f);

    auto s1 = aiGraphcore.scale({in0}, 3.0f);
    auto p1 = aiOnnx.pad({s1}, {nPadLeft, nPadRight});

    auto sum = aiOnnx.add({p1, s0});
    builder->addOutputTensor(sum);

    float s0priority = 100.0f;
    float s1priority = 444.0f;
    builder->setInplacePreferences(s0, {{"ScaleInplace", s0priority}});
    builder->setInplacePreferences(s1, {{"ScaleInplace", s1priority}});

    builder->setInplacePreferences(p0, {{"PadInplace", padInplacePriority}});
    builder->setInplacePreferences(p1, {{"PadInplace", padInplacePriority}});

    // don't inplace the sum
    builder->setInplacePreferences(
        sum, {{"AddLhsInplace", -10.0f}, {"AddRhsInplace", -10.0f}});

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    // Create the IR
    auto dataFlow = DataFlow(1, {{sum, AnchorReturnType("All")}});
    auto device   = createTestDevice(TEST_TARGET);

    auto opts = SessionOptions();

    auto session = popart::InferenceSession::createFromOnnxModel(
        proto,
        dataFlow,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::NoPatterns)
            .enableRuntimeAsserts(false)
            .enableInPlace(true));

    std::vector<float> vdata0{1.0, 1.0};
    popart::NDArrayWrapper<float> data0(vdata0.data(), info0);
    std::map<popart::TensorId, popart::IArray &> inputs = {{in0, data0}};

    std::vector<float> rawOut(infoOut.nelms());
    popart::NDArrayWrapper<float> outValues(rawOut.data(), infoOut);
    std::map<popart::TensorId, popart::IArray &> anchors = {{sum, outValues}};

    // session->prepareDevice();
    popart::StepIO stepio(inputs, anchors);

    session->prepareDevice();

    session->run(stepio);
    std::vector<float> padLeft(nPadLeft, 0);
    std::vector<float> padRight(nPadRight, 0);
    std::vector<float> expectedInternal{5, 5};
    std::vector<float> expectedOut = padLeft;
    expectedOut.insert(
        expectedOut.end(), expectedInternal.begin(), expectedInternal.end());
    expectedOut.insert(expectedOut.end(), padRight.begin(), padRight.end());

    // if the Pad is incorrectly inplace when it claims to to out of place,
    // this is [12, 12]

    if (padInplacePriority < 0) {
      BOOST_CHECK(
          session->getIr().opsOfType(Onnx::AiOnnx::OpSet9::Pad).size() == 2);
    } else if (padInplacePriority > std::max(s0priority, s1priority)) {
      BOOST_CHECK(
          session->getIr().opsOfType(Onnx::AiOnnx::OpSet9::Pad).size() == 0);
    } else {
      BOOST_CHECK(
          session->getIr().opsOfType(Onnx::AiOnnx::OpSet9::Pad).size() == 1);
    }

    BOOST_CHECK_EQUAL_COLLECTIONS(
        rawOut.begin(), rawOut.end(), expectedOut.begin(), expectedOut.end());
  };
  std::cout << "Testing, pad inplace HIGH priority" << std::endl;
  test(100000.0f);
  std::cout << "Testing, pad inplace LOW priority" << std::endl;
  test(0.1f);
  std::cout << "Testing, pad inplace NEGATIVE priority" << std::endl;
  test(-10000.0f);
}
