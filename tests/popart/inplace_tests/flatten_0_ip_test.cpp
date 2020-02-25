#define BOOST_TEST_MODULE Flatten0InplaceTest

#include <boost/test/unit_test.hpp>
#include <complex>
#include <random>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

#define protected public
#include <popart/session.hpp>
#undef protected

using namespace popart;

BOOST_AUTO_TEST_CASE(Inplace_flatten0) {

  auto test = [](float flattenInplacePriority) {
    //        in of shape {2,3,2,6}
    //            /  \
    //           /    \
    //     flatten-1    scale-3.0 (s1)
    //        |         |
    // (s0) scale-2.0  flatten-2
    //         \       /
    //          \     /
    //     matmul (6x24 "dot" 24x6)
    //             |
    //
    //  where flatten above is null flattending

    // Build an onnx model
    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();

    TensorInfo info0{"FLOAT", std::vector<int64_t>{2, 3, 2, 6}};
    TensorInfo infoOut{"FLOAT", std::vector<int64_t>{6, 6}};
    int M = info0.shape()[0] * info0.shape()[1];
    int N = info0.shape()[3]; // = M
    int K = info0.nelms() / M;

    auto in0 = builder->addInputTensor(info0);

    auto flat0         = aiOnnx.flatten({in0}, 2);
    float scaleFactor0 = 2.0;
    auto s0            = aiGraphcore.scale({flat0}, scaleFactor0);

    float scaleFactor1 = 3.0;
    auto s1            = aiGraphcore.scale({in0}, scaleFactor1);
    auto flat1         = aiOnnx.flatten({s1}, 3);

    auto dotOut = aiOnnx.matmul({s0, flat1});
    builder->addOutputTensor(dotOut);

    float s0priority = 100.0f;
    float s1priority = 444.0f;
    builder->setInplacePreferences(s0, {{"ScaleInplace", s0priority}});
    builder->setInplacePreferences(s1, {{"ScaleInplace", s1priority}});

    builder->setInplacePreferences(
        flat0, {{"FlattenInplace", flattenInplacePriority}});
    builder->setInplacePreferences(
        flat1, {{"FlattenInplace", flattenInplacePriority}});

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    // Create the IR
    auto dataFlow = DataFlow(1, {{dotOut, AnchorReturnType("ALL")}});
    auto device   = createTestDevice(TEST_TARGET);

    auto opts            = SessionOptions();
    opts.enableOutlining = false;

    auto session = popart::InferenceSession::createFromOnnxModel(
        proto,
        dataFlow,
        device,
        {},
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::NONE).enableInPlace(true));

    // generate random input data
    int seed = 1011;
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<float> fdis(0, 5);
    std::uniform_int_distribution<uint64_t> idis(
        0, std::numeric_limits<uint64_t>::max());

    std::vector<float> vdata0(info0.nelms());
    for (auto &val : vdata0) {
      val = fdis(eng);
    }

    // generate expected output
    std::vector<float> expectedOut(infoOut.nelms());
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        expectedOut[m * N + n] = 0;
        for (int k = 0; k < K; ++k) {
          expectedOut[m * N + n] += vdata0[m * K + k] * vdata0[n + k * N];
        }
        // multiply by the scale factors
        expectedOut[m * N + n] *= scaleFactor0;
        expectedOut[m * N + n] *= scaleFactor1;
      }
    }

    popart::NDArrayWrapper<float> data0(vdata0.data(), info0);
    std::map<popart::TensorId, popart::IArray &> inputs = {{in0, data0}};

    std::vector<float> rawOut(infoOut.nelms());

    popart::NDArrayWrapper<float> outValues(rawOut.data(), infoOut);
    std::map<popart::TensorId, popart::IArray &> anchors = {
        {dotOut, outValues}};

    popart::StepIO stepio(inputs, anchors);

    session->prepareDevice();
    session->run(stepio);
    if (flattenInplacePriority < 0) {
      BOOST_CHECK(session->ir.opsOfType(Onnx::AiOnnx::OpSet9::Flatten).size() ==
                  2);
    } else if (flattenInplacePriority > std::max(s0priority, s1priority)) {
      BOOST_CHECK(session->ir.opsOfType(Onnx::AiOnnx::OpSet9::Flatten).size() ==
                  0);
    } else {
      BOOST_CHECK(session->ir.opsOfType(Onnx::AiOnnx::OpSet9::Flatten).size() ==
                  1);
    }

    float absL1diff = 0;
    for (int i = 0; i < rawOut.size(); ++i) {
      absL1diff += std::abs<float>(rawOut[i] - expectedOut[i]);
    }
    std::cout << "meanL1diff = " << absL1diff / rawOut.size() << std::endl;
    BOOST_CHECK(absL1diff / rawOut.size() < 1e-4);
  };

  std::cout << "Testing, flatten inplace NEGATIVE priority" << std::endl;
  test(-10000.0f);
  std::cout << "Testing, flatten inplace HIGH priority" << std::endl;
  test(100000.0f);
  std::cout << "Testing, flatten inplace LOW priority" << std::endl;
  test(0.1f);
}
