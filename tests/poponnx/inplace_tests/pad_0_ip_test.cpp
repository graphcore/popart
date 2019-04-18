#define BOOST_TEST_MODULE Pad0InplaceTest

#include <boost/test/unit_test.hpp>
#include <vector>
#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
#include <poponnx/devicemanager.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/inputshapeinfo.hpp>
#include <poponnx/ndarraywrapper.hpp>
#include <poponnx/session.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/tensornames.hpp>
#include <poponnx/tensors.hpp>

using namespace poponnx;

BOOST_AUTO_TEST_CASE(Inplace_concat0) {

  //           in [1,1]
  //          /  \
  //         /    \
  //       pad    scale-3
  //        |       |
  //       scale-2 pad
  //         \     /
  //           add   [5,5]
  //            |
  //
  //  where pad above is null padding

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  // input and output:
  TensorInfo info0{"FLOAT", std::vector<int64_t>{2}};
  auto in0 = builder->addInputTensor(info0);

  auto p0 = aiOnnx.pad({in0}, {0, 0});
  auto s0 = aiOnnx.scale({p0}, 2.0f);

  auto s1 = aiOnnx.scale({in0}, 3.0f);
  auto p1 = aiOnnx.pad({s1}, {0, 0});

  auto sum = aiOnnx.add({p1, s0});
  builder->addOutputTensor(sum);

  builder->setInplacePreferences(s0, {{"ScaleInplace", 100.0f}});

  builder->setInplacePreferences(s1, {{"ScaleInplace", 444.0f}});

  builder->setInplacePreferences(
      sum, {{"AddLhsInplace", -10.0f}, {"AddRhsInplace", -10.0f}});

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow  = DataFlow(1, {{sum, AnchorReturnType("ALL")}});
  auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

  auto opts = SessionOptions();

  auto session = poponnx::InferenceSession::createFromOnnxModel(
      proto,
      dataFlow,
      cpuDevice,
      {},
      poponnx::InputShapeInfo(),
      opts,
      poponnx::Patterns(PatternsLevel::NONE).enableInPlace(true));

  std::vector<float> vdata0{1.0, 1.0};
  poponnx::NDArrayWrapper<float> data0(vdata0.data(), info0);
  std::map<poponnx::TensorId, poponnx::IArray &> inputs = {{in0, data0}};

  std::vector<float> rawOut(info0.nelms());
  poponnx::NDArrayWrapper<float> outValues(rawOut.data(), info0);
  std::map<poponnx::TensorId, poponnx::IArray &> anchors = {{sum, outValues}};

  // session->prepareDevice();
  poponnx::StepIO stepio(inputs, anchors);

  session->prepareDevice();

  session->run(stepio);
  std::vector<float> expectedOut{5, 5};
  // if the Pad is incorrectly inplace when it claims to to out of place,
  // this is [12, 12]

  BOOST_CHECK_EQUAL_COLLECTIONS(
      rawOut.begin(), rawOut.end(), expectedOut.begin(), expectedOut.end());
}
