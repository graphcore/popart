#define BOOST_TEST_MODULE Basic0SessionApiTest

#include <boost/test/unit_test.hpp>
#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
#include <poponnx/devicemanager.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/inputshapeinfo.hpp>
#include <poponnx/ndarraywrapper.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/session.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/tensornames.hpp>

#include <algorithm>
#include <map>
#include <random>
#include <tuple>
#include <vector>

bool prePrepCallError(const poponnx::error &ex) {
  std::string what = ex.what();
  BOOST_CHECK(what.find("be called before") != std::string::npos);
  return true;
}

BOOST_AUTO_TEST_CASE(Basic0SessionApi) {
  using namespace poponnx;

  auto opts = SessionOptions();

  auto builder = Builder::create();
  TensorInfo xInfo{"FLOAT", std::vector<int64_t>{1, 2, 3, 4, 5}};
  TensorId xId = builder->addInputTensor(xInfo);
  auto aiOnnx  = builder->aiOnnxOpset9();
  TensorId yId = aiOnnx.relu({xId});
  builder->addOutputTensor(xId);

  auto proto = builder->getModelProto();

  auto art      = AnchorReturnType("ALL");
  auto dataFlow = DataFlow(1, {{yId, art}});
  auto cpuDevice =
      poponnx::DeviceManager::createDeviceManager().createCpuDevice();

  auto session = poponnx::InferenceSession::createFromOnnxModel(
      proto,
      dataFlow,
      cpuDevice,
      {},
      poponnx::InputShapeInfo(),
      opts,
      poponnx::Patterns(PatternsLevel::NONE));

  // create anchor and co.
  std::vector<float> rawOutputValues(xInfo.nelms());
  poponnx::NDArrayWrapper<float> outValues(rawOutputValues.data(), xInfo);
  std::map<poponnx::TensorId, poponnx::IArray &> anchors = {{yId, outValues}};

  // create input and co.
  std::vector<float> vXData(xInfo.nelms());
  poponnx::NDArrayWrapper<float> xData(vXData.data(), xInfo);
  std::map<poponnx::TensorId, poponnx::IArray &> inputs = {{xId, xData}};

  poponnx::StepIO stepio(inputs, anchors);
  BOOST_CHECK_EXCEPTION(session->run(stepio), poponnx::error, prePrepCallError);
}
