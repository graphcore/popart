// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Basic0SessionApiTest

#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <testdevice.hpp>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>

#include "popart/builder.gen.hpp"
#include "popart/error.hpp"
#include "popart/names.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/stepio.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {
class IArray;
} // namespace popart

bool prePrepCallError(const popart::error &ex) {
  std::string what = ex.what();
  BOOST_CHECK(what.find("be called before") != std::string::npos);
  return true;
}

BOOST_AUTO_TEST_CASE(Basic0SessionApi) {
  using namespace popart;

  auto opts = SessionOptions();

  auto builder = Builder::create();
  TensorInfo xInfo{"FLOAT", std::vector<int64_t>{1, 2, 3, 4, 5}};
  TensorId xId = builder->addInputTensor(xInfo);
  auto aiOnnx  = builder->aiOnnxOpset9();
  TensorId yId = aiOnnx.relu({xId});
  builder->addOutputTensor(xId);

  auto proto = builder->getModelProto();

  auto art      = AnchorReturnType("All");
  auto dataFlow = DataFlow(1, {{yId, art}});
  auto device   = popart::createTestDevice(TEST_TARGET);

  auto session = popart::InferenceSession::createFromOnnxModel(
      proto,
      dataFlow,
      device,
      popart::InputShapeInfo(),
      opts,
      popart::Patterns(PatternsLevel::NoPatterns).enableRuntimeAsserts(false));

  // create anchor and co.
  std::vector<float> rawOutputValues(xInfo.nelms());
  popart::NDArrayWrapper<float> outValues(rawOutputValues.data(), xInfo);
  std::map<popart::TensorId, popart::IArray &> anchors = {{yId, outValues}};

  // create input and co.
  std::vector<float> vXData(xInfo.nelms());
  popart::NDArrayWrapper<float> xData(vXData.data(), xInfo);
  std::map<popart::TensorId, popart::IArray &> inputs = {{xId, xData}};

  popart::StepIO stepio(inputs, anchors);
  BOOST_CHECK_EXCEPTION(session->run(stepio), popart::error, prePrepCallError);
}
