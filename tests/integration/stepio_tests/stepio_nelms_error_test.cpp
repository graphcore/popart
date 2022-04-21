// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE NumElmsIOErrorTest

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
#include <popart/session.hpp>
#include <popart/tensordata.hpp>
#include <popart/testdevice.hpp>

using namespace popart;

bool invalidNelms(const error &ex) {
  std::string whatMessage = ex.what();
  BOOST_CHECK(whatMessage.find("Unexpected number") != std::string::npos);
  return true;
}

BOOST_AUTO_TEST_CASE(CorrectBufferNElmsTest0) {

  Shape inShape{1};
  TensorInfo inInfo{"FLOAT", inShape};

  // Build an onnx model, general boilerplate:
  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();
  auto in0         = builder->addInputTensor(inInfo);
  auto s1          = aiGraphcore.scale({in0}, 2.0);
  auto out         = aiOnnx.reducesum({s1}, std::vector<int64_t>{0}, false);
  builder->addOutputTensor(out);
  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("All");
  auto dataFlow   = DataFlow(1, {{out, art}});
  auto opts       = SessionOptions();
  auto device     = popart::createTestDevice(TEST_TARGET);
  auto session    = popart::InferenceSession::createFromOnnxModel(
      proto,
      dataFlow,
      device,
      popart::InputShapeInfo(),
      opts,
      popart::Patterns(PatternsLevel::NoPatterns).enableRuntimeAsserts(false));
  float rawOutputData;
  Shape outShape{};
  popart::NDArrayWrapper<float> outData(&rawOutputData, outShape);
  std::map<popart::TensorId, popart::IArray &> anchors = {
      {out, outData},
  };
  session->prepareDevice();

  auto incorrectInShape = inShape;
  incorrectInShape.push_back(2);
  std::vector<float> vIncorrectInData{1.0, 2.0};
  popart::NDArrayWrapper<float> incorrectInData(vIncorrectInData.data(),
                                                incorrectInShape);
  popart::StepIO incorrectStepio({{in0, incorrectInData}}, anchors);

  // if the first call to run is with a StepIO which does not have the correct
  // number of elements for at least one Tensor, an error is thrown from popart
  BOOST_CHECK_EXCEPTION(session->run(incorrectStepio), error, invalidNelms);

  std::vector<float> vCorrectInData{3.0};
  popart::NDArrayWrapper<float> correctInData(vCorrectInData.data(), inShape);
  popart::StepIO correctStepio({{in0, correctInData}}, anchors);

  // If the first call to run is correct, then any subsequent calls are not
  // checked, 1) call with correct stepio
  session->run(correctStepio);
  // 2) call with incorrect stepio - no tests so this will produce unexpected
  // results. The reason there is no check on this call is for performance.
  session->run(incorrectStepio);
}
