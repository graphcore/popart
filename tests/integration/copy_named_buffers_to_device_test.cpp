// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE copy_named_buffers_to_device_test

#include <boost/test/unit_test.hpp>

#include <filereader.hpp>
#include <string>
#include <testdevice.hpp>
#include <vector>
#include <popart/builder.gen.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/executablex.hpp>
#include <popart/session.hpp>
#include <popart/sgd.hpp>
#include <popart/stepio.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>

using namespace popart;

namespace {
// Create buffers b1, b2 and input data, each of same shape and initialized
// to ones. Run session once, at this point output is equal to 3*3.
// Then update buffer(s) specified by input to the value of two, rerun
// session and return the output.
std::vector<float> test_copy_buffers(std::vector<std::string> buffers) {
  const int N  = 3;
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo info{"FLOAT", std::vector<int64_t>{N}};
  std::vector<float> bufferInit(N);
  std::fill(bufferInit.begin(), bufferInit.end(), 1);

  TensorId b1Id =
      builder->addInitializedInputTensor({bufferInit.data(), info}, "b1");
  TensorId b2Id =
      builder->addInitializedInputTensor({bufferInit.data(), info}, "b2");
  auto dataId = builder->addInputTensor(info, "data");
  std::vector<float> dataInit(N);
  std::fill(dataInit.begin(), dataInit.end(), 1);
  popart::NDArrayWrapper<float> dataWrapper(dataInit.data(), info);
  std::map<popart::TensorId, popart::IArray &> inputs = {{dataId, dataWrapper}};

  auto out_  = aiOnnx.sum({b1Id, b2Id, dataId});
  auto outId = aiOnnx.reducesum({out_});

  std::vector<float> out(1);
  TensorInfo outInfo{"FLOAT", std::vector<int64_t>{1}};
  popart::NDArrayWrapper<float> outWrapper(out.data(), outInfo);

  std::map<popart::TensorId, popart::IArray &> anchors = {{outId, outWrapper}};

  popart::StepIO stepio(inputs, anchors);

  auto art       = AnchorReturnType("All");
  auto dataFlow  = DataFlow(1, {{outId, art}});
  auto opts      = SessionOptions();
  auto lossId    = builder->aiGraphcoreOpset1().l1loss({outId}, 0.0);
  auto optimizer = ConstSGD(0.001);

  for (const auto &buffer : buffers)
    opts.updatableNamedBuffers.push_back(buffer);

  auto device     = popart::createTestDevice(TEST_TARGET);
  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  auto session = popart::TrainingSession::createFromOnnxModel(
      proto,
      dataFlow,
      lossId,
      optimizer,
      device,
      popart::InputShapeInfo(),
      opts,
      popart::Patterns(PatternsLevel::Default));

  session->prepareDevice();

  session->weightsFromHost();
  session->run(stepio);

  // Sanity check. If it fails then there is a wider issue.
  BOOST_CHECK_EQUAL(out[0], 3 * 3);

  std::vector<float> bUpdate(bufferInit.size());
  std::fill(bUpdate.begin(), bUpdate.end(), 2);

  Tensor *b1 = const_cast<Tensor *>(session->getExecutable().getTensor(b1Id));
  Tensor *b2 = const_cast<Tensor *>(session->getExecutable().getTensor(b2Id));

  for (const auto &b : buffers) {
    if (b == "b1") {
      b1->tensorData()->resetData(b1->info, bUpdate.data());
    }
    if (b == "b2") {
      b2->tensorData()->resetData(b2->info, bUpdate.data());
    }
  }

  session->buffersFromHost();
  session->run(stepio);

  return out;
}
} // namespace

BOOST_AUTO_TEST_CASE(test_copy_named_buffer_to_device_single_buffer) {
  std::vector<std::vector<std::string>> buffers;
  buffers.push_back(std::vector<std::string>{"b1"});
  buffers.push_back(std::vector<std::string>{"b2"});

  for (const auto &buffer : buffers) {
    auto out = test_copy_buffers(buffer);
    BOOST_CHECK_EQUAL(out[0], 3 * 4);
  }
}

BOOST_AUTO_TEST_CASE(test_copy_named_buffer_to_device_two_buffers) {
  std::vector<std::string> buffers = {"b1", "b2"};

  auto out = test_copy_buffers(buffers);
  BOOST_CHECK_EQUAL(out[0], 3 * 5);
}

BOOST_AUTO_TEST_CASE(test_copy_named_buffer_to_device_no_buffers) {
  std::vector<std::string> buffers = {};

  BOOST_CHECK_THROW(test_copy_buffers(buffers), popart::error);
}

BOOST_AUTO_TEST_CASE(test_copy_named_buffer_to_device_invalid_buffer) {
  std::vector<std::string> buffers = {"b3"};

  BOOST_CHECK_THROW(test_copy_buffers(buffers), popart::error);
}
