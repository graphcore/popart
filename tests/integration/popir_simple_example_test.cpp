// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PopirSimpleExampleTest
// The below test is here to prevent a regression in the popart snap
// integration. It is very probable that this will fail at some point due to a
// required API not being implemented in snap yet. If this happens, please
// contact a member of the popir team, and we will be able to prioritise adding
// the required API to snap. If landing your diff is time sensitive, we can
// always disable this test temporarily.

#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <memory>
#include <snap/Graph.hpp>
#include <snap/Tensor.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include <poplar/Target.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/session.hpp>
#include <popart/sgd.hpp>
#include <popart/testdevice.hpp>

#include "popart/builder.gen.hpp"
#include "popart/names.hpp"
#include "popart/tensorinfo.hpp"

namespace poplar {
class Graph;
class Tensor;
} // namespace poplar

class GetPoplarTensorError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class GetConstPoplarTensorError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class GetPoplarGraphError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class GetConstPoplarGraphError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

const char *getPoplarTensorError =
    "There should be no calls to snap::Tensor::getPoplarTensor in the popir "
    "simple example. Please try and use the snap API if possible. If there is "
    "no snap API for the functions you require, please contact a member of the "
    "popir team, and we can prioritise getting the calls you require into the "
    "snap API.";

const char *getPoplarGraphError =
    "There should be no calls to snap::Graph::getPoplarGraph in the popir "
    "simple example. Please try and use the snap API if possible. If there is "
    "no snap API for the functions you require, please contact a member of the "
    "popir team, and we can prioritise getting the calls you require into the "
    "snap API.";

namespace snap {

// This call will override the snap call, allowing us to error and fail the test
// if it is used.
poplar::Tensor &Tensor::getPoplarTensor() {
  throw GetPoplarTensorError(getPoplarTensorError);
}

// This call will override the snap call, allowing us to error and fail the test
// if it is used.
const poplar::Tensor &Tensor::getPoplarTensor() const {
  throw GetConstPoplarTensorError(getPoplarTensorError);
}

// This call will override the snap call, allowing us to error and fail the test
// if it is used.
poplar::Graph &Graph::getPoplarGraph() {
  throw GetPoplarGraphError(getPoplarGraphError);
}

// This call will override the snap call, allowing us to error and fail the test
// if it is used.
const poplar::Graph &Graph::getPoplarGraph() const {
  throw GetConstPoplarGraphError(getPoplarGraphError);
}

} // namespace snap

BOOST_AUTO_TEST_CASE(PopirSimpleExampleTest) {
  // Check that snap::Tensor::getPoplarTensor and snap::Graph::getPoplarGraph
  // have been successfully overridden
  {
    snap::Tensor snapTensor;
    BOOST_CHECK_THROW(snapTensor.getPoplarTensor(), GetPoplarTensorError);
    const snap::Tensor constSnapTensor;
    BOOST_CHECK_THROW(constSnapTensor.getPoplarTensor(),
                      GetConstPoplarTensorError);

    auto target = poplar::Target::createCPUTarget();
    snap::Graph snapGraph(target);
    BOOST_CHECK_THROW(snapGraph.getPoplarGraph(), GetPoplarGraphError);
    const snap::Graph constSnapGraph(target);
    BOOST_CHECK_THROW(constSnapGraph.getPoplarGraph(),
                      GetConstPoplarGraphError);
  }

  // Run the simple example. This just has to run without throwing an error. If
  // a call to snap::Tensor::getPoplarTensor or snap::Graph::getPoplarGraph is
  // made in this example, an error will be thrown and the test will fail.
  using namespace popart;
  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();

  int dataSize = 32;
  TensorInfo info{"FLOAT16", std::vector<int64_t>{dataSize, dataSize}};
  auto t1 = builder->addInputTensor(info);
  auto t2 = builder->addInputTensor(info);
  auto t3 = builder->addInputTensor(info);

  auto m1 = aiOnnx.matmul({t1, t2}, "mul_1");
  auto m2 = aiOnnx.matmul({m1, t3}, "mul_2");

  auto o = aiOnnx.add({m1, m2});

  builder->addOutputTensor(o);

  auto loss = aiGraphcore.identityloss({o});

  auto dataFlow  = DataFlow(1, {{o, AnchorReturnType("All")}});
  auto optimizer = ConstSGD(0.01);
  auto device    = createTestDevice(TEST_TARGET, 1);

  auto session = TrainingSession::createFromOnnxModel(
      builder->getModelProto(), dataFlow, loss, optimizer, device);
  session->prepareDevice();
}
