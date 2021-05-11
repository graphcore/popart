// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_aapi_basic_graph
#include <boost/test/unit_test.hpp>

#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/iarray.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/op/add.hpp>
#include <popart/op/hostcopy.hpp>
#include <popart/op/init.hpp>
#include <popart/session.hpp>
#include <popart/stepio.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

#include <onnx/onnx_pb.h>

#include <memory>
#include <vector>

using namespace popart;

/*
  x = h2d_stream(...) # Stream value of 2
  w = var(0, ...)

  w += x

  c = const(5)

  y = w + c

  d2h_stream(y) # Expect value of 7
 */
BOOST_AUTO_TEST_CASE(TestBasicGraph) {
  // Will make dense tensors of this shape with the following repeated values.
  const TensorInfo tInfo{DataType::FLOAT, Shape{2, 2}};
  constexpr float xVal = 2.0f;
  constexpr float wVal = 0.0f;
  constexpr float cVal = 5.0f;

  const std::vector<float> expectedYData(tInfo.nelms(), 7.0f);

  auto ir      = std::make_unique<Ir>();
  Graph &graph = ir->getMainGraph();

  // First create the stream tensor x, then create the
  // Init -> HostLoad(x) -> xLoad. No ops should consume x, only xLoad.

  TensorId x = "x";

  graph.getTensors().addStream(x, tInfo, {"x"});

  TensorId xInit = ir->createIntermediateTensorId(x);
  TensorId xLoad = ir->createIntermediateTensorId(xInit);

  auto initXOp =
      graph.createConnectedOp<InitOp>({},
                                      {{InitOp::getOutIndex(), xInit}},
                                      Onnx::CustomOperators::Init_1,
                                      tInfo,
                                      TensorType::ActGrad,
                                      InitType::Zero,
                                      Op::Settings{graph, "xInit"});

  auto hostLoadXOp = graph.createConnectedOp<HostLoadOp>(
      {{HostLoadOp::getLocalTensorInIndex(), xInit}},
      {{HostLoadOp::getLocalTensorOutIndex(), xLoad}},
      Onnx::CustomOperators::HostLoad,
      Op::Settings{graph, "hostload_x"},
      x);

  BOOST_REQUIRE_EQUAL(hostLoadXOp->output->n(), 1);

  TensorId w = "w";
  std::vector<float> wHost(tInfo.nelms(), wVal);
  graph.getTensors().addVarInit(w, tInfo, wHost.data(), {"w"});

  BOOST_REQUIRE(graph.getTensors().get(w) != nullptr);

  TensorId wOut = ir->createIntermediateTensorId(w);

  auto accumOp = graph.createConnectedOp<AccumulateOp>(
      {{AccumulateOp::getVarToUpdateInIndex(), w},
       {AccumulateOp::getUpdaterInIndex(), xLoad}},
      {{AccumulateOp::getUpdatedVarOutIndex(), wOut}},
      AccumulationType::Add,
      OptimizerValue{}, // Presumably ignored for add.
      Op::Settings{graph, "accumIntoW"});

  BOOST_REQUIRE(accumOp->hasInput(AccumulateOp::getVarToUpdateInIndex()));

  TensorId c = "c";
  std::vector<float> cHost(tInfo.nelms(), cVal);
  graph.getTensors().addConstInit(c, tInfo, cHost.data(), {"c"});

  BOOST_REQUIRE(graph.getTensors().get(c)->hasTensorData());

  TensorId y = "y";

  auto addOp = graph.createConnectedOp<AddOp>(
      {{AddOp::getArg0InIndex(), wOut}, {AddOp::getArg1InIndex(), c}},
      {{AddOp::getOutIndex(), y}},
      Onnx::Operators::Add_7,
      Op::Settings{graph, "addY"});

  BOOST_REQUIRE(addOp->outId(AddOp::getOutIndex()) == y);

  // Must set for HostStore'd tensors, as HostStoreOpx needs this info.
  constexpr int bps = 1;
  ir->setDataFlow(DataFlow{bps, {{y, AnchorReturnType("Final")}}});

  // The original y tensor created by the user may not correspond to the tensor
  // in the (sub)graph that will actually be streamed. This maps the user's
  // tensor to the actual stream tensor.
  TensorId streamY = ir->getAnchorRemap().getRight(y);

  auto hostStoreYOp = graph.createConnectedOp<HostStoreOp>(
      {{HostStoreOp::getLocalTensorInIndex(), y}},
      {},
      Onnx::CustomOperators::HostStore,
      Op::Settings{graph, "hostStoreY"},
      streamY);

  BOOST_REQUIRE(hostStoreYOp->hasInput(HostStoreOp::getLocalTensorInIndex()));

  ///// Set Ir state required for lowering

  // Some logic in Devicex::loadEngineAndConnectStreams depends on this being
  // set: if useHostCopyOps, for each HostLoad op: get its stream tensor, then
  // get the stream created for that tensor and "connect" it as per usual.
  auto &opts          = ir->getSessionOptions();
  opts.useHostCopyOps = true;

  // Sets ScheduledPreLoss on all vertices, which determines if lowered into
  // forward or backward fragment.
  ir->updateVertices();

  // Whole bunch of Ir methods, called in various places, assume an ONNX model
  // has been set. Luckily, we can just default-construct one.
  ir->setOnnxModel({});

  // As the final step before lowering, must mark the Ir as "prepared", or the
  // lowering will immediately throw.
  ir->setIsPrepared();

  ///// Lower Ir

  const auto session = InferenceSession::createFromIr(
      std::move(ir), createTestDevice(TEST_TARGET));

  BOOST_TEST_MESSAGE("Created the Session");

  session->prepareDevice();

  BOOST_TEST_MESSAGE("Session has prepared device");

  std::vector<float> xHost(tInfo.nelms(), xVal);
  NDArrayWrapper<float> xIn(xHost.data(), tInfo);
  std::map<TensorId, IArray &> inputs = {{x, xIn}};

  std::vector<float> yHost(tInfo.nelms());
  NDArrayWrapper<float> yAnchor(yHost.data(), tInfo);
  std::map<TensorId, IArray &> anchors = {{y, yAnchor}};

  StepIO stepio(inputs, anchors);

  // Must be disabled, as these assume the Ir came from an Onnx model, and so
  // check the Ir against its Onnx model member.
  stepio.enableRuntimeAsserts(false);

  session->run(stepio);

  BOOST_TEST_MESSAGE("Session has run program.");

  // Test actual y against expectedYData.

  BOOST_REQUIRE_EQUAL(yHost.size(), expectedYData.size());

  using boost::test_tools::tolerance;

  for (std::size_t i = 0; i < yHost.size(); ++i) {
    BOOST_TEST(yHost[i] == expectedYData[i], tolerance(1e-6));
  }
}
