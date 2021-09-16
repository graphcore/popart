// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_AAPI_loop_accumulate
#include <boost/test/unit_test.hpp>

#include <popart/graph.hpp>
#include <popart/iarray.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/op/exchange/hostcopy.hpp>
#include <popart/op/loop.hpp>
#include <popart/optimizervalue.hpp>
#include <popart/session.hpp>
#include <popart/stepio.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

#include <onnx/onnx_pb.h>

#include <memory>
#include <tuple>
#include <vector>

using namespace popart;

namespace {

std::tuple<LoopOp *, Graph &> createLoopOp(Ir &ir,
                                           Graph &parentGraph,
                                           const std::string &subgraphName,
                                           const int maxTripCount);

} // namespace

/*
  int accum = 0
  for i in range(10):
    accum += 1

  ==> accum = 10
 */
BOOST_AUTO_TEST_CASE(TestLoopAccumulation) {
  constexpr int loopTripCount = 10;
  TensorInfo info{DataType::FLOAT, Shape{2, 2}};
  const std::vector<float> accumHost(info.nelms(), 0.0f);
  const std::vector<float> oneHost(info.nelms(), 1.0f);
  constexpr float expectedAccumVal = 10.0f;

  auto ir      = std::make_unique<Ir>();
  Graph &graph = ir->getMainGraph();

  ////// Create accum tensor in parent graph.

  TensorId accum    = "accum";
  TensorId accumOut = ir->createIntermediateTensorId(accum);

  graph.getTensors().addVarInit(accum, info, accumHost.data(), {"accum"});

  ////// Create loop op.

  const auto accumLG    = createLoopOp(*ir, graph, "accumLoop", 10);
  LoopOp *accumLoop     = std::get<0>(accumLG);
  Graph &accumLoopGraph = std::get<1>(accumLG);

  ////// actual body: accum += 1

  const auto sgAccum = accumLoopGraph.addScope(accum);
  accumLoop->addLoopInput(std::max(LoopOp::getFirstInputInIndex(),
                                   accumLoop->input->maxIndex() + 1),
                          accum,
                          sgAccum,
                          false);

  const auto one = accumLoopGraph.addScope(TensorId{"one"});
  accumLoopGraph.getTensors().addConstInit(one, info, oneHost.data(), {"one"});

  const TensorId sgAccumOut = ir->createIntermediateTensorId(sgAccum);

  auto accumOp = accumLoopGraph.createConnectedOp<AccumulateOp>(
      {{AccumulateOp::getVarToUpdateInIndex(), sgAccum},
       {AccumulateOp::getUpdaterInIndex(), one}},
      {{AccumulateOp::getUpdatedVarOutIndex(), sgAccumOut}},
      AccumulationType::Add,
      OptimizerValue{},
      Op::Settings{accumLoopGraph, "accum", accumLoopGraph.getScope()});

  accumOp->settings.executionContext = ExecutionContext::Normal;
  accumOp->setup();

  accumLoop->addLoopOutput(
      accumLoop->output->maxIndex() + 1, accumOut, sgAccumOut, false);

  accumLoop->setup();

  // HostStore accumOut
  // Note, instead of anchoring and HostStoring this ActGrad tensor, we could
  // alternatively use loop->addModified on sgAccum Variable, then
  // session->weightsToHost at runtime to get the final value back.

  // Must set DataFlow:
  //  - Must set bps so devicex creates stream callbacks that operate on/expect
  //    the correct amount of data. We use 1 as "batches" and "steps" do not
  //    make sense in this explicit custom program; we do not run the same
  //    program on many batches of data, we just have the one piece of data that
  //    we want to stream back to host.
  //  - Must anchor the HostStore'd tensor as the poplar streams will be created
  //    for the (remapped) anchor tensors, and HostStorex will use this stream.
  //  - TODO(T39576): Remove need to set AnchorReturnType. They implictly
  //    specify how/when to stream a tensor back to host, but we have explictly
  //    described how to do this using HostStores. Currently, we set All as this
  //    ensures the tensor is unconditionally streamed when we want it to be,
  //    and because it is the only one actually implemented.
  constexpr int bps = 1;
  ir->setDataFlow(DataFlow{bps, {{accumOut, AnchorReturnType("All")}}});

  TensorId streamAccumOut = ir->getAnchorRemap().getRight(accumOut);

  auto hostStoreYOp = graph.createConnectedOp<HostStoreOp>(
      {{HostStoreOp::getLocalTensorInIndex(), accumOut}},
      {},
      Onnx::CustomOperators::HostStore,
      Op::Settings{graph, "hostStoreAccumOut"},
      streamAccumOut);

  hostStoreYOp->setup();

  ////////////// Set Ir state required for lowering

  auto &opts                   = ir->getSessionOptions();
  opts.enableExplicitMainLoops = true;
  opts.useHostCopyOps          = true;

  // Sets ScheduledPreLoss on all vertices, which determines if lowered into
  // forward or backward fragment.
  ir->updateVertices();

  // As the final step before lowering, must mark the Ir as "prepared", or the
  // lowering will immediately throw.
  ir->setIsPrepared();

  ///// Lower Ir

  const auto session = InferenceSession::createFromIr(
      std::move(ir), createTestDevice(TEST_TARGET));

  BOOST_TEST_MESSAGE("Created the Session");

  session->prepareDevice();

  BOOST_TEST_MESSAGE("Session has prepared device");

  std::vector<float> accumOutHost(info.nelms());
  NDArrayWrapper<float> accumOutAnchor(accumOutHost.data(), info);
  std::map<TensorId, IArray &> anchors = {{accumOut, accumOutAnchor}};

  StepIO stepio({}, anchors);

  session->weightsFromHost();
  session->run(stepio);

  BOOST_TEST_MESSAGE("Session has run program.");

  // Test actual accumOut against expectedAccumOutVal.

  using boost::test_tools::tolerance;

  for (std::size_t i = 0; i < accumOutHost.size(); ++i) {
    BOOST_TEST(accumOutHost[i] == expectedAccumVal, tolerance(1e-8));
  }
}

namespace {

/**
 * @brief Create and return a LoopOp (with it's initial subgraph) in the parent
 * graph.
 *
 * @param ir The Ir this all happens in.
 * @param parentGraph The graph in which to create the LoopOp. The LoopOp's
 * subgraph will be a child graph of this graph.
 * @param loopName The name of the LoopOp. The LoopOp's subgraph will have
 * '-subgraph' appended to this as its name.
 * @param maxTripCount The max trip count to set on the LoopOp.
 * @return std::tuple<LoopOp *, Graph &> The LoopOp and its subgraph that have
 * been created.
 */
std::tuple<LoopOp *, Graph &> createLoopOp(Ir &ir,
                                           Graph &parentGraph,
                                           const std::string &loopName,
                                           const int maxTripCount) {
  const auto subgraphName = loopName + "-subgraph";

  // Create subgraph of LoopOp.

  Graph &loopSubgraph = ir.createGraph(GraphId{subgraphName});

  // Add mandatory loop iterator tensor to subgraph (is not an output)
  TensorId loopIter = loopSubgraph.addScope(reservedLoopIteratorPrefix());
  loopSubgraph.addInput(loopIter, TensorInfo{DataType::INT32, {}});

  // Add mandatory loop condition tensor to subgraph (is also an output)
  TensorId loopCond = loopSubgraph.addScope(reservedLoopCondPrefix());
  loopSubgraph.addInput(loopCond, TensorInfo{DataType::BOOL, {}});
  loopSubgraph.markAsOutput(loopCond);

  // Create LoopOp in parent.

  Op::Settings loopSettings(parentGraph, loopName);
  loopSettings.executionContext = ExecutionContext::Normal;

  auto loop = parentGraph.createOp<LoopOp>(
      Onnx::Operators::Loop_11, loopSettings, loopSubgraph);

  loop->setTripCountValue(maxTripCount);

  return std::forward_as_tuple(loop, loopSubgraph);
}

} // namespace
