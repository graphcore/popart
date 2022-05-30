// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE popXLFunctionBuffers

#include <algorithm>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/test/unit_test.hpp>
#include <cassert>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/call.hpp>
#include <popart/op/exchange/codecopy.hpp>
#include <popart/op/exchange/hostcopy.hpp>
#include <popart/op/init.hpp>
#include <popart/op/matmul.hpp>
#include <popart/session.hpp>
#include <popart/stepio.hpp>
#include <popart/testdevice.hpp>
#include <popart/topocons.hpp>
#include <popart/util.hpp>
#include <popart/vendored/optional.hpp>

#include "popart/bimap.hpp"
#include "popart/dataflow.hpp"
#include "popart/datatype.hpp"
#include "popart/graphid.hpp"
#include "popart/op.hpp"
#include "popart/op/relu.hpp"
#include "popart/operators.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensorlocation.hpp"
#include "popart/tensors.hpp"
#include <pva/pva.hpp>

namespace popart {
class IArray;
} // namespace popart

using namespace popart;

const int nLayers = 3;

/**
 * As per the functionbuffers test but with another nested graph inside. Each is
called nLayers = 3 times to avoid inlining the code:

         Main graph

     w             act
     v              |
GraphOne-x3----------+-----+
|    |              |     |
|    |              v     |
|    +----------> MatMul  |
|    |              |     |
|    |              v     |
|    |     GraphTwo-x3----+
|    |     |        |     |
|    |     |        v     |
|    +-----+----->MatMul  |
|          |        |     |
|          |        v     |
+----------+--------+-----+
                    |
                   Relu
                    v
                   ...

when codeCopy == true, both graphs have their code loaded to them before
executing.
 */
std::pair<std::vector<float>, uint64_t> runTestCase(bool codeCopy) {

  // Construct IR and main graph.
  auto ir  = std::make_unique<Ir>();
  Graph &g = ir->getMainGraph();

  // Parameters.

  const int r = 5; // Number of matrix rows.
  const int c = r; // Number of matrix cols (same as rows).
  const TensorInfo tInfo(DataType::FLOAT, Shape{r, c});

  // Initialise a normal distribution with zero mean and unit variance.
  std::default_random_engine gen(42);
  std::normal_distribution<> d{0, 1};
  // Create a function to sample from the distribution.
  auto sampleFromNormal = [&]() { return d(gen); };

  // Generate random weights on the host.
  std::vector<std::vector<float>> weightsHost = {};
  for (int i = 0; i < nLayers; i++) {
    std::vector<float> weight(tInfo.nelms());
    std::generate(weight.begin(), weight.end(), sampleFromNormal);
    weightsHost.push_back(weight);
  }

  Graph &graphOne       = ir->createGraph({"graphOne"});
  TensorId wInCallOne   = addScope(graphOne, "weight(mmLhs)");
  TensorId actInCallOne = addScope(graphOne, "act(mmRhs)");

  graphOne.addInput(wInCallOne, tInfo);
  graphOne.addInput(actInCallOne, tInfo);

  Graph &graphTwo       = ir->createGraph({"graphTwo"});
  TensorId wInCallTwo   = addScope(graphTwo, "weight(mmLhs)");
  TensorId actInCallTwo = addScope(graphTwo, "act(mmRhs)");
  TensorId outCallTwo   = addScope(graphTwo, "out");

  graphTwo.addInput(actInCallTwo, tInfo);
  graphTwo.addInput(wInCallTwo, tInfo);

  graphTwo.createConnectedOp<MatMulOp>(
      {{MatMulOp::getLhsInIndex(), wInCallTwo},
       {MatMulOp::getRhsInIndex(), actInCallTwo}},
      {{MatMulOp::getOutIndex(), outCallTwo}},
      Onnx::Operators::MatMul_9,
      Op::Settings{graphTwo, "MatMul"},
      nonstd::nullopt,
      MatMulOp::SerialiseSettings(),
      OptionalDataType());

  graphTwo.markAsOutput(outCallTwo);

  // Create a stream for data input from host.
  TensorId dataStream = "D_stream";
  g.getTensors().addStream(dataStream, tInfo);

  // Init data tensor. This Op is required for HostLoadOp - see hostcopy.hpp.
  TensorId dataPrehostload = "D_prehostload";
  g.createConnectedOp<InitOp>({},
                              {{InitOp::getOutIndex(), dataPrehostload}},
                              Onnx::CustomOperators::Init_1,
                              tInfo,
                              TensorType::ActGrad,
                              InitType::Zero,
                              Op::Settings{g, "Init"});

  // Fill data tensor with data from host.
  TensorId data = "D";
  g.createConnectedOp<HostLoadOp>(
      {{HostLoadOp::getLocalTensorInIndex(), dataPrehostload}},
      {{HostLoadOp::getLocalTensorOutIndex(), data}},
      Onnx::CustomOperators::HostLoad,
      Op::Settings{g, "HostLoad"},
      dataStream);

  RemoteCodeLoadOp *remote;
  RemoteCodeLoadOp *remoteOne;

  TensorId outCallOne;

  // For graphOne
  TensorId act1 = actInCallOne;
  for (int i = 0; i < nLayers; i++) {
    std::string layerId   = std::to_string(i + 1);
    TensorId outCallOne   = addScope(graphOne, "outCallOne" + layerId);
    TensorId matmulOutOne = addScope(graphOne, "matmulOut" + layerId);

    graphOne.createConnectedOp<MatMulOp>(
        {{MatMulOp::getLhsInIndex(), wInCallOne},
         {MatMulOp::getRhsInIndex(), act1}},
        {{MatMulOp::getOutIndex(), matmulOutOne}},
        Onnx::Operators::MatMul_9,
        Op::Settings{graphOne, "MatMul" + layerId},
        nonstd::nullopt,
        MatMulOp::SerialiseSettings(),
        OptionalDataType());

    // Go bottom up from the innermost graph to create call ops.
    if (codeCopy) {
      // Add this to graphOne graph, but loads code for
      // callGraphTwo.
      GraphId gid = graphTwo.id;

      remoteOne = graphOne.createConnectedOp<RemoteCodeLoadOp>(
          {},
          {},
          Onnx::CustomOperators::RemoteCodeLoad,
          gid,
          CodeMemoryType::ExecutableMemory,
          Op::Settings{graphOne, "CodeCopy2"});
    }

    auto callTwo = graphOne.createConnectedOp<CallOp>(
        {{graphTwo.getInputIndex(wInCallTwo), wInCallOne},
         {graphTwo.getInputIndex(actInCallTwo), matmulOutOne}},
        {{graphTwo.getOutputIndex(outCallTwo), outCallOne}},
        Onnx::AiGraphcore::OpSet1::Call,
        std::ref(graphTwo),
        Op::Settings{graphOne, "callTwo-" + layerId});

    if (codeCopy) {
      // Add a topo con to ensure the remote code load runs before the call.
      graphOne.topoCons->insert(remoteOne, callTwo);
    }
    act1 = outCallOne;
  }

  TensorId graphOneOut =
      addScope(graphOne, "outCallOne" + std::to_string(nLayers));
  graphOne.markAsOutput(graphOneOut);

  TensorId act = data;
  // For main graph
  for (int i = 0; i < nLayers; i++) {
    std::string layerId = std::to_string(i + 1);
    TensorId weights    = "W" + layerId;
    TensorId callOut    = "out" + layerId;
    g.getTensors().addVarInit(weights, tInfo, weightsHost[i].data());

    if (codeCopy) {
      // Add this to the main graph, but loads code for
      // callGraph.
      GraphId gid = graphOne.id;

      remote = g.createConnectedOp<RemoteCodeLoadOp>(
          {},
          {},
          Onnx::CustomOperators::RemoteCodeLoad,
          gid,
          CodeMemoryType::ExecutableMemory,
          Op::Settings{g, "CodeCopy1"});
    }

    auto call = g.createConnectedOp<CallOp>(
        {{graphOne.getInputIndex(wInCallOne), weights},
         {graphOne.getInputIndex(actInCallOne), act}},
        {{graphOne.getOutputIndex(graphOneOut), callOut}},
        Onnx::AiGraphcore::OpSet1::Call,
        std::ref(graphOne),
        Op::Settings{g, "CallMm-" + layerId});

    TensorId reluOut = "reluOut" + layerId;

    g.createConnectedOp<ReluOp>({{ReluOp::getInIndex(), callOut}},
                                {{ReluOp::getOutIndex(), reluOut}},
                                Onnx::Operators::Relu_6,
                                Op::Settings{g, "ReluOp" + layerId});

    if (codeCopy) {
      // Add a topo con to ensure the remote code load runs before the call.
      g.topoCons->insert(remote, call);
    }

    act = reluOut;
  }

  // Create a stream for the model's output to host.
  ir->setDataFlow(DataFlow{1, {{act, AnchorReturnType("All")}}});
  g.createConnectedOp<HostStoreOp>(
      {{HostStoreOp::getLocalTensorInIndex(), act}},
      {},
      Onnx::CustomOperators::HostStore,
      Op::Settings{g, "HostStore"},
      ir->getAnchorRemap().getRight(act));

  // Set IR state required for lowering.
  auto &opts                   = ir->getSessionOptions();
  opts.aliasZeroCopy           = true;
  opts.enableExplicitMainLoops = true;
  opts.explicitRecomputation   = true;
  opts.useHostCopyOps          = true;

  ir->updateVertices();

  ir->logIr();

  // Lower IR.
  const auto session = InferenceSession::createFromIr(
      std::move(ir), createTestDevice(TEST_TARGET));
  session->prepareDevice();

  // Prepare inputs and anchors.
  std::vector<float> dataHost(tInfo.nelms());
  std::generate(dataHost.begin(), dataHost.end(), sampleFromNormal);
  NDArrayWrapper<float> dataWrapper(dataHost.data(), tInfo);
  std::map<TensorId, IArray &> inputs = {{dataStream, dataWrapper}};

  std::vector<float> outHost(tInfo.nelms());
  NDArrayWrapper<float> outWrapper(outHost.data(), tInfo);
  std::map<TensorId, IArray &> anchors = {{act, outWrapper}};

  // Run the model.
  StepIO stepio(inputs, anchors);
  session->weightsFromHost();
  session->run(stepio);

  auto report                            = session->getReport();
  auto steps                             = report.execution().steps();
  std::vector<pva::ExecutionStep> copies = {};

  for (auto s : steps) {
    // StreamCopies always appear as a triplet {Begin, Mid, End}; just count the
    // Mid's to get the total number.
    if (s.program()->type() == pva::Program::Type::StreamCopyMid) {
      copies.push_back(s);
    }
  }

  return std::pair<std::vector<float>, uint64_t>(outHost, copies.size());
}

BOOST_AUTO_TEST_CASE(TestFunctionBuffersNested) {
  auto runTrue  = runTestCase(true);
  auto runFalse = runTestCase(false);
  auto a        = runTrue.first;
  auto b        = runFalse.first;

  // Numerical test.
  for (int i = 0; i < a.size(); i++) {
    BOOST_CHECK_CLOSE(a[i], b[i], 0.001);
  }

  // TODO  T62707 : ongoing issue with libpva / poplar w.r.t stream copies. The
  // stream copies appear as {SyncAns, Sync} pairs in nested graphs in the
  // profile. Add this check if resolved.

  // BOOST_CHECK(runTrue.second == (runFalse.second + (2 * nLayers)));
}
