// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE popXLFunctionBuffers
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cassert>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <poprithms/compute/host/tensor.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/add.hpp>
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

namespace ptensor = poprithms::compute::host;

using namespace popart;

const int nLayers = 3;

/**
 * As per Basic (inference) CallOp example, but loading the graph's code before
 * each execution.
 */

std::pair<std::vector<float>, uint64_t> runTestCase(bool codeCopy) {

  // Construct IR and main graph.
  auto ir  = std::make_unique<Ir>();
  Graph &g = ir->getMainGraph();

  // Parameters.

  const int r = 5; // Number of matrix rows.
  const int c = r; // Number of matrix cols (same as rows).
  const TensorInfo tInfo(DataType::FLOAT, Shape{r, c});

  auto t = ptensor::Tensor::uniformFloat32(0.0f, 1.0f, {r, c}, 1);

  // Generate random weights on the host.
  std::vector<ptensor::Tensor> weightsHost = {};
  for (int i = 0; i < nLayers; i++) {
    auto w = ptensor::Tensor::uniformFloat32(0.0f, 1.0f, {r, c}, i);
    weightsHost.push_back(w);
  }

  // Build the call subgraph - it only contains a MatMul.
  Graph &callGraph   = ir->createGraph({"callGraph"});
  TensorId wInCall   = addScope(callGraph, "weight(mmLhs)");
  TensorId actInCall = addScope(callGraph, "act(mmRhs)");
  TensorId outInCall = addScope(callGraph, "out");

  callGraph.addInput(wInCall, tInfo);
  callGraph.addInput(actInCall, tInfo);

  callGraph.createConnectedOp<MatMulOp>(
      {{MatMulOp::getLhsInIndex(), wInCall},
       {MatMulOp::getRhsInIndex(), actInCall}},
      {{MatMulOp::getOutIndex(), outInCall}},
      Onnx::Operators::MatMul_9,
      Op::Settings{g, "MatMul"},
      nonstd::nullopt,
      MatMulOp::SerialiseSettings(),
      OptionalDataType());

  callGraph.markAsOutput(outInCall);

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

  // Build the layers of the graph by calling the same subgraph and adding an
  // AddOp after it.
  TensorId act = data;
  for (int i = 0; i < nLayers; i++) {
    std::string layerId = std::to_string(i + 1);
    TensorId weights    = "W" + layerId;
    TensorId callOut    = "out" + layerId;
    g.getTensors().addVarInit(
        weights, tInfo, weightsHost[i].getFloat32Vector().data());
    ExternalCodeCopyOp *remote;
    if (codeCopy) {
      // Add this to the main graph, but loads code for
      // callGraph.
      GraphId gid = callGraph.id;

      remote = g.createConnectedOp<ExternalCodeCopyOp>(
          {},
          {},
          Onnx::CustomOperators::ExternalCodeCopy,
          gid,
          CodeMemoryType::ExecutableMemory,
          Op::Settings{g, "CodeCopy"});
    }
    auto call = g.createConnectedOp<CallOp>(
        {{callGraph.getInputIndex(wInCall), weights},
         {callGraph.getInputIndex(actInCall), act}},
        {{callGraph.getOutputIndex(outInCall), callOut}},
        Onnx::AiGraphcore::OpSet1::Call,
        std::ref(callGraph),
        Op::Settings{g, "CallMm-" + layerId});

    if (codeCopy) {
      // Add a topo con to ensure the remote code load runs before the call.
      g.topoCons->insert(remote, call);
    }

    TensorId addOut               = "Ao" + layerId;
    TensorId addRhs               = "AddRhs" + layerId;
    std::vector<float> addRhsHost = {static_cast<float>(i + 1)};
    g.getTensors().addConstInit(
        addRhs, {DataType::FLOAT, {1}}, addRhsHost.data());

    g.createConnectedOp<AddOp>(
        {{AddOp::getArg0InIndex(), callOut}, {AddOp::getArg1InIndex(), addRhs}},
        {{AddOp::getOutIndex(), addOut}},
        Onnx::Operators::Add_7,
        Op::Settings{g, "Add-" + layerId});

    act = addOut;
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

  session->getReport();

  NDArrayWrapper<float> dataWrapper(t.getFloat32Vector().data(), tInfo);
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

BOOST_AUTO_TEST_CASE(TestFunctionBuffers) {
  auto runTrue  = runTestCase(true);
  auto runFalse = runTestCase(false);
  auto a        = runTrue.first;
  auto b        = runFalse.first;

  // Numerical test.
  for (int i = 0; i < a.size(); i++) {
    BOOST_CHECK(a[i] == b[i]);
  }
  // There should be nLayers more stream copies in the run with code copies
  // (runTrue).
  BOOST_CHECK(runTrue.second == runFalse.second + nLayers);
}
