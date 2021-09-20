// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE AapiBasicInferenceCallOpTest

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/test/unit_test.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/add.hpp>
#include <popart/op/call.hpp>
#include <popart/op/exchange/hostcopy.hpp>
#include <popart/op/init.hpp>
#include <popart/op/matmul.hpp>
#include <popart/session.hpp>
#include <popart/testdevice.hpp>
#include <popart/util.hpp>
#include <popart/vendored/optional.hpp>

namespace ublas = boost::numeric::ublas;

namespace popart {

ublas::matrix<float> buff2mat(std::vector<float>, int, int);
std::vector<float> mat2buff(ublas::matrix<float>);

/**
 * Basic (inference) CallOp example.
 *
 * Inference on a chain of the same CallOp, which contains a single MatMul.
 *
 * The MatMul in this diagram is inside a subgraph that's called repeatedly:
 *
 *     I      w0
 *     |     /
 *     MatMul
 *      |
 *      |  1.0
 *      |  /
 *      add    w1
 *       |     /
 *       MatMul
 *        |
 *        |  2.0
 *        |  /
 *        add   w2
 *         |     /
 *         MatMul
 *          |
 *          |  3.0
 *          |  /
 *          add   w3
 *           |     /
 *           MatMul
 *            |
 *            |  4.0
 *            |  /
 *            add
 *
 */
BOOST_AUTO_TEST_CASE(TestBasicInference) {
  // Construct IR and main graph.
  auto ir  = std::make_unique<Ir>();
  Graph &g = ir->getMainGraph();

  // Parameters.
  const int nLayers = 5;
  const int r       = 5; // Number of matrix rows.
  const int c       = r; // Number of matrix cols (same as rows).
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
    g.getTensors().addVarInit(weights, tInfo, weightsHost[i].data());

    g.createConnectedOp<CallOp>(
        {{callGraph.getInputIndex(wInCall), weights},
         {callGraph.getInputIndex(actInCall), act}},
        {{callGraph.getOutputIndex(outInCall), callOut}},
        Onnx::AiGraphcore::OpSet1::Call,
        std::ref(callGraph),
        Op::Settings{g, "CallMm-" + layerId});

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
  ir->setIsPrepared();

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

  // Numerical test.
  auto expected = buff2mat(dataHost, r, c);
  for (int i = 0; i < nLayers; i++) {
    auto weights = buff2mat(weightsHost[i], r, c);
    expected     = ublas::prod(weights, expected);
    ublas::scalar_matrix<float> addRhs(r, c, static_cast<float>(i + 1));
    expected += addRhs;
  }
  BOOST_TEST(mat2buff(expected) == outHost);
}

/**
 * Convert an std::vector buffer to a matrix on which linear algebra can be
 * applied.
 *
 * \param buff The input data buffer.
 * \param r The number of rows in the matrix.
 * \param c The number of columns in the matrix.
 * \return linalg::matrix<float> The resulting matrix.
 */
ublas::matrix<float> buff2mat(std::vector<float> buff, int r, int c) {
  ublas::matrix<float> mat(r, c);
  for (int i = 0; i < mat.size1(); ++i)
    for (int j = 0; j < mat.size2(); ++j)
      mat(i, j) = buff[c * i + j];
  return mat;
}

/**
 * Flatten a matrix into an std::vector.
 *
 * \param mat The input matrix.
 * \return std::vector<float> The resulting vector.
 */
std::vector<float> mat2buff(ublas::matrix<float> mat) {
  std::vector<float> buff(mat.size1() * mat.size2());
  for (int i = 0; i < buff.size(); i++)
    buff[i] = mat.data()[i];
  return buff;
}

} // namespace popart
