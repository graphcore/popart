// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_AAPI_loop_accumulate
#include <boost/test/unit_test.hpp>
#include <memory>
#include <onnx/onnx_pb.h>
#include <tuple>
#include <vector>
#include <popart/aliasesmap.hpp>
#include <popart/graph.hpp>
#include <popart/iarray.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/op/add.hpp>
#include <popart/op/call.hpp>
#include <popart/op/exchange/hostcopy.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/init.hpp>
#include <popart/op/loop.hpp>
#include <popart/op/mul.hpp>
#include <popart/optimizervalue.hpp>
#include <popart/session.hpp>
#include <popart/stepio.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>
#include <popart/transforms/autodiff.hpp>
#include <popart/util.hpp>

float generateReferenceWeights(const float,
                               const std::vector<float> &,
                               const int,
                               const float);

namespace popart {
/**
 * Create a basic training loop directly in IR:
 *
 *     W <- [0]
 *
 *     Loss(W, D) = (W + D) * D
 *
 *     for D in [1, 2, 3, 4]:
 *       stream D to device
 *       compute W_grad = dLoss / dW
 *       W -= lr * W_grad
 *
 *     return W to device.
 *
 * This isn't using the "implied" training loop, but a constructed LoopOp.
 */
BOOST_AUTO_TEST_CASE(TestBasicTrainingLoop) {
  auto test = [](bool useAzc) {
    // Construct IR and main graph.
    auto ir      = std::make_unique<Ir>();
    Graph &graph = ir->getMainGraph();
    Op::Settings gSettings(graph, "op", {});

    // Parameters.
    float lr                     = 0.1; // Learning rate in W -= lr * dW.
    const int loopTripCount      = 10;  // Number of loop iterations.
    const float weightsInitValue = 1.0; // Initial value for weights.

    // Define type and shape of tensors.
    TensorInfo tInfo(DataType::FLOAT, Shape{1}); // Weights and data info.

    // Host buffers.
    std::vector<float> dataHost(loopTripCount);
    std::iota(
        std::begin(dataHost), std::end(dataHost), 1); // Fill with [1,2,...]
    std::vector<float> weightsHost(tInfo.nelms(), weightsInitValue);
    const std::vector<float> oneHost(tInfo.nelms(), 1);

    // Create data stream.
    TensorId dataStream = "D_stream";
    graph.getTensors().addStream(dataStream, tInfo);

    // Create and read weights.
    TensorId weights    = "W";
    TensorId weightsOut = "W_out";
    graph.getTensors().addVarInit(weights, tInfo, weightsHost.data());

    // Create the loss (forward) subgraph. The loss is Loss(W, D) = (W + D) * D.
    // Inputs:
    //   - data (D)
    //   - weights (W)
    // Outputs:
    //   - loss = (W + D) * D
    // -------------------------------------------------------------------------
    auto &fwd = ir->createGraph(GraphId("fwd-subgraph"));
    Op::Settings fwdSettings(fwd, "fwd", fwd.getScope());

    TensorId fwdData    = addScope(fwd, "data");
    TensorId fwdWeights = addScope(fwd, "weights");
    TensorId fwdTmp     = addScope(fwd, "tmp");
    TensorId fwdLoss    = addScope(fwd, "loss");
    fwd.addInput(fwdData, tInfo);
    fwd.addInput(fwdWeights, tInfo);

    fwd.createConnectedOp<AddOp>({{AddOp::getArg0InIndex(), fwdWeights},
                                  {AddOp::getArg1InIndex(), fwdData}},
                                 {{AddOp::getOutIndex(), fwdTmp}},
                                 Onnx::Operators::Add_7,
                                 fwdSettings.copy("Add"));

    fwd.createConnectedOp<MulOp>(
        {{MulOp::getArg0InIndex(), fwdTmp}, {MulOp::getArg1InIndex(), fwdData}},
        {{MulOp::getOutIndex(), fwdLoss}},
        Onnx::Operators::Mul_7,
        fwdSettings.copy("Mul"));

    fwd.markAsOutput(fwdLoss);

    // Create the backward graph of the loss subgraph.
    // Inputs:
    //   - Gradient___loss
    //   - data
    //   - loss
    //   - weights
    // Outputs:
    //   - Gradient___data = 2 * data + weights
    //   - Gradient___weights = data
    // -------------------------------------------------------------------------
    Autodiff autodiff;
    auto result = autodiff.apply(*ir,
                                 fwd.id,
                                 Autodiff::TensorIds({fwdLoss}),
                                 Autodiff::TensorIds({fwdWeights}),
                                 FwdGraphToBwdGraphInfo(),
                                 AutodiffStitchStrategy::RecomputeMinimal);

    auto &bwd = ir->getGraph(result.at(fwd.id).bwdGraphId);

    TensorId bwdData    = addScope(bwd, "data");
    TensorId bwdWeights = addScope(bwd, "weights");
    TensorId bwdLoss    = addScope(bwd, "loss");
    TensorId bwdLossGrad =
        addScope(bwd, reservedGradientPrefix() + std::string("loss"));
    TensorId bwdDataGrad =
        addScope(bwd, reservedGradientPrefix() + std::string("data"));
    TensorId bwdWeightsGrad =
        addScope(bwd, reservedGradientPrefix() + std::string("weights"));

    // Create training loop.
    // -------------------------------------------------------------------------
    Graph &loopSg = ir->createGraph(GraphId{"loop-subgraph"});
    Op::Settings loopSettings(graph, "loop");

    // Add mandatory loop iterator tensor to subgraph (is not an output)
    TensorId loopIter = addScope(loopSg, reservedLoopIteratorPrefix());
    loopSg.addInput(loopIter, TensorInfo{DataType::INT32, {}});

    // Add mandatory loop condition tensor to subgraph (is also an output)
    TensorId loopCond = addScope(loopSg, reservedLoopCondPrefix());
    loopSg.addInput(loopCond, TensorInfo{DataType::BOOL, {}});
    loopSg.markAsOutput(loopCond);

    // Create LoopOp in parent graph.
    auto loopOp =
        graph.createOp<LoopOp>(Onnx::Operators::Loop_11, loopSettings, loopSg);
    loopOp->setTripCountValue(loopTripCount);

    TensorId loopWeights = addScope(loopSg, weights);
    loopOp->addLoopInput(
        LoopOp::getFirstInputInIndex(), weights, loopWeights, false);

    // Init data tensor. This is op is required for HostLoadOp - see
    // hostcopy.hpp.
    TensorId dataPrehostload = addScope(loopSg, "D_prehostload");
    loopSg.createConnectedOp<InitOp>({},
                                     {{InitOp::getOutIndex(), dataPrehostload}},
                                     Onnx::CustomOperators::Init_1,
                                     tInfo,
                                     TensorType::ActGrad,
                                     InitType::Zero,
                                     loopSettings.copy("Init"));

    // Fill data tensor with data from host.
    TensorId loopData = addScope(loopSg, "D");
    loopSg.createConnectedOp<HostLoadOp>(
        {{HostLoadOp::getLocalTensorInIndex(), dataPrehostload}},
        {{HostLoadOp::getLocalTensorOutIndex(), loopData}},
        Onnx::CustomOperators::HostLoad,
        loopSettings.copy("HostLoad"),
        dataStream);

    // Call the loss (forward) subgraph.
    TensorId loopLoss = addScope(loopSg, "loss");
    loopSg.createConnectedOp<CallOp>(
        {
            {fwd.getInputIndex(fwdData), loopData},
            {fwd.getInputIndex(fwdWeights), loopWeights},
        },
        {
            {fwd.getOutputIndex(fwdLoss), loopLoss},
        },
        Onnx::AiGraphcore::OpSet1::Call,
        std::ref(fwd),
        loopSettings.copy("CallFwd"));

    // Call the backward loss subgraph.
    TensorId loopWeightsGrad = addScope(loopSg, "W_grad");
    TensorId loopOne         = addScope(loopSg, "one");
    loopSg.getTensors().addConstInit(loopOne, tInfo, oneHost.data());
    loopSg.createConnectedOp<CallOp>(
        {
            {bwd.getInputIndex(bwdLossGrad), loopOne},
            {bwd.getInputIndex(bwdLoss), loopLoss},
            {bwd.getInputIndex(bwdData), loopData},
            {bwd.getInputIndex(bwdWeights), loopWeights},
        },
        {
            {bwd.getOutputIndex(bwdWeightsGrad), loopWeightsGrad},
        },
        Onnx::AiGraphcore::OpSet1::Call,
        std::ref(bwd),
        loopSettings.copy("CallBwd"));

    // Weight update.
    TensorId loopWeightsOut = addScope(loopSg, weightsOut);
    loopSg.createConnectedOp<AccumulateOp>(
        {
            {AccumulateOp::getVarToUpdateInIndex(), loopWeights},
            {AccumulateOp::getUpdaterInIndex(), loopWeightsGrad},
        },
        {
            {AccumulateOp::getUpdatedVarOutIndex(), loopWeightsOut},
        },
        AccumulationType::DampenedAdd,
        OptimizerValue{-lr},
        loopSettings.copy("Accumulate"));

    loopOp->addLoopOutput(
        LoopOp::getFirstOutputOutIndex(), weightsOut, loopWeightsOut, false);
    loopOp->setup();

    // Transfer modified inputs. This allows changes to loopWeights in the
    // loop subgraph to be reflected to weights in the main graph.
    AliasesMap aliasesMap{graph};
    Aliases &aliases = aliasesMap.getAliases(graph.id);
    std::vector<popart::Op *> loopOps;
    for (auto const &opMap : loopSg.getOps())
      loopOps.push_back(opMap.second.get());
    auto modifiedRegions = loopSg.getTensors()
                               .get(loopWeights)
                               ->modifiedRegionsByOps(loopOps, aliases);
    loopOp->addModified(
        loopOp->input->indices(graph.getTensors().get(weights))[0],
        modifiedRegions);

    // This needs to be set to the number of times data is be read from
    // host.
    ir->setDataFlow(DataFlow{loopTripCount});

    // Set IR state required for lowering.
    auto &opts                   = ir->getSessionOptions();
    opts.enableExplicitMainLoops = true;
    opts.useHostCopyOps          = true;
    opts.aliasZeroCopy           = useAzc;
    opts.explicitRecomputation   = useAzc;
    ir->updateVertices();
    ir->setOnnxModel({});
    ir->setPatterns(Patterns(PatternsLevel::Minimal));
    for (auto &id_graph : ir->getGraphs()) {
      auto &_graph = ir->getGraph(id_graph.first);
      ir->applyPreAliasPatterns(_graph);
    }
    for (auto &id_graph : ir->getGraphs()) {
      auto &_graph = ir->getGraph(id_graph.first);
      ir->applyInplacePattern(_graph);
    }

    // Lower IR.
    const auto session = TrainingSession::createFromIr(
        std::move(ir), createTestDevice(TEST_TARGET));
    session->prepareDevice();

    // An extra dimension of `loopTripCount` needs to be added to the input
    // data on the host.
    TensorInfo dInfo(DataType::FLOAT, Shape{loopTripCount, 1});
    NDArrayWrapper<float> dataWrapper(dataHost.data(), dInfo);
    std::map<TensorId, IArray &> inputs = {{dataStream, dataWrapper}};

    StepIO stepio(inputs, {});
    stepio.enableRuntimeAsserts(false);

    session->weightsFromHost();
    session->run(stepio);

    WeightsIO weightsRead;
    weightsRead.insert(weights, {weightsHost.data(), tInfo});
    session->weightsToHost();
    session->readWeights(weightsRead);

    // Numerical tests.
    float weightsReference =
        generateReferenceWeights(weightsInitValue, dataHost, loopTripCount, lr);
    using boost::test_tools::tolerance;
    BOOST_TEST(weightsHost[0] == weightsReference, tolerance(1e-8));
  };

  test(true);
  test(false);
}
} // namespace popart

/**
 * Generate reference weights for the popART basic training loop.
 *
 * If the loss is L(W, D) = (W + D) * D, then the derivative of the loss w.r.t.
 * the weights W is dL / dW = D. Thus, the basic training loop is equivalent to:
 *
 *     for D in [1, 2, 3, 4]:
 *       W -= lr * D.
 *
 * This function implements this equivalent training loop.
 *
 * \param weightsInitValue The initial value of the weights.
 * \param data A vector with the input data.
 * \param loopTripCount The number of iterations.
 * \param lr The learning rate for weight updates.
 */
float generateReferenceWeights(const float weightsInitValue,
                               const std::vector<float> &data,
                               const int loopTripCount,
                               const float lr) {
  float weights = weightsInitValue;
  for (int i = 0; i < loopTripCount; i++) {
    weights -= lr * data[i];
  }
  return weights;
}
