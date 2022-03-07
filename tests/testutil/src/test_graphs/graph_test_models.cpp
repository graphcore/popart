// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <testutil/test_graphs/graph_test_models.hpp>

#include <popart/adam.hpp>
#include <popart/aliasesmap.hpp>
#include <popart/clipnormsettings.hpp>
#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/op.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/op/add.hpp>
#include <popart/op/call.hpp>
#include <popart/op/collectives/replicatedallgather.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/collectives/replicatedreducescatter.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/copyvarupdate.hpp>
#include <popart/op/exchange/hostcopy.hpp>
#include <popart/op/exchange/remote.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/init.hpp>
#include <popart/op/iotilecopy.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/loop.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/op/slice.hpp>
#include <popart/op/subtract.hpp>
#include <popart/op/transpose.hpp>
#include <popart/optimizer.hpp>
#include <popart/optimizervalue.hpp>
#include <popart/patterns/adamdecompose.hpp>
#include <popart/patterns/optimizerdecompose.hpp>
#include <popart/patterns/sgd0decompose.hpp>
#include <popart/patterns/sgd1decompose.hpp>
#include <popart/patterns/sgd2decompose.hpp>
#include <popart/sgd.hpp>
#include <popart/tensor.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/autodiff.hpp>
#include <popart/transforms/mergeexchange.hpp>
#include <popart/util.hpp>

#include <poplar/ReplicatedStreamMode.hpp>

using namespace popart;

GraphTestModel::GraphTestModel() {}

GraphTestModel1::GraphTestModel1() {
  Graph &graph     = ir.getMainGraph();
  Graph &subgraph0 = ir.createGraph({"sub0"});
  Graph &subgraph1 = ir.createGraph({"sub1"});

  auto art = AnchorReturnType("All");

  TensorInfo t0Info{DataType::INT32, {4, 4}};
  float t0Data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  graph.getTensors().addVarInit("t0", t0Info, static_cast<void *>(&t0Data));

  TensorInfo t4Info{DataType::INT32, {4, 4}};
  float t4Data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  graph.getTensors().addVarInit("t4", t4Info, static_cast<void *>(&t4Data));

  Op::Settings gSettings(graph, "op", {});
  Op::Settings sg0Settings(subgraph0, "sub0/op", subgraph0.getScope());
  Op::Settings sg1Settings(subgraph1, "sub1/op", subgraph1.getScope());

  Op *s0 = graph.createConnectedOp<SliceOp>({{SliceOp::getInIndex(), "t0"}},
                                            {{SliceOp::getOutIndex(), "t1"}},
                                            Onnx::Operators::Slice_11,
                                            std::vector<int64_t>{0},
                                            std::vector<int64_t>{1},
                                            std::vector<int64_t>{0},
                                            gSettings.copy("Slice0"));

  // t2 is only consumed by a pruneable CallOp
  Op *s1 = graph.createConnectedOp<SliceOp>({{SliceOp::getInIndex(), "t0"}},
                                            {{SliceOp::getOutIndex(), "t2"}},
                                            Onnx::Operators::Slice_11,
                                            std::vector<int64_t>{1},
                                            std::vector<int64_t>{3},
                                            std::vector<int64_t>{0},
                                            gSettings.copy("Slice1"));

  // Inplace slice to create an alias of the weight
  Op *s2 = graph.createConnectedOp<SliceInplaceOp>(
      {{SliceInplaceOp::getInIndex(), "t0"}},
      {{SliceInplaceOp::getOutIndex(), "t3"}},
      Onnx::CustomOperators::SliceInplace,
      std::vector<int64_t>{3},
      std::vector<int64_t>{4},
      std::vector<int64_t>{0},
      std::vector<int64_t>{1},
      gSettings.copy("Slice2"));

  graph.topoCons->insert(s1, s0, true);
  graph.topoCons->insert(s2, s0, false);

  // Subgraph 0
  subgraph0.addInput(addScope(subgraph0, "t3"),
                     graph.getTensors().get("t3")->info);
  subgraph0.addInput(addScope(subgraph0, "t1"),
                     graph.getTensors().get("t1")->info);
  subgraph0.createConnectedOp<SGD0VarUpdateOp>(
      {{SGD0VarUpdateOp::getVarToUpdateInIndex(), addScope(subgraph0, "t3")},
       {SGD0VarUpdateOp::getUpdaterInIndex(), addScope(subgraph0, "t1")}},
      {{SGD0VarUpdateOp::getUpdatedVarOutIndex(), addScope(subgraph0, "t7")}},
      OptimizerValue(0.5, true),
      OptimizerValue(0.5, true),
      sg0Settings.copy("SGD0VarUpdate"));
  subgraph0.markAsOutput(addScope(subgraph0, "t7"));

  // Call which modifies part of a weight indirectly
  graph.createConnectedOp<CallOp>({{0, "t3"}, {1, "t1"}},
                                  {{0, "t7"}},
                                  Onnx::CustomOperators::Call_1,
                                  subgraph0,
                                  std::vector<int>{0},
                                  gSettings.copy("Call0"));

  // Subgraph 1
  subgraph1.addInput(addScope(subgraph1, "t2"),
                     graph.getTensors().get("t2")->info);
  subgraph1.createConnectedOp<IdentityOp>(
      {{IdentityOp::getInIndex(), addScope(subgraph1, "t2")}},
      {{IdentityOp::getOutIndex(), addScope(subgraph1, "t8")}},
      Onnx::Operators::Identity_1,
      sg1Settings.copy("Identity"));
  subgraph1.markAsOutput(addScope(subgraph1, "t8"));

  // Pruneable call
  graph.createConnectedOp<CallOp>({{0, "t2"}},
                                  {{0, "t8"}},
                                  Onnx::CustomOperators::Call_1,
                                  subgraph1,
                                  gSettings.copy("Call1"));

  graph.createConnectedOp<ConcatOp>(
      {{0, "t1"}, {1, "t1"}, {2, "t3"}, {3, "t3"}},
      {{ConcatOp::getOutIndex(), "t6"}},
      Onnx::Operators::Concat_11,
      0,
      gSettings.copy("Concat"));

  graph.createConnectedOp<AddLhsInplaceOp>(
      {{AddOp::getArg0InIndex(), "t4"}, {AddOp::getArg1InIndex(), "t6"}},
      {{AddOp::getOutIndex(), "t5"}},
      Onnx::CustomOperators::AddLhsInplace,
      gSettings.copy("AddLhsInplace"));

  ir.updateVertices();

  df = DataFlow(1, {{"t3", art}});
  ir.setDataFlow(df);
}

GraphTestModel2::GraphTestModel2() {
  Graph &graph = ir.getMainGraph();

  auto art = AnchorReturnType("All");

  TensorInfo t0Info{DataType::INT32, {4, 4}};
  float t0Data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  graph.getTensors().addVarInit("t0", t0Info, static_cast<void *>(&t0Data));

  TensorInfo t1Info{DataType::INT32, {2, 4}};
  float t1Data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  graph.getTensors().addVarInit("t1", t1Info, static_cast<void *>(&t1Data));

  Op::Settings gSettings(graph, "op", {});

  Op *s0 = graph.createConnectedOp<SliceInplaceOp>(
      {{SliceOp::getInIndex(), "t0"}},
      {{SliceOp::getOutIndex(), "t2"}},
      Onnx::CustomOperators::SliceInplace,
      std::vector<int64_t>{0},
      std::vector<int64_t>{2},
      std::vector<int64_t>{0},
      std::vector<int64_t>{1},
      gSettings.copy("Slice0"));

  Op *s1 = graph.createConnectedOp<SliceInplaceOp>(
      {{SliceOp::getInIndex(), "t0"}},
      {{SliceOp::getOutIndex(), "t3"}},
      Onnx::CustomOperators::SliceInplace,
      std::vector<int64_t>{2},
      std::vector<int64_t>{4},
      std::vector<int64_t>{0},
      std::vector<int64_t>{1},
      gSettings.copy("Slice1"));

  graph.topoCons->insert(s0, s1, false);

  Op *cpv = graph.createConnectedOp<CopyVarUpdateOp>(
      {{CopyVarUpdateOp::getVarToUpdateInIndex(), "t2"},
       {CopyVarUpdateOp::getUpdaterInIndex(), "t1"}},
      {{CopyVarUpdateOp::getUpdatedVarOutIndex(), "t4"}},
      gSettings.copy("CopyVarUpdate"));

  Op *add0 = graph.createConnectedOp<AddOp>(
      {{AddOp::getArg0InIndex(), "t3"}, {AddOp::getArg1InIndex(), "t1"}},
      {{AddOp::getOutIndex(), "t5"}},
      Onnx::Operators::Add_7,
      gSettings.copy("AddOp"));

  graph.topoCons->insert(cpv, add0, false);

  Op *sgd0 = graph.createConnectedOp<SGD0VarUpdateOp>(
      {{SGD0VarUpdateOp::getVarToUpdateInIndex(), "t3"},
       {SGD0VarUpdateOp::getUpdaterInIndex(), "t1"}},
      {{SGD0VarUpdateOp::getUpdatedVarOutIndex(), "t6"}},
      OptimizerValue(0.5, true),
      OptimizerValue(0.5, true),
      gSettings.copy("SGD0VarUpdate"));

  graph.topoCons->insert(add0, sgd0, false);

  ir.updateVertices();

  df = DataFlow(1, {{"t5", art}});
  ir.setDataFlow(df);
}

GraphTestModel3::GraphTestModel3(popart::ExchangeStrategy strategyA,
                                 popart::ExchangeStrategy strategyB,
                                 popart::ExchangeStrategy strategyC) {
  Graph &graph = ir.getMainGraph();

  SessionOptions flags;

  int64_t batchesPerStep           = 2;
  flags.accumulationFactor         = 3;
  flags.enableGradientAccumulation = true;

  ir.setUserOptions(flags);

  TensorInfo tInfo(DataType::FLOAT, Shape{1});

  Op::Settings gSettings(graph, "op", {});

  auto addMandatoryLoopSubgraphIO = [](Graph &sg) {
    // Add mandatory loop iterator tensor to subgraph (is not an output)
    TensorId loopIter = addScope(sg, reservedLoopIteratorPrefix());
    sg.addInput(loopIter, TensorInfo{DataType::INT32, {}});

    // Add mandatory loop condition tensor to subgraph (is also an output)
    TensorId loopCond = addScope(sg, reservedLoopCondPrefix());
    sg.addInput(loopCond, TensorInfo{DataType::BOOL, {}});
    sg.markAsOutput(loopCond);
  };

  auto &sg0 = ir.createGraph(GraphId(MainLoops::getStepGraphName()));
  auto &sg1 = ir.createGraph(GraphId(MainLoops::getAccumulationGraphName()));

  addMandatoryLoopSubgraphIO(sg0);
  addMandatoryLoopSubgraphIO(sg1);

  Op::Settings sg0Settings(sg0, "op", {});

  // For Ops on the Compute tiles
  Op::Settings sg1ComputeSettings(sg1, "op", {});

  // For Ops on the IO tiles
  Op::Settings sg1IOSettings(sg1, "op", {});

  sg1ComputeSettings.tileSet = TileSet::Compute;
  sg1IOSettings.tileSet      = TileSet::IO;

  LoopOp *loop0 = graph.createOp<LoopOp>(
      Onnx::Operators::Loop_11, gSettings.copy("loop0"), sg0);
  LoopOp *loop1 = sg0.createOp<LoopOp>(
      Onnx::Operators::Loop_11, sg0Settings.copy("loop1"), sg1);

  loop0->setTripCountValue(batchesPerStep);
  loop1->setTripCountValue(flags.accumulationFactor);

  loop0->setup();
  loop1->setup();

  sg1.createConnectedOp<InitOp>({},
                                {{InitOp::getOutIndex(), addScope(sg1, "A")}},
                                Onnx::CustomOperators::Init_1,
                                tInfo,
                                TensorType::ActGrad,
                                InitType::Zero,
                                sg1IOSettings.copy("Init_A"));

  sg1.createConnectedOp<InitOp>({},
                                {{InitOp::getOutIndex(), addScope(sg1, "B")}},
                                Onnx::CustomOperators::Init_1,
                                tInfo,
                                TensorType::ActGrad,
                                InitType::Zero,
                                sg1IOSettings.copy("Init_B"));

  TensorId streamA = "A";
  graph.getTensors().addStream(
      streamA, tInfo, InputSettings(TileSet::IO, strategyA));

  TensorId streamB = "B";
  graph.getTensors().addStream(
      streamB, tInfo, InputSettings(TileSet::IO, strategyB));

  TensorId streamC = "C";
  graph.getTensors().addActGrad(streamC);
  graph.getTensors().get(streamC)->info = tInfo;

  sg1.createConnectedOp<HostLoadOp>(
      {{HostLoadOp::getLocalTensorInIndex(), addScope(sg1, "A")}},
      {{HostLoadOp::getLocalTensorOutIndex(), addScope(sg1, "A1")}},
      Onnx::CustomOperators::HostLoad,
      sg1IOSettings.copy("HostLoad_A"),
      streamA);

  sg1.createConnectedOp<HostLoadOp>(
      {{HostLoadOp::getLocalTensorInIndex(), addScope(sg1, "B")}},
      {{HostLoadOp::getLocalTensorOutIndex(), addScope(sg1, "B1")}},
      Onnx::CustomOperators::HostLoad,
      sg1IOSettings.copy("HostLoad_B"),
      streamB);

  // IoTileCopyOp: Copies to the tile set specified by the settings (to compute
  // tiles)
  sg1.createConnectedOp<IoTileCopyOp>(
      {{IoTileCopyOp::getInIndex(), addScope(sg1, "A1")}},
      {{IoTileCopyOp::getOutIndex(), addScope(sg1, "A2")}},
      Onnx::CustomOperators::IoTileCopy,
      sg1ComputeSettings.copy("IoTileCopyOp_A"));

  // IoTileCopyOp: Copies to the tile set specified by the settings (to compute
  // tiles)
  sg1.createConnectedOp<IoTileCopyOp>(
      {{IoTileCopyOp::getInIndex(), addScope(sg1, "B1")}},
      {{IoTileCopyOp::getOutIndex(), addScope(sg1, "B2")}},
      Onnx::CustomOperators::IoTileCopy,
      sg1ComputeSettings.copy("IoTileCopyOp_B"));

  sg1.createConnectedOp<AddOp>({{AddOp::getArg0InIndex(), addScope(sg1, "A2")},
                                {AddOp::getArg1InIndex(), addScope(sg1, "B2")}},
                               {{AddOp::getOutIndex(), addScope(sg1, "C2")}},
                               Onnx::Operators::Add_7,
                               sg1ComputeSettings.copy("AddOp"));

  // IoTileCopyOp: Copies to the tile set specified by the settings (to IO
  // tiles)
  sg1.createConnectedOp<IoTileCopyOp>(
      {{IoTileCopyOp::getInIndex(), addScope(sg1, "C2")}},
      {{IoTileCopyOp::getOutIndex(), addScope(sg1, "C")}},
      Onnx::CustomOperators::IoTileCopy,
      sg1IOSettings.copy("IoTileCopyOp_C"));

  sg1.createConnectedOp<HostStoreOp>(
      {{HostStoreOp::getLocalTensorInIndex(), addScope(sg1, "C")}},
      {},
      Onnx::CustomOperators::HostStore,
      sg1IOSettings.copy("HostStore_C"),
      streamC);

  ir.updateVertices();

  auto art = AnchorReturnType("All", TileSet::IO, strategyC);
  df       = DataFlow(batchesPerStep, {{"C", art}});
  ir.setDataFlow(df);
}

GraphTestModel4::GraphTestModel4()
    : GraphTestModel4(popart::ReplicatedStreamMode::Replicate) {}

GraphTestModel4::GraphTestModel4(popart::ReplicatedStreamMode xMode) {
  // Will make dense tensors of this shape with the following repeated values.
  const TensorInfo tInfo{DataType::FLOAT, Shape{2, 2}};
  constexpr float cVal = 5.0f;
  constexpr float wVal = 0.0f;

  const std::vector<float> expectedYData(tInfo.nelms(), 7.0f);

  Graph &graph = ir.getMainGraph();

  // First create the stream tensor x, then create the
  // Init -> HostLoad(x) -> xLoad. No ops should consume x, only xLoad.

  TensorId x = "x";

  graph.getTensors().addStream(x, tInfo, {"x"});
  graph.getTensors().get(x)->setReplicatedStreamMode(xMode);

  TensorId xInit = ir.createIntermediateTensorId(x);
  TensorId xLoad = ir.createIntermediateTensorId(xInit);

  graph.createConnectedOp<InitOp>({},
                                  {{InitOp::getOutIndex(), xInit}},
                                  Onnx::CustomOperators::Init_1,
                                  tInfo,
                                  TensorType::ActGrad,
                                  InitType::Zero,
                                  Op::Settings{graph, "xInit"});

  graph.createConnectedOp<HostLoadOp>(
      {{HostLoadOp::getLocalTensorInIndex(), xInit}},
      {{HostLoadOp::getLocalTensorOutIndex(), xLoad}},
      Onnx::CustomOperators::HostLoad,
      Op::Settings{graph, "hostload_x"},
      x);

  TensorId w = "w";
  std::vector<float> wHost(tInfo.nelms(), wVal);
  graph.getTensors().addVarInit(w, tInfo, wHost.data(), {"w"});

  TensorId wOut = ir.createIntermediateTensorId(w);

  graph.createConnectedOp<AccumulateOp>(
      {{AccumulateOp::getVarToUpdateInIndex(), w},
       {AccumulateOp::getUpdaterInIndex(), xLoad}},
      {{AccumulateOp::getUpdatedVarOutIndex(), wOut}},
      AccumulationType::Add,
      OptimizerValue{}, // Presumably ignored for add.
      Op::Settings{graph, "accumIntoW"});

  TensorId c = "c";
  std::vector<float> cHost(tInfo.nelms(), cVal);
  graph.getTensors().addConstInit(c, tInfo, cHost.data(), {"c"});

  TensorId y = "y";

  graph.createConnectedOp<AddOp>(
      {{AddOp::getArg0InIndex(), wOut}, {AddOp::getArg1InIndex(), c}},
      {{AddOp::getOutIndex(), y}},
      Onnx::Operators::Add_7,
      Op::Settings{graph, "addY"});

  // Must set for HostStored tensors, as HostStoreOpx needs this info.
  // We use AnchorReturnType("All"), as the notion of, for example "Final" does
  // not really translate when there is no implicit or even explicit training
  // loop. Final would work though.
  constexpr int bps = 1;
  ir.setDataFlow(DataFlow{bps, {{y, AnchorReturnType("All")}}});

  // In general, the user's y tensor that they want to stream may get replaced,
  // copied into subgraphs, etc., so there is an "anchor remap" from their
  // original tensor to the actual streamed tensor (note, anchors can be any
  // TensorType except Stream). This latter tensor is the "stream tensor"
  // attribute of the HostStoreOp.
  //
  // In this case, no transformation of the original y tensor has occured, so
  // the anchor remap maps to y, but we follow the general precedent regardless.
  TensorId streamY = ir.getAnchorRemap().getRight(y);

  graph.createConnectedOp<HostStoreOp>(
      {{HostStoreOp::getLocalTensorInIndex(), y}},
      {},
      Onnx::CustomOperators::HostStore,
      Op::Settings{graph, "hostStoreY"},
      streamY);

  ///// Set Ir state required for lowering

  // Some logic in Devicex::loadEngineAndConnectStreams depends on this being
  // set: if useHostCopyOps, for each HostLoad op: get its stream tensor, then
  // get the stream created for that tensor and "connect" it as per usual.
  auto &opts          = ir.getSessionOptions();
  opts.useHostCopyOps = true;

  // Need this as prevents devicex from creating implicit for loop.
  opts.enableExplicitMainLoops = true;

  // Sets ScheduledPreLoss on all vertices, which determines if lowered into
  // forward or backward fragment.
  ir.updateVertices();

  // As the final step before lowering, must mark the Ir as "prepared", or the
  // lowering will immediately throw.
  ir.setIsPrepared();
}

GraphTestModel5::GraphTestModel5(SG1 sg1, SG2 sg2) {

  Graph &graph = ir.getMainGraph();

  TensorId inputs = "inputs";
  TensorInfo inputsInfo{DataType::FLOAT, {1, 1, 10}};

  graph.getTensors().addStream(inputs, inputsInfo, {"inputs"});
  graph.getTensors().get(inputs)->setReplicatedStreamMode(
      ReplicatedStreamMode::Replicate);

  TensorId labels = "labels";
  TensorInfo labelsInfo{DataType::FLOAT, {1, 1, 5}};

  graph.getTensors().addStream(labels, labelsInfo, {"labels"});
  graph.getTensors().get(labels)->setReplicatedStreamMode(
      ReplicatedStreamMode::Replicate);

  TensorId weights = "weights";
  TensorInfo weightsInfo{DataType::FLOAT, {1, 10, 5}};
  std::vector<float> weightsData(50, 0.0f);
  graph.getTensors().addVarInit(
      "weights", weightsInfo, static_cast<void *>(weightsData.data()));

  auto idOp =
      graph.createConnectedOp<IdentityOp>({{IdentityOp::getInIndex(), inputs}},
                                          {{IdentityOp::getOutIndex(), "t1"}},
                                          Onnx::Operators::Identity_1,
                                          Op::Settings{graph, "Identity"});

  auto matMulOp = graph.createConnectedOp<MatMulOp>(
      {{MatMulOp::getLhsInIndex(), inputs},
       {MatMulOp::getRhsInIndex(), weights}},
      {{MatMulOp::getOutIndex(), "t2"}},
      Onnx::Operators::MatMul_1,
      Op::Settings{graph, "MatMul"},
      0.6f,
      MatMulBaseOp::SerialiseSettings{
          MatMulBaseOp::SerialiseSettings::Mode::None, 0, false},
      DataType::FLOAT,
      MatMulPartialsType::FLOAT);

  auto subtractOp = graph.createConnectedOp<SubtractOp>(
      {{SubtractOp::getArg0InIndex(), matMulOp->outId(MatMulOp::getOutIndex())},
       {SubtractOp::getArg1InIndex(), labels}},
      {{SubtractOp::getOutIndex(), "t3"}},
      Onnx::Operators::Sub_1,
      Op::Settings{graph, "Sub"});

  auto l1Op = graph.createConnectedOp<L1Op>(
      {{L1Op::getInIndex(), subtractOp->outId(SubtractOp::getOutIndex())}},
      {{L1Op::getOutIndex(), "loss"}},
      Onnx::CustomOperators::L1,
      0.1f,
      ReductionType::Mean,
      Op::Settings{graph, "L1"});

  TensorId grad1 = "grad1";
  TensorInfo grad1Info{DataType::FLOAT, {}};
  std::vector<float> grad1Data(grad1Info.nelms(), 1.0f);
  graph.getTensors().addConstInit(grad1, grad1Info, grad1Data.data(), {grad1});

  auto l1gradOp = graph.createConnectedOp<L1GradOp>(
      {{L1GradOp::getFwdActInIndex(), l1Op->inId(L1Op::getInIndex())},
       {L1GradOp::getGradInIndex(), grad1}},
      {{L1GradOp::getOutIndex(), "t3grad"}},
      *l1Op);

  auto transposeOp = graph.createConnectedOp<TransposeOp>(
      {{TransposeOp::getInIndex(), idOp->inId(IdentityOp::getInIndex())}},
      {{TransposeOp::getOutIndex(), "tmp1"}},
      Onnx::Operators::Transpose_1,
      Shape{0, 2, 1},
      Op::Settings{graph, "Transpose"});

  auto gradMatMul = graph.createConnectedOp<MatMulOp>(
      {{MatMulOp::getLhsInIndex(),
        transposeOp->outId(TransposeOp::getOutIndex())},
       {MatMulOp::getRhsInIndex(), l1gradOp->outId(L1GradOp::getOutIndex())}},
      {{MatMulOp::getOutIndex(), "inputsgrad"}},
      Onnx::Operators::MatMul_1,
      Op::Settings{graph, "MatMul"},
      0.6f,
      MatMulBaseOp::SerialiseSettings{
          MatMulBaseOp::SerialiseSettings::Mode::None, 0, false},
      DataType::FLOAT,
      MatMulPartialsType::FLOAT);

  Op *allReduce           = nullptr;
  TensorId allReduceOutId = "inputsgradrepl";

  if (sg1 == SG1::No) {

    allReduce = graph.createConnectedOp<ReplicatedAllReduceOp>(
        {{ReplicatedAllReduceOp::getInIndex(),
          gradMatMul->outId(MatMulOp::getOutIndex())}},
        {{ReplicatedAllReduceOp::getOutIndex(), allReduceOutId}},
        Onnx::CustomOperators::ReplicatedAllReduce,
        Op::Settings{graph, "ReplicatedAllReduce"});

  } else {

    auto &sg1Graph    = ir.createGraph(GraphId{"sg1"});
    auto allReduceIn  = addScope(sg1Graph, "in");
    auto allReduceOut = addScope(sg1Graph, "out");

    sg1Graph.addInput(allReduceIn,
                      gradMatMul->outTensor(MatMulOp::getOutIndex())->info);

    allReduce = sg1Graph.createConnectedOp<ReplicatedAllReduceOp>(
        {{ReplicatedAllReduceOp::getInIndex(), allReduceIn}},
        {{ReplicatedAllReduceOp::getOutIndex(), allReduceOut}},
        Onnx::CustomOperators::ReplicatedAllReduce,
        Op::Settings{graph, "ReplicatedAllReduce"});

    sg1Graph.markAsOutput(allReduceOut);

    graph.createConnectedOp<CallOp>(
        {{0, gradMatMul->outId(MatMulOp::getOutIndex())}},
        {{0, allReduceOutId}},
        Onnx::CustomOperators::Call_1,
        sg1Graph,
        Op::Settings{graph, "Call"});
  }

  if (sg2 == SG2::No) {

    graph.createConnectedOp<SGD0VarUpdateOp>(
        {{SGD0VarUpdateOp::getVarToUpdateInIndex(), weights},
         {SGD0VarUpdateOp::getUpdaterInIndex(), allReduceOutId}},
        {{SGD0VarUpdateOp::getUpdatedVarOutIndex(), "weights__updated"}},
        OptimizerValue(0.5, true),
        OptimizerValue(0.5, true),
        Op::Settings{graph, "SGD0VarUpdate"});

  } else {

    auto &sg2Graph  = ir.createGraph(GraphId{"sg2"});
    auto weightsIn  = addScope(sg2Graph, "weightsIn");
    auto varIn      = addScope(sg2Graph, "varIn");
    auto weightsOut = addScope(sg2Graph, "weightsOut");

    sg2Graph.addInput(weightsIn, weightsInfo);
    sg2Graph.addInput(
        varIn,
        allReduce->outTensor(ReplicatedAllReduceOp::getOutIndex())->info);

    sg2Graph.createConnectedOp<SGD0VarUpdateOp>(
        {{SGD0VarUpdateOp::getVarToUpdateInIndex(), weightsIn},
         {SGD0VarUpdateOp::getUpdaterInIndex(), varIn}},
        {{SGD0VarUpdateOp::getUpdatedVarOutIndex(), weightsOut}},
        OptimizerValue(0.5, true),
        OptimizerValue(0.5, true),
        Op::Settings{graph, "SGD0VarUpdate"});

    graph.createConnectedOp<CallOp>({{0, weights}, {1, allReduceOutId}},
                                    {},
                                    Onnx::CustomOperators::Call_1,
                                    sg2Graph,
                                    std::vector<int>{0}, /* modifies  weight */
                                    Op::Settings{graph, "Call"});
  }
}

OptimizerTestModel::OptimizerTestModel(TestOptimizer opt,
                                       unsigned accumulationFactor,
                                       SessionOptions options) {
  Graph &graph = ir.getMainGraph();

  Op::Settings settings(graph, "op", {});

  int64_t batchesPerStep = 2;
  if (accumulationFactor > 1) {
    options.enableGradientAccumulation = true;
    options.accumulationFactor         = accumulationFactor;
  }

  ir.setUserOptions(options);

  TensorInfo t0Info{DataType::FLOAT, {4, 4}};
  float t0Data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  graph.getTensors().addVarInit("t0", t0Info, static_cast<void *>(&t0Data));

  TensorInfo t1Info{DataType::FLOAT, {4}};
  float t1Data[] = {1, 2, 3, 4};
  graph.getTensors().addVarInit("t1", t1Info, static_cast<void *>(&t1Data));

  TensorInfo t2Info{DataType::FLOAT, {4, 4}};
  graph.getTensors().addStream("t2", t2Info);

  graph.createConnectedOp<InitOp>({},
                                  {{InitOp::getOutIndex(), "t2"}},
                                  Onnx::CustomOperators::Init_1,
                                  t2Info,
                                  TensorType::ActGrad,
                                  InitType::Zero,
                                  settings.copy("Init_t2"));

  graph.createConnectedOp<HostLoadOp>(
      {{HostLoadOp::getLocalTensorInIndex(), "t2"}},
      {{HostLoadOp::getLocalTensorOutIndex(), "t2_loaded"}},
      Onnx::CustomOperators::HostLoad,
      settings.copy("HostLoad_t2"),
      "t2");

  graph.createConnectedOp<MatMulOp>({{MatMulOp::getLhsInIndex(), "t0"},
                                     {MatMulOp::getRhsInIndex(), "t2_loaded"}},
                                    {{MatMulOp::getOutIndex(), "t3"}},
                                    Onnx::Operators::MatMul_9,
                                    settings.copy("MatMulOp"),
                                    nonstd::nullopt,
                                    MatMulOp::SerialiseSettings(),
                                    OptionalDataType());

  graph.createConnectedOp<AddOp>(
      {{AddOp::getArg0InIndex(), "t1"}, {AddOp::getArg1InIndex(), "t3"}},
      {{AddOp::getOutIndex(), "t4"}},
      Onnx::Operators::Add_7,
      settings.copy("AddOp"));

  graph.createConnectedOp<L1Op>({{L1Op::getInIndex(), "t4"}},
                                {{L1Op::getOutIndex(), "t5"}},
                                Onnx::CustomOperators::L1,
                                1.0,
                                ReductionType::Mean,
                                settings.copy("L1Op"));

  ir.setFinalLoss("t5");

  std::shared_ptr<Optimizer> optimizer;
  std::shared_ptr<OptimizerDecompose> decomposer;

  switch (opt) {
  case TestOptimizer::SGD0: {
    optimizer  = std::make_shared<SGD>(0.1, 0.1, 0.0, 0.0, 0.1, 1.0);
    decomposer = std::make_shared<SGD0Decompose>();
    break;
  }
  case TestOptimizer::SGD1: {
    optimizer  = std::make_shared<SGD>(0.1,
                                      0.1,
                                      0.8,
                                      0.1,
                                      0.1,
                                      1.0,
                                      std::vector<ClipNormSettings>{},
                                      SGDAccumulatorAndMomentum::Combined);
    decomposer = std::make_shared<SGD1Decompose>();
    break;
  }
  case TestOptimizer::SGD2: {
    optimizer  = std::make_shared<SGD>(0.1,
                                      0.1,
                                      0.8,
                                      0.1,
                                      0.1,
                                      1.0,
                                      std::vector<ClipNormSettings>{},
                                      SGDAccumulatorAndMomentum::Separate);
    decomposer = std::make_shared<SGD2Decompose>();
    break;
  }
  case TestOptimizer::Adam: {
    optimizer  = std::make_shared<Adam>(0.1,
                                       0.1,
                                       0.99,
                                       0.9,
                                       1e-6,
                                       1.0,
                                       AdamMode::Adam,
                                       WeightDecayMode::L2Regularization,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT);
    decomposer = std::make_shared<AdamDecompose>();
    break;
  }
  case TestOptimizer::Lamb: {
    optimizer  = std::make_shared<Adam>(0.1,
                                       0.1,
                                       0.99,
                                       0.9,
                                       1e-6,
                                       1.0,
                                       AdamMode::Lamb,
                                       WeightDecayMode::L2Regularization,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT);
    decomposer = std::make_shared<AdamDecompose>();
    break;
  }
  case TestOptimizer::N:
  default:
    throw internal_error("Unsupported TestOptimizer {}",
                         static_cast<unsigned>(opt));
  }

  ir.setOptimizer(*optimizer);

  ir.updateVertices();

  ir.constructBackwards();

  ir.applyPreAliasPattern(decomposer.get(), graph);

  ir.updateVertices();

  df = DataFlow(batchesPerStep);
  ir.setDataFlow(df);
}

class GraphTestModel6::GTM6Op : public Op {
public:
  GTM6Op(const Op::Settings &settings)
      : Op(OperatorIdentifier("TestOps", "GTM6Op", 1), settings) {}

  void setup() override {
    outInfo(0) = inInfo(0);
    outInfo(1) = inInfo(1);
    outInfo(2) = inInfo(2);
  }

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<GTM6Op>(*this);
  }

  std::vector<std::unique_ptr<Op>> getGradOps() override {
    std::vector<std::unique_ptr<Op>> grads;
    grads.emplace_back(std::make_unique<GraphTestModel6::GTM6GradOp>(*this));
    return grads;
  }

  virtual float getSubgraphValue() const override {
    return getLowSubgraphValue();
  }
};

class GraphTestModel6::GTM6GradOp : public Op {
public:
  GTM6GradOp(const GraphTestModel6::GTM6Op &op)
      : Op(OperatorIdentifier("TestOps", "GTM6GradOp", 1),
           op.Op::getSettings()) {}

  void setup() override {
    outInfo(0) = inInfo(1);
    outInfo(1) = inInfo(0);
    outInfo(2) = inInfo(2);
  }

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<GTM6GradOp>(*this);
  }

  const std::vector<GradInOutMapper> &gradInputInfo() const override {
    static const std::vector<GradInOutMapper> inInfo = {
        {0, 1, GradOpInType::In},
        {1, 0, GradOpInType::GradOut},
        {2, 2, GradOpInType::GradOut}};
    return inInfo;
  }

  const std::map<int, int> &gradOutToNonGradIn() const override {
    static const std::map<int, int> outInfo = {{0, 0}, {1, 1}, {2, 2}};
    return outInfo;
  }

  virtual float getSubgraphValue() const override {
    return getLowSubgraphValue();
  }
};

GraphTestModel6::GraphTestModel6() {

  // Tensor info for tensors in the IR.
  TensorInfo tInfo{DataType::INT32, {}};

  // Create the subgraph.
  auto &subgraphA = ir.createGraph(GraphId("A"));

  // Create subgraph A.
  auto a_in0  = addScope(subgraphA, "in0");
  auto a_in1  = addScope(subgraphA, "in1");
  auto a_in2  = addScope(subgraphA, "in2");
  auto a_out0 = addScope(subgraphA, "out0");
  auto a_out1 = addScope(subgraphA, "out1");
  auto a_out2 = addScope(subgraphA, "out2");

  subgraphA.addInput(a_in0, tInfo);
  subgraphA.addInput(a_in1, tInfo);
  subgraphA.addInput(a_in2, tInfo);

  // Add GTM6Op.
  Op::Settings settingsSubgraphA = Op::Settings{subgraphA, "GTM6Op"};
  subgraphA.createConnectedOp<GTM6Op>({{0, a_in0}, {1, a_in1}, {2, a_in2}},
                                      {{0, a_out0}, {1, a_out1}, {2, a_out2}},
                                      settingsSubgraphA);

  subgraphA.markAsOutput(a_out0);
  subgraphA.markAsOutput(a_out1);
  subgraphA.markAsOutput(a_out2);
}

RemoteRTSTestModel::RemoteRTSTestModel(popart::SessionOptions options) {

  Graph &graph = ir.getMainGraph();

  Op::Settings gSettings(graph, "op", {});

  // For most ops, optimizer status is required to enable RTS
  gSettings.optimizerOp = true;

  auto repFactor = options.replicatedGraphCount;

  // Non-sharded shapes. The first three are intentionally the same, but only
  // the first and the third will be in the same RTS domain
  std::vector<Shape> d{{5, 5, 3}, {5, 5, 3}, {5, 5, 3}, {10, 12, 4}};
  std::vector<RemoteBufferId> remoteBufferIds = {0, 1, 0, 2};
  std::vector<uint64_t> remoteBufferRepeats   = {2, 1, 2, 1};

  std::vector<int> dnelms(d.size());
  std::vector<Shape> ds(d.size());
  std::vector<TensorInfo> domainTSInfos;
  std::vector<TensorInfo> domainTInfos;

  for (size_t i = 0; i < d.size(); ++i) {
    // Number of elements
    dnelms[i] = std::accumulate(d[i].begin(), d[i].end(), 0);
    // Sharded shapes
    ds[i] = {(dnelms[i] + repFactor - 1) / repFactor};

    // Domain tensor infos (non-sharded, just the shape)
    domainTInfos.push_back({DataType::FLOAT, d[i]});

    // Domain tensor infos (sharded tensors, add shape and meta shape)
    domainTSInfos.push_back({DataType::FLOAT, ds[i], d[i]});

    auto initOpS = graph.createConnectedOp<InitOp>(
        {},
        {{InitOp::getOutIndex(), logging::format("D{}_sharded", i)}},
        Onnx::CustomOperators::Init_1,
        domainTSInfos[i],
        TensorType::ActGrad,
        InitType::Zero,
        gSettings);
    initOps.push_back(initOpS);

    auto initOpF = graph.createConnectedOp<InitOp>(
        {},
        {{InitOp::getOutIndex(), logging::format("D{}_full", i)}},
        Onnx::CustomOperators::Init_1,
        domainTInfos[i],
        TensorType::ActGrad,
        InitType::Zero,
        gSettings);
    initOps.push_back(initOpF);

    std::vector<int32_t> data(1, i);
    graph.addConstInit(logging::format("index_{}", i),
                       TensorInfo(DataType::INT32, {}),
                       static_cast<void *>(data.data()),
                       DebugContext{});

    ir.setRemoteBufferInfo(
        remoteBufferIds[i],
        RemoteBufferInfo{domainTInfos[i], remoteBufferRepeats[i]});

    auto loadOp = graph.createConnectedOp<RemoteLoadOp>(
        {{RemoteLoadOp::getLocalTensorInIndex(),
          logging::format("D{}_sharded", i)},
         {RemoteLoadOp::getRemoteBufferOffsetInIndex(),
          logging::format("index_{}", i)}},
        {{RemoteLoadOp::getLocalTensorOutIndex(),
          logging::format("D{}_loaded", i)}},
        Onnx::CustomOperators::RemoteLoad,
        gSettings,
        remoteBufferIds[i]);

    // Facilitate merging
    loadOps.push_back(loadOp);
    if (loadOps.size() > 1) {
      graph.topoCons->insert(loadOps[i - 1], loadOps[i], true);
    }

    auto reduceScatterOp = graph.createConnectedOp<ReplicatedReduceScatterOp>(
        {{ReplicatedReduceScatterOp::getInIndex(),
          logging::format("D{}_full", i)}},
        {{ReplicatedReduceScatterOp::getOutIndex(),
          logging::format("D{}_scattered", i)}},
        Onnx::CustomOperators::ReplicatedReduceScatter,
        CollectiveOperator::Add,
        CommGroup(CommGroupType::All, 0),
        true,
        gSettings);

    reduceScatterOps.push_back(reduceScatterOp);

    auto copyVarUpdate = graph.createConnectedOp<CopyVarUpdateOp>(
        {{CopyVarUpdateOp::getVarToUpdateInIndex(),
          logging::format("D{}_loaded", i)},
         {CopyVarUpdateOp::getUpdaterInIndex(),
          logging::format("D{}_scattered", i)}},
        {{CopyVarUpdateOp::getUpdatedVarOutIndex(),
          logging::format("D{}_updated", i)}},
        gSettings);

    varUpdateOps.push_back(copyVarUpdate);

    domainTensors.push_back(
        graph.getTensors().get(logging::format("D{}_updated", i)));
  }

  ir.applyTransform(MergeExchange::id(), graph);
}

ExplicitRecomputeTestModel::ExplicitRecomputeTestModel(bool pipelining,
                                                       int numLayers,
                                                       int numMatMulsPerLayer) {

  SessionOptions options;

  options.enablePipelining = pipelining;
  options.enableExplicitIR(true);

  Graph &graph = ir.getMainGraph();
  ir.setUserOptions(options);

  std::shared_ptr<Optimizer> optimizer;
  std::shared_ptr<OptimizerDecompose> decomposer;

  optimizer  = std::make_shared<SGD>(0.1, 0.1, 0.0, 0.0, 0.1, 1.0);
  decomposer = std::make_shared<SGD0Decompose>();

  ir.setOptimizer(*optimizer);

  TensorInfo tInfo{DataType::FLOAT, {4, 4}};
  float tData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  std::string input = "I";
  graph.getTensors().addVarInit("I", tInfo, static_cast<void *>(&tData));
  std::string output = "";

  std::vector<TensorId> identities;

  for (size_t i = 0; i < numLayers; ++i) {
    for (size_t j = 0; j < numMatMulsPerLayer; ++j) {
      // W: Weights
      std::string w = "Wt_" + std::to_string(i) + "_" + std::to_string(j);
      // OM: Matmul outputs
      std::string moutput = "OMt" + std::to_string(i) + "_" + std::to_string(j);
      // OA: Addition outputs
      std::string aoutput = "OAt" + std::to_string(i) + "_" + std::to_string(j);
      // OI: Identity outputs
      std::string ioutput = "OIt" + std::to_string(i) + "_" + std::to_string(j);

      // Add skip connection to trigger multiple recompute
      std::string skipoutput = "";
      if (i >= 2) {
        skipoutput = "OMt" + std::to_string(i - 2) + "_" + std::to_string(j);
      }

      graph.getTensors().addVarInit(w, tInfo, static_cast<void *>(&tData));

      auto matMulOp = graph.createConnectedOp<MatMulOp>(
          {{MatMulOp::getLhsInIndex(), input}, {MatMulOp::getRhsInIndex(), w}},
          {{MatMulOp::getOutIndex(), moutput}},
          Onnx::Operators::MatMul_1,
          Op::Settings{graph,
                       "MatMul_" + std::to_string(i) + "_" + std::to_string(j)},
          0.6f,
          MatMulBaseOp::SerialiseSettings{
              MatMulBaseOp::SerialiseSettings::Mode::None, 0, false},
          DataType::FLOAT,
          MatMulPartialsType::FLOAT);

      if (j == 0) {
        matMulOp->settings.recomputeType = RecomputeType::Checkpoint;
      } else {
        matMulOp->settings.recomputeType = RecomputeType::Recompute;
      }
      matMulOp->setPipelineStage(i);
      matMulOp->setVirtualGraphId(i % 2);
      output = moutput;

      // Create an identity to anchor, which will result in the
      // IdentityOp being identified as a `OpFinalLossRelation::FromToLoss`,
      // so an Op that has no direct path to or from the loss, but comes
      // from an upstream `OpFinalLossRelation::ToLoss` operation.
      auto identity = graph.createConnectedOp<IdentityOp>(
          {{IdentityOp::getInIndex(), moutput}},
          {{IdentityOp::getOutIndex(), ioutput}},
          Onnx::Operators::Identity_1,
          Op::Settings{graph,
                       "Identity_" + std::to_string(i) + "_" +
                           std::to_string(j)});

      identities.push_back(ioutput);

      identity->settings.recomputeType = RecomputeType::Checkpoint;
      identity->setPipelineStage(i);
      identity->setVirtualGraphId(i % 2);

      if (!skipoutput.empty()) {
        auto addOp = graph.createConnectedOp<AddOp>(
            {{AddOp::getArg0InIndex(), moutput},
             {AddOp::getArg1InIndex(), skipoutput}},
            {{AddOp::getOutIndex(), aoutput}},
            Onnx::Operators::Add_7,
            Op::Settings{graph,
                         "Add_" + std::to_string(i) + "_" + std::to_string(j)});

        addOp->settings.recomputeType = RecomputeType::Recompute;
        addOp->setPipelineStage(i);
        addOp->setVirtualGraphId(i % 2);
        output = aoutput;
      }

      input = output;
    }
  }

  auto l1Op = graph.createConnectedOp<L1Op>({{L1Op::getInIndex(), output}},
                                            {{L1Op::getOutIndex(), "loss"}},
                                            Onnx::CustomOperators::L1,
                                            0.1f,
                                            ReductionType::Mean,
                                            Op::Settings{graph, "L1"});
  l1Op->setPipelineStage(numLayers - 1);
  l1Op->setVirtualGraphId((numLayers - 1) % 2);

  // Anchor all identities so we generate some "FromToLoss" operations
  auto art = AnchorReturnType("All");
  df       = DataFlow(1, identities, art);
  ir.setDataFlow(df);

  ir.setFinalLoss("loss");
  ir.updateVertices();
  ir.constructBackwards();
  ir.updateVertices();
}

TraverseCallSiteTestModel::TraverseCallSiteTestModel() {
  Graph &graph    = ir.getMainGraph();
  Graph &subgraph = ir.createGraph({"sub"});

  auto art = AnchorReturnType("All");

  TensorInfo tInfo{DataType::INT32, {4, 4}};
  float tData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  graph.getTensors().addVarInit("t0", tInfo, static_cast<void *>(&tData));
  graph.getTensors().addVarInit("t1", tInfo, static_cast<void *>(&tData));
  graph.getTensors().addVarInit("t2", tInfo, static_cast<void *>(&tData));
  graph.getTensors().addVarInit("t3", tInfo, static_cast<void *>(&tData));

  Op::Settings gSettings(graph, "op", {});
  Op::Settings sgSettings(subgraph, "sub/op", subgraph.getScope());

  // Subgraph 0
  subgraph.addInput(addScope(subgraph, "st0"), tInfo);
  subgraph.addInput(addScope(subgraph, "st1"), tInfo);
  subgraph.createConnectedOp<CopyVarUpdateOp>(
      {{CopyVarUpdateOp::getVarToUpdateInIndex(), addScope(subgraph, "st0")},
       {CopyVarUpdateOp::getUpdaterInIndex(), addScope(subgraph, "st1")}},
      {{CopyVarUpdateOp::getUpdatedVarOutIndex(), addScope(subgraph, "st3")}},
      sgSettings);
  subgraph.markAsOutput(addScope(subgraph, "st3"));

  graph.createConnectedOp<CallOp>({{0, "t0"}, {1, "t1"}},
                                  {{0, "t4"}},
                                  Onnx::CustomOperators::Call_1,
                                  subgraph,
                                  gSettings.copy("Call0"));

  graph.createConnectedOp<CallOp>({{0, "t2"}, {1, "t3"}},
                                  {{0, "t5"}},
                                  Onnx::CustomOperators::Call_1,
                                  subgraph,
                                  gSettings.copy("Call0"));

  ir.updateVertices();
}

ExplicitPipelineTestModel0::ExplicitPipelineTestModel0(
    int numPipelineStages,
    int numParallelPaths,
    std::map<int, InputSettings> inputExStrategy,
    std::map<int, AnchorReturnType> outputExStrategy) {

  SessionOptions options;

  // Enable pipelining, IO tiles, explicit IR
  options.enablePipelining = true;
  options.numIOTiles       = 32;
  options.enableExplicitIR(true);

  ir.setUserOptions(options);

  // Main graph
  graphPt = &ir.getMainGraph();

  // Loop body graph
  subgraphPt = &ir.createGraph({"sub"});

  auto &graph    = *graphPt;
  auto &subgraph = *subgraphPt;

  // Settings for operations in the main graph
  Op::Settings gSettings(graph, "op", {});

  // Settings for operations in the loop body graph
  Op::Settings sgSettings(subgraph, "sub/op", subgraph.getScope());

  // For setting HostStore exchange strategies
  AnchorReturnTypeMap artm;

  // Add mandatory loop iterator tensor to subgraph (is not an output)
  TensorId loopItScopedId = addScope(subgraph, reservedLoopIteratorPrefix());
  subgraph.addInput(loopItScopedId, TensorInfo(DataType::INT32, {}));

  // Add mandatory loop condition tensor to subgraph (is also an output)
  TensorId loopCondScopedId = addScope(subgraph, reservedLoopCondPrefix());
  subgraph.addInput(loopCondScopedId, TensorInfo(DataType::BOOL, {}));
  subgraph.markAsOutput(loopCondScopedId);

  // Create the LoopOp
  loopOp = graph.createOp<LoopOp>(
      Onnx::Operators::Loop_11, gSettings.copy("loop"), subgraph);

  // Reuse same shape and data for all tensors
  TensorInfo tInfo{DataType::FLOAT, {4, 4}};
  float tData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  for (size_t i = 0; i < numParallelPaths; ++i) {
    // Process settings, set defaults
    if (inputExStrategy.find(i) == inputExStrategy.end()) {
      inputExStrategy[i] =
          InputSettings{TileSet::Compute, ExchangeStrategy::JustInTime};
    }
    if (outputExStrategy.find(i) == outputExStrategy.end()) {
      outputExStrategy[i] = AnchorReturnType{
          "all", TileSet::Compute, ExchangeStrategy::JustInTime};
    }

    // Declare tensor names

    // AI: Accumulator in tensor
    TensorId accin = "AIt_" + std::to_string(i);

    // AO: Accumulator out tensor
    TensorId accout = "AOt_" + std::to_string(i);

    // ISt: HostLoad input stream
    TensorId instream = "ISt_" + std::to_string(i);

    // OSt: HostStore output stream
    TensorId outstream = "OSt_" + std::to_string(i);

    // IIt: Init
    TensorId init = "IIt_" + std::to_string(i);

    // ILt: Input
    TensorId input = "ILt_" + std::to_string(i);

    // ILIOCt_: Input
    TensorId iocinput = "ILIOCt_" + std::to_string(i);

    // Add accumulator
    graph.getTensors().addVarInit(accin, tInfo, static_cast<void *>(&tData));
    loopOp->addLoopInput(LoopOp::getFirstInputInIndex() + i,
                         accin,
                         addScope(subgraph, accin),
                         false);

    // Mark loop input (accumulator) as modified
    loopOp->addModified(
        LoopOp::getFirstInputInIndex() + i,
        {view::Region::getFull(tInfo.shape(), view::AccessType::ReadWrite)});

    // Add input stream
    graph.getTensors().addStream(instream, tInfo, inputExStrategy.at(i));

    // InitOp + HostLoadOp: Load one input for one loop iteration
    auto initOp = subgraph.createConnectedOp<InitOp>(
        {},
        {{InitOp::getOutIndex(), addScope(subgraph, init)}},
        Onnx::CustomOperators::Init_1,
        tInfo,
        TensorType::ActGrad,
        InitType::Zero,
        sgSettings.copy("Init_" + std::to_string(i)));

    initOp->settings.tileSet = inputExStrategy.at(i).tileSet();
    initOp->setPipelineStage(0);
    initOp->setVirtualGraphId(0);

    auto hostLoadOp = subgraph.createConnectedOp<HostLoadOp>(
        {{HostLoadOp::getLocalTensorInIndex(), addScope(subgraph, init)}},
        {{HostLoadOp::getLocalTensorOutIndex(), addScope(subgraph, input)}},
        Onnx::CustomOperators::HostLoad,
        sgSettings.copy("HostLoad_" + std::to_string(i)),
        instream);

    hostLoadOp->settings.tileSet = inputExStrategy.at(i).tileSet();
    hostLoadOp->setPipelineStage(0);
    hostLoadOp->setVirtualGraphId(0);

    // Insert copy betweeen IO and compute tiles
    if (inputExStrategy.at(i).tileSet() == TileSet::IO) {
      auto ioTileCopyOp = subgraph.createConnectedOp<IoTileCopyOp>(
          {{IoTileCopyOp::getInIndex(), addScope(subgraph, input)}},
          {{IoTileCopyOp::getOutIndex(), addScope(subgraph, iocinput)}},
          Onnx::CustomOperators::IoTileCopy,
          gSettings.copy("IpuCopyToIo_" + std::to_string(i)));
      ioTileCopyOp->settings.tileSet = TileSet::Compute;
      ioTileCopyOp->setPipelineStage(0);
      ioTileCopyOp->setVirtualGraphId(0);
      input = iocinput;
    }

    // Final output
    TensorId output;

    // Add one matmul per pipeline stage
    for (size_t p = 0; p < numPipelineStages; ++p) {
      // Wt: Weight tensor for the matmul
      TensorId w = "Wt_" + std::to_string(i) + "_" + std::to_string(p);
      // Mt: Matmul output tensor
      TensorId moutput = "Mt" + std::to_string(i) + "_" + std::to_string(p);
      // Ct: IpuCopy output tensor
      TensorId coutput = "Ct" + std::to_string(i) + "_" + std::to_string(p);

      graph.getTensors().addVarInit(w, tInfo, static_cast<void *>(&tData));
      loopOp->addLoopInput(LoopOp::getFirstInputInIndex() + loopOp->input->n(),
                           w,
                           addScope(subgraph, w),
                           false);

      auto matMulOp = subgraph.createConnectedOp<MatMulOp>(
          {{MatMulOp::getLhsInIndex(), addScope(subgraph, input)},
           {MatMulOp::getRhsInIndex(), addScope(subgraph, w)}},
          {{MatMulOp::getOutIndex(), addScope(subgraph, moutput)}},
          Onnx::Operators::MatMul_1,
          sgSettings.copy("MatMul_" + std::to_string(i) + "_" +
                          std::to_string(p)),
          0.6f,
          MatMulBaseOp::SerialiseSettings{
              MatMulBaseOp::SerialiseSettings::Mode::None, 0, false},
          DataType::FLOAT,
          MatMulPartialsType::FLOAT);

      matMulOp->setPipelineStage(p);
      matMulOp->setVirtualGraphId(p);

      // If not switching pipeline stages, the next input is the matmul output
      input = moutput;

      if (p < numPipelineStages - 1) {
        auto ipuCopyOp = subgraph.createConnectedOp<IpuCopyOp>(
            {{0, addScope(subgraph, moutput)}},
            {{0, addScope(subgraph, coutput)}},
            Onnx::CustomOperators::IpuCopy,
            static_cast<VGraphId>(p + 1),
            sgSettings.copy("IpuCopy_" + std::to_string(i)));

        ipuCopyOp->setPipelineStage(p);
        ipuCopyOp->setVirtualGraphId(p);

        // If switching pipeline stages, the next input is the copy output
        input = coutput;
      }

      output = moutput;
    }

    // Update the accumulator
    auto accumulateOp = subgraph.createConnectedOp<AccumulateOp>(
        {{AccumulateOp::getVarToUpdateInIndex(), addScope(subgraph, accin)},
         {AccumulateOp::getUpdaterInIndex(), addScope(subgraph, output)}},
        {{AccumulateOp::getUpdatedVarOutIndex(), addScope(subgraph, accout)}},
        AccumulationType::Add,
        OptimizerValue{},
        sgSettings.copy("Accum_" + std::to_string(i)));

    accumulateOp->setPipelineStage(numPipelineStages - 1);
    accumulateOp->setVirtualGraphId(numPipelineStages - 1);

    // Output the updated accumulator (both input modified and loop carried)
    loopOp->addLoopOutput(LoopOp::getFirstOutputOutIndex() + i,
                          accout,
                          addScope(subgraph, accout),
                          false);

    // Mark the updated accumulator as an aliased LoopOp input -> output
    AliasesMap aliasesMap{subgraph};
    Aliases &aliases = aliasesMap.getAliases(subgraph.id);
    auto fwdAliasRegions =
        aliases.getChainsFromTo(subgraph.getTensor(addScope(subgraph, accin)),
                                subgraph.getTensor(addScope(subgraph, accout)));
    auto bwdAliasRegions =
        aliases.getChainsFromTo(subgraph.getTensor(addScope(subgraph, accout)),
                                subgraph.getTensor(addScope(subgraph, accin)));
    loopOp->addAlias(LoopOp::getFirstInputInIndex() + i,
                     LoopOp::getFirstOutputOutIndex() + i,
                     fwdAliasRegions,
                     bwdAliasRegions);

    // IOCOt: Ouptut of the copy from compute to IO tiles
    TensorId iocoutput = "IOCOt_" + std::to_string(i);

    // Insert copy betweeen IO and compute tiles
    if (outputExStrategy.at(i).tileSet() == TileSet::IO) {
      auto ioTileCopyOp = subgraph.createConnectedOp<IoTileCopyOp>(
          {{IoTileCopyOp::getInIndex(), addScope(subgraph, output)}},
          {{IoTileCopyOp::getOutIndex(), addScope(subgraph, iocoutput)}},
          Onnx::CustomOperators::IoTileCopy,
          gSettings.copy("IpuCopyToIo_" + std::to_string(i)));
      ioTileCopyOp->settings.tileSet = TileSet::IO;
      ioTileCopyOp->setPipelineStage(numPipelineStages - 1);
      ioTileCopyOp->setVirtualGraphId(numPipelineStages - 1);
      output = iocoutput;
    }

    // Store the output of one loop iteration
    auto hostStoreOp = subgraph.createConnectedOp<HostStoreOp>(
        {{HostStoreOp::getLocalTensorInIndex(), addScope(subgraph, output)}},
        {},
        Onnx::CustomOperators::HostStore,
        sgSettings.copy("HostStore_" + std::to_string(i)),
        outstream);

    // Set the output exchange strategy
    artm[outstream] = outputExStrategy.at(i);

    hostStoreOp->settings.tileSet = outputExStrategy.at(i).tileSet();
    hostStoreOp->setPipelineStage(numPipelineStages - 1);
    hostStoreOp->setVirtualGraphId(numPipelineStages - 1);
  }

  loopOp->setup();
  loopOp->setTripCountValue(numPipelineStages + 2);

  // Set output strategies (anchors / HostStore)
  df = DataFlow(1, artm);
  ir.setDataFlow(df);

  ir.updateVertices();
}
