// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <testutil/test_graphs/graph_test_models.hpp>

#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/add.hpp>
#include <popart/op/call.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/copyvarupdate.hpp>
#include <popart/op/exchange/hostcopy.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/init.hpp>
#include <popart/op/iotilecopy.hpp>
#include <popart/op/loop.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/op/slice.hpp>
#include <popart/optimizer.hpp>
#include <popart/optimizervalue.hpp>
#include <popart/sgd.hpp>
#include <popart/tensor.hpp>
#include <popart/topocons.hpp>

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
  subgraph0.addInput(subgraph0.addScope("t3"),
                     graph.getTensors().get("t3")->info);
  subgraph0.addInput(subgraph0.addScope("t1"),
                     graph.getTensors().get("t1")->info);
  subgraph0.createConnectedOp<SGD0VarUpdateOp>(
      {{SGD0VarUpdateOp::getVarToUpdateInIndex(), subgraph0.addScope("t3")},
       {SGD0VarUpdateOp::getUpdaterInIndex(), subgraph0.addScope("t1")}},
      {{SGD0VarUpdateOp::getUpdatedVarOutIndex(), subgraph0.addScope("t7")}},
      OptimizerValue(0.5, true),
      OptimizerValue(0.5, true),
      sg0Settings.copy("SGD0VarUpdate"));
  subgraph0.markAsOutput(subgraph0.addScope("t7"));

  // Call which modifies part of a weight indirectly
  graph.createConnectedOp<CallOp>({{0, "t3"}, {1, "t1"}},
                                  {{0, "t7"}},
                                  Onnx::CustomOperators::Call_1,
                                  subgraph0,
                                  std::vector<int>{0},
                                  gSettings.copy("Call0"));

  // Subgraph 1
  subgraph1.addInput(subgraph1.addScope("t2"),
                     graph.getTensors().get("t2")->info);
  subgraph1.createConnectedOp<IdentityOp>(
      {{IdentityOp::getInIndex(), subgraph1.addScope("t2")}},
      {{IdentityOp::getOutIndex(), subgraph1.addScope("t8")}},
      Onnx::Operators::Identity_1,
      sg1Settings.copy("Identity"));
  subgraph1.markAsOutput(subgraph1.addScope("t8"));

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
    TensorId loopIter = sg.addScope(reservedLoopIteratorPrefix());
    sg.addInput(loopIter, TensorInfo{DataType::INT32, {}});

    // Add mandatory loop condition tensor to subgraph (is also an output)
    TensorId loopCond = sg.addScope(reservedLoopCondPrefix());
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
                                {{InitOp::getOutIndex(), sg1.addScope("A")}},
                                Onnx::CustomOperators::Init_1,
                                tInfo,
                                TensorType::ActGrad,
                                InitType::Zero,
                                sg1IOSettings.copy("Init_A"));

  sg1.createConnectedOp<InitOp>({},
                                {{InitOp::getOutIndex(), sg1.addScope("B")}},
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
      {{HostLoadOp::getLocalTensorInIndex(), sg1.addScope("A")}},
      {{HostLoadOp::getLocalTensorOutIndex(), sg1.addScope("A1")}},
      Onnx::CustomOperators::HostLoad,
      sg1IOSettings.copy("HostLoad_A"),
      streamA);

  sg1.createConnectedOp<HostLoadOp>(
      {{HostLoadOp::getLocalTensorInIndex(), sg1.addScope("B")}},
      {{HostLoadOp::getLocalTensorOutIndex(), sg1.addScope("B1")}},
      Onnx::CustomOperators::HostLoad,
      sg1IOSettings.copy("HostLoad_B"),
      streamB);

  // IoTileCopyOp: Copies to the tile set specified by the settings (to compute
  // tiles)
  sg1.createConnectedOp<IoTileCopyOp>(
      {{IoTileCopyOp::getInIndex(), sg1.addScope("A1")}},
      {{IoTileCopyOp::getOutIndex(), sg1.addScope("A2")}},
      Onnx::CustomOperators::IoTileCopy,
      sg1ComputeSettings.copy("IoTileCopyOp_A"));

  // IoTileCopyOp: Copies to the tile set specified by the settings (to compute
  // tiles)
  sg1.createConnectedOp<IoTileCopyOp>(
      {{IoTileCopyOp::getInIndex(), sg1.addScope("B1")}},
      {{IoTileCopyOp::getOutIndex(), sg1.addScope("B2")}},
      Onnx::CustomOperators::IoTileCopy,
      sg1ComputeSettings.copy("IoTileCopyOp_B"));

  sg1.createConnectedOp<AddOp>({{AddOp::getArg0InIndex(), sg1.addScope("A2")},
                                {AddOp::getArg1InIndex(), sg1.addScope("B2")}},
                               {{AddOp::getOutIndex(), sg1.addScope("C2")}},
                               Onnx::Operators::Add_7,
                               sg1ComputeSettings.copy("AddOp"));

  // IoTileCopyOp: Copies to the tile set specified by the settings (to IO
  // tiles)
  sg1.createConnectedOp<IoTileCopyOp>(
      {{IoTileCopyOp::getInIndex(), sg1.addScope("C2")}},
      {{IoTileCopyOp::getOutIndex(), sg1.addScope("C")}},
      Onnx::CustomOperators::IoTileCopy,
      sg1IOSettings.copy("IoTileCopyOp_C"));

  sg1.createConnectedOp<HostStoreOp>(
      {{HostStoreOp::getLocalTensorInIndex(), sg1.addScope("C")}},
      {},
      Onnx::CustomOperators::HostStore,
      sg1IOSettings.copy("HostStore_C"),
      streamC);

  ir.updateVertices();

  auto art = AnchorReturnType("All", TileSet::IO, strategyC);
  df       = DataFlow(batchesPerStep, {{"C", art}});
  ir.setDataFlow(df);
}
