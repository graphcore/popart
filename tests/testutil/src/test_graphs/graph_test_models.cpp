// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <testutil/test_graphs/graph_test_models.hpp>

#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/add.hpp>
#include <popart/op/call.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/op/slice.hpp>
#include <popart/optimizer.hpp>
#include <popart/optimizervalue.hpp>
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
  graph.getTensors().addVarInit("t4", t0Info, static_cast<void *>(&t4Data));

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
      OptimizerReductionType::None,
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

  ir.updateAliases();
  ir.updateVertices();

  df = DataFlow(1, {{"t3", art}});
  ir.setDataFlow(df);
}
