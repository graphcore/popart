// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE HistogramOpTest

#include <boost/test/unit_test.hpp>

#include <filereader.hpp>
#include <popart/builder.hpp>
#include <popart/graph.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/dynamic/dynamicslice.hpp>
#include <popart/op/dynamic/dynamicupdate.hpp>
#include <popart/op/exchange/hostcopy.hpp>
#include <popart/op/init.hpp>
#include <popart/opidentifier.hpp>
#include <popart/session.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/testdevice.hpp>

#include <testutil/irquery/irquery.hpp>

using namespace popart;
using namespace popart::irquery;

// Test that DynamicUpdate gets inplaced
BOOST_AUTO_TEST_CASE(DynamicUpdateInplace_test) {
  SessionOptions opts;
  Ir ir;
  ir.setUserOptions(opts);
  Graph &g = ir.getMainGraph();

  Op::Settings opSettings(g, "");

  TensorInfo tInfoA(DataType::FLOAT, Shape{5, 2, 2});
  TensorInfo tInfoB(DataType::FLOAT, Shape{1, 2, 2});
  TensorInfo tInfoC(DataType::INT32, Shape{});

  g.createConnectedOp<InitOp>({},
                              {{InitOp::getOutIndex(), "A"}},
                              Onnx::CustomOperators::Init_1,
                              tInfoA,
                              TensorType::ActGrad,
                              InitType::Zero,
                              opSettings.copy("Init_A"));

  g.createConnectedOp<InitOp>({},
                              {{InitOp::getOutIndex(), "B"}},
                              Onnx::CustomOperators::Init_1,
                              tInfoB,
                              TensorType::ActGrad,
                              InitType::Zero,
                              opSettings.copy("Init_B"));

  g.createConnectedOp<InitOp>({},
                              {{InitOp::getOutIndex(), "C_init"}},
                              Onnx::CustomOperators::Init_1,
                              tInfoC,
                              TensorType::ActGrad,
                              InitType::Zero,
                              opSettings.copy("Init_C"));

  TensorId streamC = "C";
  g.getTensors().addActGrad(streamC);
  g.getTensors().get(streamC)->info = tInfoC;

  g.createConnectedOp<HostLoadOp>(
      {{HostLoadOp::getLocalTensorInIndex(), "C_init"}},
      {{HostLoadOp::getLocalTensorOutIndex(), "C"}},
      Onnx::CustomOperators::HostLoad,
      opSettings.copy("HostLoad_C"),
      streamC);

  // Use any VarUpdateOp
  g.createConnectedOp<DynamicUpdateOp>(
      {{DynamicUpdateOp::getInIndex(), "B"},
       {DynamicUpdateOp::getIndexInIndex(), "C"},
       {DynamicUpdateOp::getUpdateInIndex(), "A"}},
      {{DynamicUpdateOp::getOutIndex(), "D"}},
      Onnx::CustomOperators::DynamicUpdate_1,
      std::vector<int64_t>{0},
      std::vector<int64_t>{1},
      true,
      opSettings.copy("DynamicUpdate"));

  ir.applyInplacePattern(g);

  IrTestWrapper tw_ir{ir};
  auto tw_mainGraph = tw_ir.hasGraph(ir.getMainGraph().id, Require::MustBeTrue);
  auto tw_dynamicUpdate = tw_mainGraph->ops().hasOp<DynamicUpdateInplaceOp>(
      [&](auto &op) -> bool { return true; }, Require::MustBeTrue);
}

// Test that DynamicSlice gets inplaced
BOOST_AUTO_TEST_CASE(DynamicSliceInplace_test) {
  SessionOptions opts;
  Ir ir;
  ir.setUserOptions(opts);
  Graph &g = ir.getMainGraph();

  Op::Settings opSettings(g, "");

  TensorInfo tInfoA(DataType::FLOAT, Shape{5, 2, 2});
  TensorInfo tInfoB(DataType::FLOAT, Shape{1, 2, 2});
  TensorInfo tInfoC(DataType::INT32, Shape{});

  g.createConnectedOp<InitOp>({},
                              {{InitOp::getOutIndex(), "A"}},
                              Onnx::CustomOperators::Init_1,
                              tInfoA,
                              TensorType::ActGrad,
                              InitType::Zero,
                              opSettings.copy("Init_A"));

  g.createConnectedOp<InitOp>({},
                              {{InitOp::getOutIndex(), "B"}},
                              Onnx::CustomOperators::Init_1,
                              tInfoB,
                              TensorType::ActGrad,
                              InitType::Zero,
                              opSettings.copy("Init_B"));

  g.createConnectedOp<InitOp>({},
                              {{InitOp::getOutIndex(), "C_init"}},
                              Onnx::CustomOperators::Init_1,
                              tInfoC,
                              TensorType::ActGrad,
                              InitType::Zero,
                              opSettings.copy("Init_C"));

  TensorId streamC = "C";
  g.getTensors().addActGrad(streamC);
  g.getTensors().get(streamC)->info = tInfoC;

  g.createConnectedOp<HostLoadOp>(
      {{HostLoadOp::getLocalTensorInIndex(), "C_init"}},
      {{HostLoadOp::getLocalTensorOutIndex(), "C"}},
      Onnx::CustomOperators::HostLoad,
      opSettings.copy("HostLoad_C"),
      streamC);

  // Use any VarUpdateOp
  g.createConnectedOp<DynamicSliceOp>(
      {{DynamicSliceOp::getInIndex(), "A"},
       {DynamicSliceOp::getIndexInIndex(), "C"},
       {DynamicSliceOp::getSliceInIndex(), "B"}},
      {{DynamicSliceOp::getOutIndex(), "D"}},
      Onnx::CustomOperators::DynamicSlice_1,
      std::vector<int64_t>{0},
      std::vector<int64_t>{1},
      true,
      opSettings.copy("DynamicSlice"));

  ir.applyInplacePattern(g);

  IrTestWrapper tw_ir{ir};
  auto tw_mainGraph = tw_ir.hasGraph(ir.getMainGraph().id, Require::MustBeTrue);
  auto tw_dynamicUpdate = tw_mainGraph->ops().hasOp<DynamicSliceInplaceOp>(
      [&](auto &op) -> bool { return true; }, Require::MustBeTrue);
}
