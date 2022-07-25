// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <cstddef>
#define BOOST_TEST_MODULE UnittestWillowPopxPopTensors
#include <boost/test/unit_test.hpp>

#include <popart/graph.hpp>
#include <popart/graphcoreoperators.hpp>
#include <popart/ir.hpp>
#include <popart/op/init.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensors.hpp>

#include <popart/popx/poptensors.hpp>

#include <snap/Graph.hpp>
#include <snap/Tensor.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(
    TestInsertingTensorsWithSameShapeExceptLeadingOnesIsValid) {

  TensorId tId                       = "t";
  Shape irShape                      = {1, 1, 2, 3};
  std::vector<std::size_t> snapShape = {1, 2, 3};

  // Create Ir tensor
  Ir ir;
  TensorInfo tInfo{DataType::FLOAT, irShape};
  ir.getMainGraph().createConnectedOp<InitOp>(
      {},
      {{InitOp::getOutIndex(), tId}},
      Onnx::CustomOperators::Init_1,
      tInfo,
      TensorType::ActGrad,
      InitType::Zero,
      Op::Settings{ir.getMainGraph(), "Init"});

  // Create snap tensors
  auto snapGraph  = snap::Graph(poplar::Target::createCPUTarget());
  auto snapTensor = snapGraph.addVariable(poplar::FLOAT, snapShape);

  // Test can insert into PopTensors without shape verification throwing.

  popx::PopTensors popTensors{ir};

  BOOST_REQUIRE_NO_THROW(popTensors.insert(tId, snapTensor));
}

BOOST_AUTO_TEST_CASE(TestInsertingTensorsWithDifferentShapesThrows) {

  TensorId tId                            = "t";
  Shape irShape                           = {1, 1, 2, 3};
  poplar::ArrayRef<std::size_t> snapShape = {1, 2, 2, 3};

  // Create Ir tensor
  Ir ir;
  TensorInfo tInfo{DataType::FLOAT, irShape};
  ir.getMainGraph().createConnectedOp<InitOp>(
      {},
      {{InitOp::getOutIndex(), tId}},
      Onnx::CustomOperators::Init_1,
      tInfo,
      TensorType::ActGrad,
      InitType::Zero,
      Op::Settings{ir.getMainGraph(), "Init"});

  // Create snap tensors
  auto snapGraph  = snap::Graph(poplar::Target::createCPUTarget());
  auto snapTensor = snapGraph.addVariable(poplar::FLOAT, snapShape);

  // Test can insert into PopTensors without shape verification throwing.

  popx::PopTensors popTensors{ir};

  BOOST_REQUIRE_THROW(popTensors.insert(tId, snapTensor), popart::error);
}
