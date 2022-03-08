// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_Op_AddGrad
#include <boost/test/unit_test.hpp>

#include <popart/ir.hpp>
#include <popart/op/add.hpp>

using namespace popart;

/**
 * Check that calling the Op::setup() function for a AddArg0GradOp
 * is able to affect the data type of the output tensor after construction.
 * i.e.:
 *  1. Construct a AddArg0GradOp, and connect its input and output tensors.
 *  2. Change the data type of its input tensor, and re-call the Op's setup
 *     function
 *  3. Verify that the output tensor's data type has changed accordingly.
 **/
BOOST_AUTO_TEST_CASE(TestAddGradOutputDataTypeIsChangeable) {
  Ir ir;
  Graph &g = ir.getMainGraph();

  // Create a reference AddOp from which the AddArg0GradOp is
  // constructed
  TensorId fwd0In           = "fwd0In";
  TensorId fwd1In           = "fwd1In";
  DataType originalDataType = DataType::FLOAT;
  const TensorInfo fwdInInfo{originalDataType, Shape{2, 3, 4}};
  g.getTensors().addStream(fwd0In, fwdInInfo);
  g.getTensors().addStream(fwd1In, fwdInInfo);
  auto fwdOp = g.createConnectedOp<AddOp>(
      {{AddOp::getArg0InIndex(), fwd0In}, {AddOp::getArg1InIndex(), fwd1In}},
      {{AddOp::getOutIndex(), "fwdOut"}},
      Onnx::Operators::Add_7,
      Op::Settings(g, ""));

  TensorId grad0In           = "grad0In";
  TensorId grad0Out          = "grad0Out";
  TensorId grad1In           = "grad1In";
  TensorId grad1Out          = "grad1Out";
  std::vector<int64_t> axes0 = {0};
  std::vector<int64_t> axes1 = {0};
  const TensorInfo gradInInfo{originalDataType, Shape{2, 3, 4}};
  g.getTensors().addStream(grad0In, gradInInfo);
  g.getTensors().addStream(grad1In, gradInInfo);

  // 1.
  auto grad0Op = g.createConnectedOp<AddArg0GradOp>(
      {{AddArg0GradOp::getInIndex(), grad0In}},
      {{AddArg0GradOp::getOutIndex(), grad0Out}},
      *fwdOp,
      axes0);

  auto grad1Op = g.createConnectedOp<AddArg1GradOp>(
      {{AddArg1GradOp::getInIndex(), grad1In}},
      {{AddArg1GradOp::getOutIndex(), grad1Out}},
      *fwdOp,
      axes1);

  BOOST_TEST(g.getTensors().get(grad0In)->info.dataType() == originalDataType);
  BOOST_TEST(g.getTensors().get(grad1In)->info.dataType() == originalDataType);
  BOOST_TEST(g.getTensors().get(grad0Out)->info.dataType() == originalDataType);
  BOOST_TEST(g.getTensors().get(grad1Out)->info.dataType() == originalDataType);
  DataType newDataType = DataType::FLOAT16; // != originalDataType

  // 2.
  g.getTensors().get(grad0In)->info.set(newDataType);
  g.getTensors().get(grad1In)->info.set(newDataType);
  grad0Op->setup();
  grad1Op->setup();

  // 3.
  BOOST_TEST(g.getTensors().get(grad0In)->info.dataType() == newDataType);
  BOOST_TEST(g.getTensors().get(grad1In)->info.dataType() == newDataType);
  BOOST_TEST(g.getTensors().get(grad0Out)->info.dataType() == newDataType);
  BOOST_TEST(g.getTensors().get(grad1Out)->info.dataType() == newDataType);
}
