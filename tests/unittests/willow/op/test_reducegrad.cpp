// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_Op_ReduceGrad
#include <boost/test/unit_test.hpp>

#include <popart/ir.hpp>
#include <popart/op/reducesum.hpp>

using namespace popart;

/**
 * Check that calling the Op::setup() function for a Reduce*GradOp
 * is able to affect the data type of the output tensor after construction.
 * i.e.:
 *  1. Construct a ReduceSumGradOp, and connect its input and output tensors.
 *  2. Change the data type of its input tensor, and re-call the Op's setup
 *     function
 *  3. Verify that the output tensor's data type has changed accordingly.
 **/
BOOST_AUTO_TEST_CASE(TestReduceGradOutputDataTypeIsChangeable) {
  Ir ir;
  Graph &g = ir.getMainGraph();

  // Create a reference ReduceSumOp from which the ReduceSumGradOp is
  // constructed
  nonstd::optional<std::vector<int64_t>> axes;
  TensorId fwdIn            = "fwdIn";
  DataType originalDataType = DataType::FLOAT;
  const TensorInfo fwdInInfo{originalDataType, Shape{6, 5, 4}};
  g.getTensors().addStream(fwdIn, fwdInInfo);
  auto fwdOp =
      g.createConnectedOp<ReduceSumOp>({{ReduceSumOp::getInIndex(), fwdIn}},
                                       {{ReduceSumOp::getOutIndex(), "fwdOut"}},
                                       Onnx::Operators::ReduceSum_1,
                                       axes,
                                       1,
                                       Op::Settings(g, ""));

  TensorId gradIn  = "gradIn";
  TensorId gradOut = "gradOut";
  const TensorInfo gradInInfo{DataType::FLOAT, Shape{6, 5, 4}};
  g.getTensors().addStream(gradIn, gradInInfo);

  // 1.
  auto gradOp = g.createConnectedOp<ReduceSumGradOp>(
      {{ReduceSumGradOp::getInIndex(), gradIn}},
      {{ReduceSumGradOp::getOutIndex(), gradOut}},
      *fwdOp,
      Shape{1, 1, 1});

  BOOST_TEST(g.getTensors().get(gradIn)->info.dataType() == originalDataType);
  BOOST_TEST(g.getTensors().get(gradOut)->info.dataType() == originalDataType);
  DataType newDataType = DataType::FLOAT16; // != originalDataType

  // 2.
  g.getTensors().get(gradIn)->info.set(newDataType);
  gradOp->setup();

  // 3.
  BOOST_TEST(g.getTensors().get(gradIn)->info.dataType() == newDataType);
  BOOST_TEST(g.getTensors().get(gradOut)->info.dataType() == newDataType);
}
