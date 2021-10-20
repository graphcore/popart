// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_Ir_TensorAccessors
#include <boost/test/unit_test.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(TestOptimizerTensorAccessor) {
  auto ir      = std::make_unique<Ir>();
  Graph &graph = ir->getMainGraph();

  TensorInfo info{DataType::FLOAT, popart::Shape{1}};

  std::vector<TensorId> ids;
  // Some example optimizer tensor ids with prefixes that should
  // be picked up by the accessor
  ids.push_back("lossScaling_FLOAT");
  ids.push_back("learningRate___specific___model.lin.weight");
  ids.push_back("weightDecay___specific___model.lin.bias");

  for (auto id : ids) {
    graph.getTensors().addStream(id, info);
  }

  std::vector<Tensor *> tensors = ir->optimizerTensors();
  BOOST_REQUIRE(tensors.size() == 3);
}

BOOST_AUTO_TEST_CASE(TestOptimizerStateTensorAccessor) {
  auto ir          = std::make_unique<Ir>();
  Tensors &tensors = ir->getMainGraph().getTensors();

  TensorInfo w_info{DataType::FLOAT, popart::Shape{2, 2}};
  TensorInfo b_info{DataType::FLOAT, popart::Shape{2}};
  TensorInfo step_info{DataType::FLOAT, popart::Shape{1}};

  std::vector<float> dummy_buffer(4);

  // Some example optimizer state tensor ids with prefixes that should
  // be picked up
  tensors.addVarInit("Accl1___model.lin.weight", w_info, dummy_buffer.data());
  tensors.addVarInit("Accl2___model.lin.weight", w_info, dummy_buffer.data());
  tensors.addVarInit("Accl1___model.lin.bias", b_info, dummy_buffer.data());
  tensors.addVarInit("Accl2___model.lin.bias", b_info, dummy_buffer.data());
  tensors.addVarInit("Step___model.lin.weight", step_info, dummy_buffer.data());
  tensors.addVarInit("Step___model.lin.bias", step_info, dummy_buffer.data());

  std::vector<Tensor *> state_tensors = ir->optimizerStateTensors();
  BOOST_REQUIRE(state_tensors.size() == 6);
}
