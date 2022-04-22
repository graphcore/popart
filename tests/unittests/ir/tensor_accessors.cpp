// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_Ir_TensorAccessors
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <memory>
#include <string>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>

#include "popart/datatype.hpp"
#include "popart/graphid.hpp"
#include "popart/names.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensors.hpp"

namespace popart {
class Tensor;
} // namespace popart

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
  const TensorId t1 = "Accl1___model.lin.weight";
  const TensorId t2 = "Accl2___model.lin.weight";
  const TensorId t3 = "Accl1___model.lin.bias";
  const TensorId t4 = "Accl2___model.lin.bias";
  const TensorId t5 = "Step___model.lin.weight";
  const TensorId t6 = "Step___model.lin.bias";

  tensors.addVarInit(t1, w_info, dummy_buffer.data());
  tensors.addVarInit(t2, w_info, dummy_buffer.data());
  tensors.addVarInit(t3, b_info, dummy_buffer.data());
  tensors.addVarInit(t4, b_info, dummy_buffer.data());
  tensors.addVarInit(t5, step_info, dummy_buffer.data());
  tensors.addVarInit(t6, step_info, dummy_buffer.data());

  BOOST_REQUIRE_NO_THROW(ir->addAdditionalModelProtoTensor(ir->getTensor(t1)));
  BOOST_REQUIRE_NO_THROW(ir->addAdditionalModelProtoTensor(ir->getTensor(t2)));
  BOOST_REQUIRE_NO_THROW(ir->addAdditionalModelProtoTensor(ir->getTensor(t3)));
  BOOST_REQUIRE_NO_THROW(ir->addAdditionalModelProtoTensor(ir->getTensor(t4)));
  BOOST_REQUIRE_NO_THROW(ir->addAdditionalModelProtoTensor(ir->getTensor(t5)));
  BOOST_REQUIRE_NO_THROW(ir->addAdditionalModelProtoTensor(ir->getTensor(t6)));

  std::vector<Tensor *> state_tensors = ir->optimizerStateTensors();
  BOOST_REQUIRE(state_tensors.size() == 6);
}
