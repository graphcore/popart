// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PatternsTest

#include <boost/test/unit_test.hpp>
#include <vector>

#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/graph.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/transforms/prune.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(PruneTest) {
  // Build an onnnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto i1 = builder->addInputTensor(shape);
  std::vector<TensorId> tensorIds{i1};
  // Create some ops which are used
  for (int i = 0; i < 6; i++) {
    auto x = aiOnnx.identity({tensorIds[tensorIds.size() - 1]});
    tensorIds.push_back(x);
  }
  builder->addOutputTensor(tensorIds.back());

  // Add some operations which should be pruned
  aiOnnx.add({i1, i1});
  aiOnnx.add({i1, i1});
  aiOnnx.add({i1, i1});
  aiOnnx.add({i1, i1});
  aiOnnx.add({i1, i1});
  auto i2 = aiOnnx.add({i1, i1});
  aiOnnx.add({i2, i2});
  aiOnnx.add({i2, i2}, "test_add");
  aiOnnx.add({i2, i2});
  aiOnnx.add({i2, i2});
  aiOnnx.add({i2, i2});

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto dataFlow = DataFlow(1,
                           {{tensorIds.back(), AnchorReturnType("ALL")},
                            {tensorIds[2], AnchorReturnType("ALL")}});

  Ir ir;
  ir.setOnnxModel(modelProto);
  ir.setDataFlow(dataFlow);
  ir.registerInputTensors();
  ir.constructForwards();
  ir.applyTransform(Prune::id(), ir.getMainGraph());

  // All but the original 6 operations should be pruned
  BOOST_CHECK(ir.getMainGraphOps().size() == 6);
}

BOOST_AUTO_TEST_CASE(SelectivePruning) {
  // Build an onnnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto i1 = builder->addInputTensor(shape);
  auto c0 = i1;
  auto c1 = i1;
  auto c2 = i1;
  // Create 3 chains of identity ops
  int chainSize = 6;
  for (int i = 0; i < chainSize; i++) {
    c0 = aiOnnx.identity({c0});
    c1 = aiOnnx.identity({c1});
    c2 = aiOnnx.identity({c2});
  }
  builder->addOutputTensor(c0);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto dataFlow = DataFlow(1, {{c0, AnchorReturnType("ALL")}});

  Ir ir;
  ir.setOnnxModel(modelProto);
  ir.setDataFlow(dataFlow);
  ir.registerInputTensors();
  ir.constructForwards();

  auto c1Tensor         = ir.getMainGraph().getTensors().get(c1);
  auto c1Producer       = c1Tensor->getProducer();
  c1Producer->pruneable = false;

  ir.applyTransform(Prune::id(), ir.getMainGraph());
  ir.logIr();

  // Only 2 chains of ops should be left
  BOOST_CHECK(ir.getMainGraphOps().size() == 2 * chainSize);
}
