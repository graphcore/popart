// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PatternsTest

#include <memory>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <testutil/test_graphs/graph_test_models.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/graph.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/op/hostreducevarupdate.hpp>
#include <popart/opmanager.hpp>
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
                           {{tensorIds.back(), AnchorReturnType("All")},
                            {tensorIds[2], AnchorReturnType("All")}});

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
  auto dataFlow = DataFlow(1, {{c0, AnchorReturnType("All")}});

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

BOOST_AUTO_TEST_CASE(KeepOpsWithSideEffects) {
  // The GradCopyToHost op must be registered as it is not done so by default.
  auto gradCopyToHostOpDef =
      OpDefinition({OpDefinition::Inputs({{"X", {DataType::FLOAT}}}),
                    OpDefinition::Outputs({}),
                    OpDefinition::Attributes({})});

  auto gradCopyToHostCreator = OpCreator<GradCopyToHostOp>(
      OpDefinitions(
          {{Onnx::CustomOperators::GradCopyToHost, gradCopyToHostOpDef}}),
      [](const OpCreatorInfo &info) {
        return std::unique_ptr<GradCopyToHostOp>(
            new GradCopyToHostOp(info.settings));
      },
      true);

  // Given: A model with two ops with side effects and one identity output op.
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset11();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};
  auto input = builder->addInputTensor(shape);

  builder->customOp(Onnx::CustomOperators::RemoteStore,
                    1,
                    {input},
                    0,
                    {{"bufferid", 0}},
                    "remote_store");

  builder->customOp(Onnx::CustomOperators::GradCopyToHost,
                    1,
                    {input},
                    0,
                    {},
                    "grad_copy_to_host");

  auto output = aiOnnx.identity({input}, "output");
  builder->addOutputTensor(output);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto dataFlow   = DataFlow(1, {{output, AnchorReturnType("All")}});

  Ir ir;

  ir.setOnnxModel(modelProto);
  ir.setDataFlow(dataFlow);
  ir.registerInputTensors();
  ir.constructForwards();

  // When: Running the prune transform.
  ir.applyTransform(Prune::id(), ir.getMainGraph());
  ir.logIr();

  // Then: All the ops should have been kept.
  BOOST_TEST(ir.getMainGraphOps().size() == 3);
}

BOOST_AUTO_TEST_CASE(PruneGraphModel1Test) {
  GraphTestModel1 model;

  Prune prune;

  // Expected ops
  std::map<std::string, bool> expectedOps;
  expectedOps.insert({"Concat", true});
  expectedOps.insert({"Slice0", true});
  expectedOps.insert({"Slice1", false}); // pruned
  expectedOps.insert({"Slice2", true});
  expectedOps.insert({"AddLhsInplace", true});
  expectedOps.insert({"Call0", true});
  expectedOps.insert({"Call1", false}); // pruned

  // Expected tensors
  std::map<std::string, bool> expectedTensors;
  expectedTensors.insert({"t0", true});
  expectedTensors.insert({"t1", true});
  expectedTensors.insert({"t2", false}); // pruned
  expectedTensors.insert({"t3", true});
  expectedTensors.insert({"t4", true});
  expectedTensors.insert({"t5", true});
  expectedTensors.insert({"t6", true});
  expectedTensors.insert({"t7", true});
  expectedTensors.insert({"t8", false}); // pruned

  // Test
  auto testExistingOpsAndTensors = [&model, &expectedOps, &expectedTensors](
                                       bool isAfter) {
    auto &ops = model.getIr().getMainGraph().getOps();
    std::set<std::string> opNames;
    for (auto &op : ops) {
      opNames.insert(op.second->name());
    }

    auto tensors = model.getIr().getMainGraph().getTensors().getAllTensorIds();

    if (!isAfter) {
      // Check that all tensors and ops are in the expected map
      BOOST_CHECK_EQUAL(ops.size(), expectedOps.size());
      BOOST_CHECK_EQUAL(tensors.size(), expectedTensors.size());
    }

    for (auto &expected : expectedOps) {
      logging::trace("Checking: {}", expected.first);
      if (expected.second || !isAfter) {
        BOOST_CHECK(opNames.find(expected.first) != opNames.end());
      } else {
        BOOST_CHECK(opNames.find(expected.first) == opNames.end());
      }
    }

    for (auto &expected : expectedTensors) {
      logging::trace("Checking: {}", expected.first);
      if (expected.second || !isAfter) {
        BOOST_CHECK(std::find(tensors.begin(), tensors.end(), expected.first) !=
                    tensors.end());
      } else {
        BOOST_CHECK(std::find(tensors.begin(), tensors.end(), expected.first) ==
                    tensors.end());
      }
    }
  };

  // Test before
  testExistingOpsAndTensors(false);
  prune.apply(model.getIr().getMainGraph());
  testExistingOpsAndTensors(true);
}
