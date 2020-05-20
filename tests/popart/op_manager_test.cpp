// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE OpManagerTest

#include <memory>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/op/add.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/nll.hpp>
#include <popart/opmanager.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/testdevice.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(OpManager_Test1) {

  popart::Ir ir;
  auto a = OpManager::createOp(Onnx::Operators::Add_7, ir.getMainGraph());
  BOOST_CHECK(a != nullptr);

  // This will fail as we do not have the custom op
  auto c = OpManager::createOp({"ai_simon", "MyAdd", 2}, ir.getMainGraph());
  BOOST_CHECK(c == nullptr);

  // register the new op, (but it is just an AddOp)
  static OpDefinition opDef(
      {OpDefinition::Inputs({
           {"input", {{DataType::FLOAT, DataType::FLOAT16}}},
       }),
       OpDefinition::Outputs(
           {{"output", {{DataType::FLOAT, DataType::FLOAT16}}}}),
       OpDefinition::Attributes({})});

  OpManager::registerOp({"ai_simon", "MyAdd", 2},
                        opDef,
                        false,
                        [](const OpCreatorInfo &info) -> std::unique_ptr<Op> {
                          return std::unique_ptr<AddOp>(
                              new AddOp(info.opid, info.settings));
                        });

  // Now we do...
  auto b = OpManager::createOp({"ai_simon", "MyAdd", 2}, ir.getMainGraph());
  BOOST_CHECK(b != nullptr);

  // test that we can query for the new operation
  auto ops = OpManager::getSupportedOperations(true);
  auto it  = std::find(
      ops.begin(), ops.end(), OperatorIdentifier("ai_simon", "MyAdd", 2));
  BOOST_CHECK(it != ops.end());
}

BOOST_AUTO_TEST_CASE(OpxManager_Test1) {

  popart::Ir ir;
  auto a = OpManager::createOp(Onnx::Operators::Add_7, ir.getMainGraph());
  BOOST_CHECK(a != nullptr);

  auto aX = popx::OpxManager::createOpx(a.get(), nullptr);
  BOOST_CHECK(aX != nullptr);
}

BOOST_AUTO_TEST_CASE(OpManager_Test2) {

  // register the new op, (but it is just an AddOp)
  static OpDefinition opDef(
      {OpDefinition::Inputs({
           {"input", {{DataType::FLOAT, DataType::FLOAT16}}},
       }),
       OpDefinition::Outputs(
           {{"output", {{DataType::FLOAT, DataType::FLOAT16}}}}),
       OpDefinition::Attributes({})});

  OpManager::registerOp({"ai_simon", "MyAdd2", 1},
                        opDef,
                        false,
                        [](const OpCreatorInfo &info) -> std::unique_ptr<Op> {
                          return std::unique_ptr<AddOp>(
                              new AddOp(info.opid, info.settings));
                        });

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto input1 = builder->addInputTensor(shape);
  auto input2 = builder->addInputTensor(shape);

  auto customOut = builder->customOp({"ai_simon", "MyAdd2", 1},
                                     1,
                                     {input1, input2},
                                     1,
                                     {{"attr1", 42}},
                                     "customOp");

  auto l1 = builder->aiGraphcoreOpset1().l1loss({customOut[0]}, 0.1);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto dataFlow  = DataFlow(1, {{customOut[0], AnchorReturnType("All")}});
  auto optimizer = SGD({{"defaultLearningRate", {0.01, false}}});
  auto device    = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              l1,
              &optimizer,
              *device,
              {},
              Patterns({PreAliasPatternType::PreUniRepl})});

  // Check the ir
  BOOST_CHECK(ir.opsOfType({"ai_simon", "MyAdd2", 1}).size() == 1);
}

// Tests that the input tensor ids are available to the op factory func.
BOOST_AUTO_TEST_CASE(OpManager_Test3) {

  // register the new op, (but it is just an AddOp)
  static OpDefinition opDef(
      {OpDefinition::Inputs({
           {"input", {{DataType::FLOAT, DataType::FLOAT16}}},
       }),
       OpDefinition::Outputs(
           {{"output", {{DataType::FLOAT, DataType::FLOAT16}}}}),
       OpDefinition::Attributes({})});

  // Make sure the factory function actually executes.
  // If it didn't run, none of the other checks would run and the test would
  // pass without checking anything.
  bool factory_func_executed = false;

  OpManager::registerOp(
      {"ai_test", "MyAdd2", 1},
      opDef,
      false,
      // Check that the input ids have been passed to the ops factory function.
      [&factory_func_executed](
          const OpCreatorInfo &info) -> std::unique_ptr<Op> {
        std::cout << "In factory func for MyAdd2.\nInput ids are:\n";
        auto &inputIds = info.getInputIds();
        for (auto &id : inputIds) {
          std::cout << "  " << id << "\n";
        }

        BOOST_CHECK_EQUAL(inputIds.size(), 2);
        BOOST_CHECK_EQUAL(inputIds.at(0), "foo");
        BOOST_CHECK_EQUAL(inputIds.at(1), "bar");
        factory_func_executed = true;

        return std::unique_ptr<AddOp>(new AddOp(info.opid, info.settings));
      });

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto input1 = builder->addInputTensor(shape, "foo");
  auto input2 = builder->addInputTensor(shape, "bar");

  auto customOut = builder->customOp(
      {"ai_test", "MyAdd2", 1}, 1, {input1, input2}, 1, {}, "customOp");

  builder->addOutputTensor(customOut[0]);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto dataFlow = DataFlow(1, {{customOut[0], AnchorReturnType("All")}});
  auto device   = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto, {}, dataFlow, {}, nullptr, *device, {}, Patterns()});

  // Check that the factory func actually executed.
  BOOST_CHECK(factory_func_executed);
}
