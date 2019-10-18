#define BOOST_TEST_MODULE OpManagerTest

#include <boost/test/unit_test.hpp>
#include <vector>

#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/op/add.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/nll.hpp>
#include <popart/opmanager.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(OpManager_Test1) {

  popart::Ir ir;
  auto a = OpManager::createOp(Onnx::Operators::Add_7, ir.getMainGraph());
  BOOST_CHECK(a != nullptr);

  // This will fail as we do not have the custom op
  auto c = OpManager::createOp({"ai_simon", "MyAdd", 2}, ir.getMainGraph());
  BOOST_CHECK(c == nullptr);

  // register the new op, (but it is just an AddOp)
  OpManager::registerOp({"ai_simon", "MyAdd", 2},
                        false,
                        [](const OperatorIdentifier &_opid,
                           const Op::Settings &settings,
                           const Attributes &attr = {}) -> std::unique_ptr<Op> {
                          return std::unique_ptr<AddOp>(
                              new AddOp(_opid, settings));
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
  OpManager::registerOp({"ai_simon", "MyAdd2", 1},
                        false,
                        [](const OperatorIdentifier &_opid,
                           const Op::Settings &settings,
                           const Attributes &attr = {}) -> std::unique_ptr<Op> {
                          return std::unique_ptr<AddOp>(
                              new AddOp(_opid, settings));
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
  auto padOut    = aiOnnx.add({input1, input2});

  builder->addOutputTensor(customOut[0]);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto dataFlow  = DataFlow(1, {{customOut[0], AnchorReturnType("ALL")}});
  auto optimizer = SGD({{"defaultLearningRate", {0.01, false}}});
  std::vector<Loss *> losses{
      new L1Loss(customOut[0], "l1LossVal", 0.1, ReductionType::SUM)};
  auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              *cpuDevice,
              {},
              Patterns({PreAliasPatternType::PREUNIREPL})});

  // Check the ir
  BOOST_CHECK(ir.opsOfType({"ai_simon", "MyAdd2", 1}).size() == 1);
}
