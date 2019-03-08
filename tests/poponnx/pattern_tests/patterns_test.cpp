#define BOOST_TEST_MODULE PatternsTest

#include <boost/test/unit_test.hpp>
#include <vector>
#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/inputshapeinfo.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/op/nll.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/tensornames.hpp>
#include <poponnx/tensors.hpp>

using namespace poponnx;

BOOST_AUTO_TEST_CASE(PostNRepl_IdentityOp) {
  // clang-format off
  //
  // (*) -> [Identity] -> () -> [Identity] -> (*)
  //     -> [Identity] -> () -> [Identity] ->  ()
  //     -> [Identity] -> () -> [Identity] ->  () -> [Identity] -> (*)
  //
  // where (*) are Anchors should become
  //
  // (*) -> [Identity] -> (*) -> [Identity] -> (*)
  //
  // clang-format on

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();
  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto i1 = builder->addInputTensor(shape);
  std::vector<TensorId> tensorIds{i1};
  // Create a chain of identity ops
  for (int i = 0; i < 6; i++) {
    auto x = aiOnnx.identity({tensorIds[tensorIds.size() - 1]});
    tensorIds.push_back(x);
  }
  builder->addOutputTensor(tensorIds.back());

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto art       = AnchorReturnType("ALL");
  auto dataFlow  = DataFlow(1, {{tensorIds.back(), art}, {tensorIds[2], art}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(tensorIds.back(), "l1LossVal", 0.1)};

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              {},
              Patterns({PreAliasPatternType::POSTNREPL})});

  // Check the ir
  // All but one of the identityOps should have been removed from the ir
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Identity).size() == 2);

  // All but the 1st, 3rd and last tensors should have been removed
  for (int i = 0; i < tensorIds.size(); i++) {
    bool tensorExists = ir.getTensors().contains(tensorIds[i]);
    bool shouldExist  = i == 0 | i == 2 | i == 6;
    BOOST_CHECK(tensorExists == shouldExist);
  }
}

BOOST_AUTO_TEST_CASE(PreUniRepl) {
  // {(i1), (i2)} -> [Add] -> () -> [Pad] -> () -> [Identity] -> (identOut)
  //
  // should become
  //
  // {(i1), (i2)} -> [Add] -> () -> [Identity] -> (identOut)

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto input1 = builder->addInputTensor(shape);
  auto input2 = builder->addInputTensor(shape);

  auto padIn    = aiOnnx.add({input1, input2});
  auto padOut   = aiOnnx.pad({padIn}, {0, 0}, "constant", 0.0);
  auto identOut = aiOnnx.identity({padOut});

  builder->addOutputTensor(identOut);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto dataFlow  = DataFlow(1, {{identOut, AnchorReturnType("ALL")}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(identOut, "l1LossVal", 0.1)};

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              {},
              Patterns({PreAliasPatternType::PREUNIREPL})});

  // Check the ir
  // the PadOp should have been removed
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Pad).size() == 0);
  // padIn should have been removed
  BOOST_CHECK(ir.getTensors().contains(padIn) == false);
}

BOOST_AUTO_TEST_CASE(OpToIdentity) {
  // {(i1), (i2)} -> [Add] -> () -> [Pad] -> () -> [Identity] -> (identOut)
  //
  // should become
  //
  // {(i1), (i2)} -> [Add] -> () -> [Identity] () -> [Identity] -> (identOut)

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto input1 = builder->addInputTensor(shape);
  auto input2 = builder->addInputTensor(shape);

  auto padIn    = aiOnnx.add({input1, input2});
  auto padOut   = aiOnnx.pad({padIn}, {0, 0}, "constant", 0.0);
  auto identOut = aiOnnx.identity({padOut});

  builder->addOutputTensor(identOut);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto dataFlow  = DataFlow(1, {{identOut, AnchorReturnType("ALL")}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(identOut, "l1LossVal", 0.1)};

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              {},
              Patterns({PreAliasPatternType::OPTOIDENTITY})});

  // Check the ir
  // the PadOp should have been replaced with an IdentityOp
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Pad).size() == 0);
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Identity).size() == 2);
}

BOOST_AUTO_TEST_CASE(GatherToIdentity) {
  // {(i1), (i2)} -> [Gather] -> ()
  //
  // should become
  //
  // (i1) -> [Identity] -> ()

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape1{"FLOAT", std::vector<int64_t>{2, 1, 2}};
  TensorInfo shape2{"INT32", std::vector<int64_t>{1}};

  auto input1 = builder->addInputTensor(shape1);
  auto input2 = builder->addInputTensor(shape2);

  auto out = aiOnnx.gather({input1, input2}, 1);

  builder->addOutputTensor(out);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow  = DataFlow(1, {{out, AnchorReturnType("ALL")}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(out, "l1LossVal", 0.1)};

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              {},
              Patterns({PreAliasPatternType::OPTOIDENTITY})});

  // Check the ir
  // the GatherOp should have been replaced with an IdentityOp
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Gather).size() == 0);
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Identity).size() == 1);
}

BOOST_AUTO_TEST_CASE(SplitConvBias) {
  // {(i1), (i2), (i3)} -> [Conv] -> () -> [Identity] -> (identOut)
  //
  // should become
  //
  // {(i1), (i2)} -> [Conv] -> (convOut)
  // {(i3), (convOut)} -> [AddBias] () -> [Identity] -> (identOut)

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{1, 2, 2, 2}};

  auto input1 = builder->addInputTensor(shape);
  auto input2 = builder->addInputTensor(shape);
  auto input3 = builder->addInputTensor(shape);

  auto convOut = aiOnnx.conv(
      {input1, input2, input3}, {1, 1}, 1, {}, {0, 0, 0, 0}, {1, 1});
  auto identOut = aiOnnx.identity({convOut});

  builder->addOutputTensor(identOut);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto dataFlow  = DataFlow(1, {{identOut, AnchorReturnType("ALL")}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(identOut, "l1LossVal", 0.1)};

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              {},
              Patterns({PreAliasPatternType::SPLITCONVBIAS})});

  // Check the ir
  // Input 1 should connect to ConvOp
  // ConvOp should only have 2 inputs
  auto input1Tensor = ir.getTensors().get(input1);
  auto convOp       = input1Tensor->consumers.getOps()[0];
  BOOST_CHECK(convOp->input->n() == 2);

  auto bias = convOp->output->tensor(0)->consumers.getOps()[0];
  BOOST_CHECK(bias->opid == Onnx::CustomOperators::AddBias);

  // Input3 should be consumed only by the AddBiasOp
  auto input3Tensor = ir.getTensors().get(input3);
  BOOST_CHECK(input3Tensor->consumers.getTotal() == 1);
  BOOST_CHECK(bias == input3Tensor->consumers.getOps()[0]);
}

BOOST_AUTO_TEST_CASE(ScaleByOne) {
  // () -> [Scale(by 1.0)] -> ()
  //
  // should become
  //
  // () -> [Identity] -> ()
  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto inp = builder->addInputTensor(shape);

  auto scale = aiOnnx.scale({inp}, 1.0f);

  builder->addOutputTensor(scale);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto art      = AnchorReturnType("ALL");
  auto dataFlow = DataFlow(1, {{scale, art}});

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              {},
              Patterns({PreAliasPatternType::OPTOIDENTITY})});

  // Check the ir
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Identity).size() == 1);
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Scale).size() == 0);
}

BOOST_AUTO_TEST_CASE(ScaleByNegativeOne) {
  // () -> [Scale(by -1.0)] -> ()
  //
  // should become
  //
  // () -> [Negate] -> ()
  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto inp = builder->addInputTensor(shape);

  auto scale = aiOnnx.scale({inp}, -1.0f);

  builder->addOutputTensor(scale);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto art      = AnchorReturnType("ALL");
  auto dataFlow = DataFlow(1, {{scale, art}});

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              {},
              Patterns({PreAliasPatternType::NEGATIVEONESCALE})});

  // Check the ir
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Neg).size() == 1);
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Scale).size() == 0);
}

BOOST_AUTO_TEST_CASE(SubtractArg1GradOp) {
  // () -> [SubtractGradArg1Op] -> ()
  //
  // should become
  //
  // () -> [Negate] -> () -> [ReduceSum] -> ()
  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto input1 = builder->addInputTensor(shape);
  auto input2 = builder->addInputTensor(shape);

  auto subtractOut = aiOnnx.sub({input1, input2});
  auto identOut    = aiOnnx.identity({subtractOut});

  builder->addOutputTensor(identOut);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto art       = AnchorReturnType("ALL");
  auto dataFlow  = DataFlow(1,
                           {{identOut, art},
                            {reservedGradientPrefix() + input1, art},
                            {reservedGradientPrefix() + input2, art}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(identOut, "l1LossVal", 0.1)};

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              {},
              Patterns({PreAliasPatternType::SUBTRACTARG1GRADOP})});

  // Check the ir
  // SubtractArg1Grad should have been replaced with Negate and ReduceSum
  BOOST_CHECK(ir.opsOfType(Onnx::GradOperators::SubArg1Grad).size() == 0);
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Neg).size() == 1);
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::ReduceSum).size() == 1);
}

BOOST_AUTO_TEST_CASE(ReciprocalGradOp) {
  // () -> [ReciprocalGrad] -> ()
  //
  // should become
  //
  // () -> [Square] -> () -> [Reciprocal] -> () -> [Negate] -> ()

  // Build an onnnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto input = builder->addInputTensor(shape);

  auto output = aiOnnx.reciprocal({input});

  builder->addOutputTensor(output);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto art       = AnchorReturnType("ALL");
  auto dataFlow  = DataFlow(1,
                           {{output, art},
                            {reservedGradientPrefix() + input, art},
                            {"l1LossVal", art}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(output, "l1LossVal", 0.1)};

  auto opts = SessionOptions();

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              opts,
              Patterns({PreAliasPatternType::RECIPROCALGRADOP})});

  // Check the ir
  // ReciprocalGradOp should have been replace with SquareOp, ReciprocalOp,
  // NegateOp, and a MulOp
  BOOST_CHECK(ir.opsOfType(Onnx::GradOperators::ReciprocalGrad).size() == 0);
  BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::Square).size() == 1);
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Reciprocal).size() == 2);
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Neg).size() == 1);
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Mul).size() == 1);
}

// When a pattern replaces an op with another op
// it should copy any debug context to it
BOOST_AUTO_TEST_CASE(Attribute_Inheritance) {
  // {(i1), (i2)} -> [Add] -> () -> [Pad] -> () -> [Identity] -> (identOut)
  //
  // should become
  //
  // {(i1), (i2)} -> [Add] -> () -> [Identity] -> () -> [Identity] -> (identOut)

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto input1 = builder->addInputTensor(shape);
  auto input2 = builder->addInputTensor(shape);

  int64_t vgNumber   = 20;
  bool recompute     = true;
  std::string opName = "MyPadOp";

  auto padIn = aiOnnx.add({input1, input2});
  builder->virtualGraph(padIn, vgNumber);

  // Op to be replaced by pattern
  auto padOut = aiOnnx.pad({padIn}, {0, 0}, "constant", 0.0, opName);
  builder->virtualGraph(padOut, vgNumber);
  builder->recomputeOutputInBackwardPass(padOut, recompute);

  auto identOut = aiOnnx.identity({padOut});
  builder->recomputeOutputInBackwardPass(identOut, recompute);
  builder->virtualGraph(identOut, vgNumber);

  builder->addOutputTensor(identOut);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto dataFlow  = DataFlow(1, {{identOut, AnchorReturnType("ALL")}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(identOut, "l1LossVal", 0.1)};

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              {},
              Patterns({PreAliasPatternType::OPTOIDENTITY})});

  // Check the PadOp has been removed
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Pad).size() == 0);

  // Check attributes of the replaced op:
  auto tensor = ir.getTensors().get(padOut);
  auto op     = tensor->getProducer();

  // name
  BOOST_CHECK(op->name().find(opName) == 0);

  // virtual graph id
  BOOST_CHECK(*(op->getVirtualGraphId()) == vgNumber);

  // recomputation
  BOOST_CHECK(*(op->getRecomputeOutput()) == recompute);
}

BOOST_AUTO_TEST_CASE(PadSumPatternTest) {
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2, 2, 2}};

  auto input1 = builder->addInputTensor(shape);
  auto input2 = builder->addInputTensor(shape);

  auto padded1 =
      aiOnnx.pad({input1}, {0, 0, 0, 0, 2, 0}, "constant", 0, "pad1");
  auto padded2 =
      aiOnnx.pad({input2}, {0, 2, 0, 0, 0, 0}, "constant", 0, "pad2");

  auto out = aiOnnx.sum({padded1, padded2}, "sum");

  builder->addOutputTensor(out);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(1, {{out, AnchorReturnType("ALL")}});

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              {},
              Patterns({PreAliasPatternType::PADSUM})});

  // Check the ir
  // Sum op should have been replaced with a concat
  BOOST_CHECK(ir.opsOfType(poponnx::Onnx::AiOnnx::OpSet9::Sum).size() == 0);
  BOOST_CHECK(ir.opsOfType(poponnx::Onnx::AiOnnx::OpSet9::Concat).size() == 1);
}

// Checking PreUniRepl doesn't throw an exception
// when PadOp connects to graph input
BOOST_AUTO_TEST_CASE(PreUniRepl_0) {
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2, 2, 2}};

  auto input1 = builder->addInputTensor(shape);
  auto input2 = builder->addInputTensor(shape);

  auto padded1 =
      aiOnnx.pad({input1}, {0, 0, 0, 0, 0, 0}, "constant", 0, "pad1");
  auto padded2 =
      aiOnnx.pad({input2}, {0, 0, 0, 0, 0, 0}, "constant", 0, "pad2");

  auto out = aiOnnx.concat({padded1, padded2}, 1, "concat");

  builder->addOutputTensor(out);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(1, {{out, AnchorReturnType("ALL")}});

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              {},
              Patterns({PreAliasPatternType::PREUNIREPL})});
}

BOOST_AUTO_TEST_CASE(SplitGatherTest) {
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape1{"FLOAT", std::vector<int64_t>{2, 2}};
  TensorInfo shape2{"INT32", std::vector<int64_t>{2}};

  float dummy[4];
  ConstVoidData t1Data = {dummy, shape1};
  auto input1          = builder->addInitializedInputTensor(t1Data);
  auto input2          = builder->addInputTensor(shape2);

  auto out = aiOnnx.gather({input1, input2}, 1);

  builder->virtualGraph(out, 0);
  builder->addOutputTensor(out);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(1, {{out, AnchorReturnType("ALL")}});

  SessionOptions userOptions;
  userOptions.enableVirtualGraphs      = true;
  userOptions.minimumVirtualGraphCount = 2;

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              userOptions,
              Patterns({PreAliasPatternType::SPLITGATHER})});

  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Gather_1).size() == 2);
}
