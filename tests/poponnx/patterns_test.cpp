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

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto i1 = builder->addInputTensor(shape);
  std::vector<TensorId> tensorIds{i1};
  // Create a chain of identity ops
  for (int i = 0; i < 6; i++) {
    auto x = builder->identity({tensorIds[tensorIds.size() - 1]});
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
              Patterns({PatternType::POSTNREPL})});

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

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto input1 = builder->addInputTensor(shape);
  auto input2 = builder->addInputTensor(shape);

  auto padIn    = builder->add({input1, input2});
  auto padOut   = builder->pad({padIn}, "constant", {0, 0}, 0.0);
  auto identOut = builder->identity({padOut});

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
              Patterns({PatternType::PREUNIREPL})});

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

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto input1 = builder->addInputTensor(shape);
  auto input2 = builder->addInputTensor(shape);

  auto padIn    = builder->add({input1, input2});
  auto padOut   = builder->pad({padIn}, "constant", {0, 0}, 0.0);
  auto identOut = builder->identity({padOut});

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
              Patterns({PatternType::OPTOIDENTITY})});

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

  TensorInfo shape1{"FLOAT", std::vector<int64_t>{2, 1, 2}};
  TensorInfo shape2{"INT32", std::vector<int64_t>{1}};

  auto input1 = builder->addInputTensor(shape1);
  auto input2 = builder->addInputTensor(shape2);

  auto out = builder->gather({input1, input2}, 1);

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
              Patterns({PatternType::OPTOIDENTITY})});

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

  TensorInfo shape{"FLOAT", std::vector<int64_t>{1, 2, 2, 2}};

  auto input1 = builder->addInputTensor(shape);
  auto input2 = builder->addInputTensor(shape);
  auto input3 = builder->addInputTensor(shape);

  auto convOut = builder->convolution(
      {input1, input2, input3}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, false);
  auto identOut = builder->identity({convOut});

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
              Patterns({PatternType::SPLITCONVBIAS})});

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

BOOST_AUTO_TEST_CASE(SubtractArg1GradOp) {
  // () -> [SubtractGradArg1Op] -> ()
  //
  // should become
  //
  // () -> [Negate] -> () -> [ReduceSum] -> ()
  // Build an onnx model
  auto builder = Builder::create();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto input1 = builder->addInputTensor(shape);
  auto input2 = builder->addInputTensor(shape);

  auto subtractOut = builder->sub({input1, input2});
  auto identOut    = builder->identity({subtractOut});

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
              Patterns({PatternType::SUBTRACTARG1GRADOP})});

  // Check the ir
  // SubtractArg1Grad should have been replaced with Negate and ReduceSum
  BOOST_CHECK(ir.opsOfType(Onnx::GradOperators::SubArg1Grad).size() == 0);
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Neg).size() == 1);
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::ReduceSum).size() == 1);
}

BOOST_AUTO_TEST_CASE(SoftmaxGradDirect) {
  // (label), (probs) -> [NLLGrad]
  // [NllGrad] -> (d_probs)
  // (d_probs), (probs) -> [SoftmaxGrad] -> (d_acts)
  //
  // should become
  //
  // (label), (probs) -> [SoftmaxGradDirect] -> (d_acts)

  // Build an onnx model
  auto builder = Builder::create();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto input1 = builder->addInputTensor(shape);
  auto input2 = builder->addInputTensor(shape);

  auto identOut   = builder->identity({input1});
  auto softmaxOut = builder->softmax({identOut});

  builder->addOutputTensor(softmaxOut);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto art       = AnchorReturnType("ALL");
  auto dataFlow  = DataFlow(1,
                           {{softmaxOut, art},
                            {reservedGradientPrefix() + input1, art},
                            {"nllLossVal", art}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new NllLoss(softmaxOut, input2, "nllLossVal")};

  auto opts      = SessionOptions();
  opts.exportDot = true;

  Ir ir;
  ir.prepare(
      {modelProto,
       InputShapeInfo(),
       dataFlow,
       losses,
       &optimizer,
       opts,
       Patterns({PatternType::PREUNIREPL, PatternType::SOFTMAXGRADDIRECT})});

  // Check the ir
  // NllGradOp and SoftmaxGradOp should have been replaced with
  // SoftmaxGradDirectOp
  BOOST_CHECK(ir.opsOfType(Onnx::CustomGradOperators::NllGrad).size() == 0);
  BOOST_CHECK(ir.opsOfType(Onnx::GradOperators::SoftmaxGrad).size() == 0);
  BOOST_CHECK(
      ir.opsOfType(Onnx::CustomGradOperators::SoftmaxGradDirect).size() == 1);
}

// where we test that a series of Relus is converted
// into InplaceRelus by the Inplace0 pattern.
BOOST_AUTO_TEST_CASE(Inplace0_series) {

  // Consider the SERIES of Relu Ops:
  //
  // (in0) -> [Relu] -> (h0)
  //       -> [Relu] -> (h1)
  //       -> [Relu] -> (preId)
  //       -> [Identity] -> (out),
  //
  // with (out) as an anchor tensor. This should become,
  //
  // (in0) -> {[ReluInplace], [ReluInplace], [ReluInplace]}
  // (in0) -> [Identity] -> (out).

  // Build an onnx model
  auto builder = Builder::create();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{1}};
  auto in0   = builder->addInputTensor(shape);
  auto h0    = builder->relu({in0});
  auto h1    = builder->relu({h0});
  auto preId = builder->relu({h1});
  auto out   = builder->identity({preId});
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
              Patterns({PatternType::INPLACE0})});

  // Check the ir
  // All the Relus have been optimised out,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu).size() == 0);
  // and have been replaced with ReluInplace.
  BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ReluInplace).size() == 3);
}

// where we test that with Relus is parallel, exactly 1 of
// them is converted into an InplaceRelu by the Inplace0 pattern.
BOOST_AUTO_TEST_CASE(Inplace0_parallel) {

  // Consider the Relu Ops in PARALLEL
  //
  //           | -- [Relu] -- (h0) -- |
  //           |                      | --- [Add] -- (h3) -|
  // (in0) >---| -- [Relu] -- (h1) -- |                    |
  //           |                                           | -> [Add] -- (out)
  //           | -- [Relu] -- (h2) ----------------------- |
  //
  // We can make the first relu in-place, but then stall as
  // an in-place op must run after all other consumers (and
  // therefore there can only be one in-place consumer here). So, we expect:
  //
  //           | -------------------- |
  //           |                      |
  //           | -- [ReluInplace]     |
  //           |                      | --- [Add] -- (h3) -|
  // (in0) >---| -- [Relu] -- (h1) -- |                    |
  //           |                                           | -> [Add] -- (out)
  //           | -- [Relu] -- (h2) ----------------------- |
  //
  //

  // Build an onnx model
  auto builder = Builder::create();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{1}};
  auto in0 = builder->addInputTensor(shape);
  auto h0  = builder->relu({in0});
  auto h1  = builder->relu({in0});
  auto h2  = builder->relu({in0});
  auto h3  = builder->add({h0, h1});
  auto out = builder->add({h2, h3});
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
              Patterns({PatternType::INPLACE0})});

  // Check the ir
  // All the Relus have been optimised out,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu).size() == 3 - 1);
  // and have been replaced with ReluInplace.
  BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ReluInplace).size() == 1);
}

BOOST_AUTO_TEST_CASE(ReciprocalGradOp) {
  // () -> [ReciprocalGrad] -> ()
  //
  // should become
  //
  // () -> [Square] -> () -> [Reciprocal] -> () -> [Negate] -> ()

  // Build an onnnx model
  auto builder = Builder::create();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto input = builder->addInputTensor(shape);

  auto output = builder->reciprocal({input});

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

  auto opts      = SessionOptions();
  opts.exportDot = true;

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              opts,
              Patterns({PatternType::RECIPROCALGRADOP})});

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

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto input1 = builder->addInputTensor(shape);
  auto input2 = builder->addInputTensor(shape);

  int64_t vgNumber   = 20;
  bool recompute     = true;
  std::string opName = "MyPadOp";

  auto padIn = builder->add({input1, input2});
  builder->virtualGraph(padIn, vgNumber);

  // Op to be replaced by pattern
  auto padOut = builder->pad({padIn}, "constant", {0, 0}, 0.0, opName);
  builder->virtualGraph(padOut, vgNumber);
  builder->recomputeOutputInBackwardPass(padOut, recompute);

  auto identOut = builder->identity({padOut});
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
              Patterns({PatternType::OPTOIDENTITY})});

  // Check the PadOp has been removed
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Pad).size() == 0);

  // Check attributes of the replaced op:
  auto tensor = ir.getTensors().get(padOut);
  auto op     = tensor->getProducer();

  // name
  BOOST_CHECK(op->name().find(opName) == 0);

  // virtual graph id
  BOOST_CHECK(op->nAtts.at(sVirtualGraphAttribute)->i() == vgNumber);

  // recomputation
  BOOST_CHECK(op->nAtts.at(sRecomputeOutputAttribute)->i() == recompute);
}
