// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PatternsTest

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <filereader.hpp>
#include <map>
#include <memory>
#include <string>
#include <testdevice.hpp>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/op/cast.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/nll.hpp>
#include <popart/patterns/removeunnecessarylossgradcastpattern.hpp>
#include <popart/sgd.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>

#include "popart/builder.gen.hpp"
#include "popart/datatype.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"
#include "popart/vendored/optional.hpp"

using namespace popart;

BOOST_AUTO_TEST_CASE(Atan2GradOps) {
  // () -> [Atan2GradArg0] -> ()
  // () -> [Atan2GradArg1] -> ()
  // should become
  //
  // Combined by transform
  // () -> [Square] -> [Add] -> (*)
  // () -> [Square] -^
  //
  // For arg 0 (*) -> [Div] -> [Reduce] -> ()
  // For arg 1 (*) -> [Div] -> [Neg] -> [Reduce] -> ()

  // Build an onnx model
  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto input1 = builder->addInputTensor(shape);
  auto input2 = builder->addInputTensor(shape);

  auto atan2Out = aiGraphcore.atan2({input1, input2});
  auto l1       = builder->aiGraphcoreOpset1().l1loss({atan2Out}, 0.1);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto art       = AnchorReturnType("All");
  auto dataFlow  = DataFlow(1,
                           {{atan2Out, art},
                            {reservedGradientPrefix() + input1, art},
                            {reservedGradientPrefix() + input2, art}});
  auto optimizer = ConstSGD(0.01);
  auto device    = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              l1,
              &optimizer,
              *device,
              {},
              Patterns::create({"Atan2Arg0GradOp", "Atan2Arg1GradOp"})
                  .enableRuntimeAsserts(false)});

  // Check the ir
  // Sum of op coun as above
  BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::Square).size() == 2);
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Add).size() == 1);
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Div).size() == 2);
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Neg).size() == 1);
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::ReduceSum).size() == 2);
}

struct RemoveUnnecessaryLossGradCastSetup {
  RemoveUnnecessaryLossGradCastSetup() : mainGraph(ir.getMainGraph()) {}

  void createMainGraph() {
    // Add input tensors to the graph
    mainGraph.addInput(fp32LossScale, fp32Info);
    mainGraph.addInput(i16Labels, i16Info);
    mainGraph.addInput(fp16Probs, fp16Info);

    // Add the CastOp
    castOp = mainGraph.createConnectedOp<CastOp>(
        {{CastOp::getInIndex(), fp32LossScale}},
        {{CastOp::getOutIndex(), fp16LossScale}},
        Onnx::Operators::Cast_6,
        DataType::FLOAT16,
        Op::Settings(mainGraph, ""));

    // Add the NllGradOp
    nllGradOp = mainGraph.createConnectedOp<NllGradOp>(
        {{NllGradOp::getProbsInIndex(), fp16Probs},
         {NllGradOp::getLabelInIndex(), i16Labels},
         {NllGradOp::getGradInIndex(), fp16LossScale}},
        {{NllGradOp::getOutIndex(), fp16Grad}},
        tmpLoss,
        nonstd::optional<int>(),
        ReductionType::Mean,
        false,
        Op::Settings(mainGraph, ""));

    // Mark fp16Grad as output
    mainGraph.markAsOutput(fp16Grad);
  }

  Ir ir;
  Graph &mainGraph;
  CastOp *castOp;
  NllGradOp *nllGradOp;

  // Input to CastOp
  TensorId fp32LossScale = {"fp32LossScale"};
  // Output from CastOp
  TensorId fp16LossScale = {"fp16LossScale"};
  // Input to NllLossGradOp
  TensorId fp16Probs = {"fp16Probs"};
  TensorId i16Labels = {"i16Labels"};
  // NllLossGradOp placeholder
  TensorId tmpLoss = {"tmpLoss"};
  // Output form NllLossGradOp
  TensorId fp16Grad = {"fp16Grad"};

  // The pattern to check
  RemoveUnnecessaryLossGradCast pattern = RemoveUnnecessaryLossGradCast();

  // Define the TensorInfos and the TensorIds
  TensorInfo fp32Info{DataType::FLOAT, std::vector<int64_t>{1}};
  TensorInfo i16Info{DataType::INT16, std::vector<int64_t>{2}};
  TensorInfo fp16Info{DataType::FLOAT16, std::vector<int64_t>{2, 2}};
};

BOOST_FIXTURE_TEST_CASE(RemoveUnnecessaryLossGradCastMatchCastTest,
                        RemoveUnnecessaryLossGradCastSetup) {
  // Test that the match function for RemoveUnnecessaryLossGradCast fails when
  // there is no CastOp

  // Create the main graph
  createMainGraph();

  // Check that we get a match on the original nllGradOp
  BOOST_CHECK(pattern.matches(nllGradOp));

  // Replace the CastOp with IdentityOp
  castOp->disconnectAllInputs();
  castOp->disconnectAllOutputs();
  mainGraph.eraseOp(castOp->id);
  mainGraph.createConnectedOp<IdentityOp>(
      {{IdentityOp::getInIndex(), fp32LossScale}},
      {{IdentityOp::getOutIndex(), fp16LossScale}},
      Onnx::Operators::Identity_1,
      Op::Settings(mainGraph, ""));

  BOOST_CHECK(!pattern.matches(nllGradOp));
}

BOOST_FIXTURE_TEST_CASE(RemoveUnnecessaryLossGradCastMatchHasProducerTest,
                        RemoveUnnecessaryLossGradCastSetup) {
  // Test that the match function for RemoveUnnecessaryLossGradCast fails when
  // the loss scale tensor has no producer

  // Create the main graph
  createMainGraph();

  // Check that we get a match on the original nllGradOp
  BOOST_CHECK(pattern.matches(nllGradOp));

  // Ensure the lossScaleTensor doesn't have a producer
  castOp->disconnectAllOutputs();
  BOOST_CHECK(!mainGraph.getTensor(fp16LossScale)->hasProducer());
  BOOST_CHECK(!pattern.matches(nllGradOp));
}

BOOST_FIXTURE_TEST_CASE(RemoveUnnecessaryLossGradCastMatchNelmsTest,
                        RemoveUnnecessaryLossGradCastSetup) {
  // Test that the match function for RemoveUnnecessaryLossGradCast fails when
  // the loss scale tensor has more than one element

  // Create a tensor with more than one element
  TensorInfo fp16Info2Elms{DataType::FLOAT16, std::vector<int64_t>{2}};
  TensorId fp16LossScaleW2Elements = {"fp16LossScaleW2Elements"};

  // Add tensors to the graph
  mainGraph.addInput(fp16LossScaleW2Elements, fp16Info2Elms);
  mainGraph.addInput(i16Labels, i16Info);
  mainGraph.addInput(fp16Probs, fp16Info2Elms);

  // Setup the new nllGradOp
  nllGradOp = mainGraph.createConnectedOp<NllGradOp>(
      {{NllGradOp::getProbsInIndex(), fp16Probs},
       {NllGradOp::getLabelInIndex(), i16Labels},
       {NllGradOp::getGradInIndex(), fp16LossScaleW2Elements}},
      {{NllGradOp::getOutIndex(), fp16Grad}},
      tmpLoss,
      nonstd::optional<int>(),
      ReductionType::Mean,
      false,
      Op::Settings(mainGraph, ""));

  BOOST_CHECK(!pattern.matches(nllGradOp));
}

BOOST_FIXTURE_TEST_CASE(RemoveUnnecessaryLossGradCastMatchTypeTest,
                        RemoveUnnecessaryLossGradCastSetup) {
  // Test that the match function for RemoveUnnecessaryLossGradCast fails when
  // the loss scale tensor is not of FLOAT16

  // Add tensors to the graph
  mainGraph.addInput(fp32LossScale, fp32Info);
  mainGraph.addInput(i16Labels, i16Info);
  mainGraph.addInput(fp16Probs, fp16Info);

  // Setup the new nllGradOp
  nllGradOp = mainGraph.createConnectedOp<NllGradOp>(
      {{NllGradOp::getProbsInIndex(), fp16Probs},
       {NllGradOp::getLabelInIndex(), i16Labels},
       {NllGradOp::getGradInIndex(), fp32LossScale}},
      {{NllGradOp::getOutIndex(), fp16Grad}},
      tmpLoss,
      nonstd::optional<int>(),
      ReductionType::Mean,
      false,
      Op::Settings(mainGraph, ""));

  BOOST_CHECK(!pattern.matches(nllGradOp));
}

BOOST_FIXTURE_TEST_CASE(RemoveUnnecessaryLossGradCastNotMixedPrecisionTest,
                        RemoveUnnecessaryLossGradCastSetup) {
  // Test that the match function for RemoveUnnecessaryLossGradCast fails when
  // it's not a mixed precision loss grad

  // Add tensors to the graph
  TensorId fp32Probs = {"fp32Probs"};
  mainGraph.addInput(fp32LossScale, fp32Info);
  mainGraph.addInput(fp32Probs, fp32Info);

  auto identityOp = mainGraph.createConnectedOp<IdentityOp>(
      {{IdentityOp::getInIndex(), fp32LossScale}},
      {{IdentityOp::getOutIndex(), fp16LossScale}},
      Onnx::Operators::Identity_1,
      Op::Settings(mainGraph, ""));

  BOOST_CHECK(!pattern.matches(identityOp));
}

BOOST_FIXTURE_TEST_CASE(RemoveUnnecessaryLossGradCastTest,
                        RemoveUnnecessaryLossGradCastSetup) {
  // clang-format off
  // Check that
  //
  // {(fp32LossScale)} -> [Cast] -> (fp16LossScale) -> [NllLossGradOp] -> (fp16Grad)
  // {(fp16Probs)}     --------------------------------------^
  //
  // becomes
  //
  // {(fp32LossScale)} -> [NllLossGradOp] -> (fp16Grad)
  // {(fp16Probs)}     ---------^
  //
  // when applying RemoveUnnecessaryLossGradCast
  // clang-format on

  // Create the main graph
  createMainGraph();

  // Check that the IR contains the CastOp and fp16LossScale
  BOOST_CHECK(castOp->input->tensors().size() == 1);
  BOOST_CHECK(castOp->output->tensors().size() == 1);
  BOOST_CHECK(ir.getMainGraphTensors().contains(fp16LossScale) == true);
  // Check the input and output of NllGradOp
  auto inputs  = nllGradOp->input->tensorIdMap();
  auto outputs = nllGradOp->output->tensorIdMap();
  BOOST_CHECK(inputs.at(NllGradOp::getProbsInIndex()) == fp16Probs);
  BOOST_CHECK(inputs.at(NllGradOp::getLabelInIndex()) == i16Labels);
  BOOST_CHECK(inputs.at(NllGradOp::getGradInIndex()) == fp16LossScale);
  BOOST_CHECK(outputs.at(NllGradOp::getOutIndex()) == fp16Grad);
  // Check that the output has the expected data type
  BOOST_CHECK(mainGraph.getTensors().get(fp16Grad)->info.dataType() ==
              DataType::FLOAT16);

  // Apply the pattern
  RemoveUnnecessaryLossGradCast alsPattern;
  ir.applyPreAliasPattern(&alsPattern, mainGraph);

  // Check that the CastOp has no connected tensors and that fp16LossScale has
  // been removed from the IR
  BOOST_CHECK(castOp->input->tensors().size() == 0);
  BOOST_CHECK(castOp->output->tensors().size() == 0);
  BOOST_CHECK(ir.getMainGraphTensors().contains(fp16LossScale) == false);
  // Check the input and output of NllGradOp
  inputs  = nllGradOp->input->tensorIdMap();
  outputs = nllGradOp->output->tensorIdMap();
  BOOST_CHECK(inputs.at(NllGradOp::getProbsInIndex()) == fp16Probs);
  BOOST_CHECK(inputs.at(NllGradOp::getLabelInIndex()) == i16Labels);
  BOOST_CHECK(inputs.at(NllGradOp::getGradInIndex()) == fp32LossScale);
  BOOST_CHECK(outputs.at(NllGradOp::getOutIndex()) == fp16Grad);
  // Check that the output has the expected data type
  BOOST_CHECK(mainGraph.getTensors().get(fp16Grad)->info.dataType() ==
              DataType::FLOAT16);
}

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
  auto l1 = builder->aiGraphcoreOpset1().l1loss({tensorIds.back()}, 0.1);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto art       = AnchorReturnType("All");
  auto dataFlow  = DataFlow(1, {{tensorIds.back(), art}, {tensorIds[2], art}});
  auto optimizer = ConstSGD(0.01);
  auto device    = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              l1,
              &optimizer,
              *device,
              {},
              Patterns::create({"PostNRepl"}).enableRuntimeAsserts(false)});

  // Check the ir
  // All but one of the identityOps should have been removed from the ir
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Identity).size() == 2);

  // All but the 1st, 3rd and last tensors should have been removed
  for (int i = 0; i < tensorIds.size(); i++) {
    bool tensorExists = ir.getMainGraphTensors().contains(tensorIds[i]);
    bool shouldExist  = (i == 0) | (i == 2) | (i == 6);
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

  auto l1 = builder->aiGraphcoreOpset1().l1loss({identOut}, 0.1);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto dataFlow  = DataFlow(1, {{identOut, AnchorReturnType("All")}});
  auto optimizer = ConstSGD(0.01);
  auto device    = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              l1,
              &optimizer,
              *device,
              {},
              Patterns::create({"PreUniRepl"}).enableRuntimeAsserts(false)});

  // Check the ir
  // the PadOp should have been removed
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Pad).size() == 0);
  // padIn should have been removed
  BOOST_CHECK(ir.getMainGraphTensors().contains(padIn) == false);
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
  auto l1       = builder->aiGraphcoreOpset1().l1loss({identOut}, 0.1);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto dataFlow  = DataFlow(1, {{identOut, AnchorReturnType("All")}});
  auto optimizer = ConstSGD(0.01);
  auto device    = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              l1,
              &optimizer,
              *device,
              {},
              Patterns::create({"OpToIdentity"}).enableRuntimeAsserts(false)});

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
  auto l1  = builder->aiGraphcoreOpset1().l1loss({out}, 0.1);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow  = DataFlow(1, {{out, AnchorReturnType("All")}});
  auto optimizer = ConstSGD(0.01);
  auto device    = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              l1,
              &optimizer,
              *device,
              {},
              Patterns::create({"OpToIdentity"}).enableRuntimeAsserts(false)});

  // Check the ir
  // the GatherOp should have been replaced with an IdentityOp
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Gather).size() == 0);
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Identity).size() == 1);
}

BOOST_AUTO_TEST_CASE(ScaleByOne) {
  // () -> [Scale(by 1.0)] -> ()
  //
  // should become
  //
  // () -> [Identity] -> ()
  // Build an onnx model
  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto inp = builder->addInputTensor(shape);

  auto scale = aiGraphcore.scale({inp}, 1.0f);

  builder->addOutputTensor(scale);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto art      = AnchorReturnType("All");
  auto dataFlow = DataFlow(1, {{scale, art}});
  auto device   = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              *device,
              {},
              Patterns::create({"OpToIdentity"}).enableRuntimeAsserts(false)});

  // Check the ir
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Identity).size() == 1);
  BOOST_CHECK(ir.opsOfType(Onnx::AiGraphcore::OpSet1::Scale).size() == 0);
}

BOOST_AUTO_TEST_CASE(ScaleByNegativeOne) {
  // () -> [Scale(by -1.0)] -> ()
  //
  // should become
  //
  // () -> [Negate] -> ()
  // Build an onnx model
  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto inp = builder->addInputTensor(shape);

  auto scale = aiGraphcore.scale({inp}, -1.0f);

  builder->addOutputTensor(scale);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto art      = AnchorReturnType("All");
  auto dataFlow = DataFlow(1, {{scale, art}});
  auto device   = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare(
      {modelProto,
       InputShapeInfo(),
       dataFlow,
       {},
       nullptr,
       *device,
       {},
       Patterns::create({"NegativeOneScale"}).enableRuntimeAsserts(false)});

  // Check the ir
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Neg).size() == 1);
  BOOST_CHECK(ir.opsOfType(Onnx::AiGraphcore::OpSet1::Scale).size() == 0);
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
  auto l1          = builder->aiGraphcoreOpset1().l1loss({identOut}, 0.1);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto art       = AnchorReturnType("All");
  auto dataFlow  = DataFlow(1,
                           {{identOut, art},
                            {reservedGradientPrefix() + input1, art},
                            {reservedGradientPrefix() + input2, art}});
  auto optimizer = ConstSGD(0.01);
  auto device    = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare(
      {modelProto,
       InputShapeInfo(),
       dataFlow,
       l1,
       &optimizer,
       *device,
       {},
       Patterns::create({"SubtractArg1GradOp"}).enableRuntimeAsserts(false)});

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

  auto l1 = builder->aiGraphcoreOpset1().l1loss({output}, 0.1);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto art      = AnchorReturnType("ALL");
  auto dataFlow = DataFlow(
      1, {{output, art}, {reservedGradientPrefix() + input, art}, {l1, art}});
  auto optimizer = ConstSGD(0.01);

  auto opts   = SessionOptions();
  auto device = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare(
      {modelProto,
       InputShapeInfo(),
       dataFlow,
       l1,
       &optimizer,
       *device,
       opts,
       Patterns::create({"ReciprocalGradOp"}).enableRuntimeAsserts(false)});

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
  std::string opName = "MyPadOp";

  auto recompute = RecomputeType::Recompute;

  auto padIn = aiOnnx.add({input1, input2});
  builder->virtualGraph(padIn, vgNumber);

  // Op to be replaced by pattern
  auto padOut = aiOnnx.pad({padIn}, {0, 0}, "constant", 0.0, opName);
  builder->virtualGraph(padOut, vgNumber);
  builder->recomputeOutputInBackwardPass(padOut, recompute);

  auto identOut = aiOnnx.identity({padOut});
  builder->recomputeOutputInBackwardPass(identOut, recompute);
  builder->virtualGraph(identOut, vgNumber);

  auto l1 = builder->aiGraphcoreOpset1().l1loss({identOut}, 0.1);
  builder->virtualGraph(l1, vgNumber);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto dataFlow  = DataFlow(1, {{identOut, AnchorReturnType("All")}});
  auto optimizer = ConstSGD(0.01);

  auto device = createTestDevice(TEST_TARGET);

  SessionOptions opts;
  opts.virtualGraphMode = VirtualGraphMode::Manual;

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              l1,
              &optimizer,
              *device,
              opts,
              Patterns::create({"OpToIdentity"}).enableRuntimeAsserts(false)});

  // Check the PadOp has been removed
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Pad).size() == 0);

  // Check attributes of the replaced op:
  auto tensor = ir.getMainGraphTensors().get(padOut);
  auto op     = tensor->getProducer();

  // name
  BOOST_CHECK(op->name().find(opName) == 0);

  // virtual graph id
  BOOST_CHECK(op->getVirtualGraphId() == vgNumber);

  // recomputation
  if (recompute == RecomputeType::Recompute) {
    BOOST_CHECK(op->settings.recomputeType == RecomputeType::Recompute);
  } else {
    BOOST_CHECK(op->settings.recomputeType == RecomputeType::Checkpoint ||
                op->settings.recomputeType == RecomputeType::Undefined);
  }
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
  auto dataFlow = DataFlow(1, {{out, AnchorReturnType("All")}});
  auto device   = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              *device,
              {},
              Patterns::create({"PadSum"}).enableRuntimeAsserts(false)});

  // Check the ir
  // Sum op should have been replaced with a concat
  BOOST_CHECK(ir.opsOfType(popart::Onnx::AiOnnx::OpSet9::Sum).size() == 0);
  BOOST_CHECK(ir.opsOfType(popart::Onnx::AiOnnx::OpSet9::Concat).size() == 1);
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
  auto dataFlow = DataFlow(1, {{out, AnchorReturnType("All")}});
  auto device   = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              *device,
              {},
              Patterns::create({"PreUniRepl"}).enableRuntimeAsserts(false)});
}

BOOST_AUTO_TEST_CASE(SumToAddTest) {
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2, 2}};

  auto input1 = builder->addInputTensor(shape);
  auto input2 = builder->addInputTensor(shape);

  auto out = aiOnnx.sum({input1, input2});

  builder->addOutputTensor(out);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(1, {{out, AnchorReturnType("All")}});

  SessionOptions userOptions;

  std::map<std::string, std::string> deviceOpts{{"numIPUs", "2"}};
  auto device = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              *device,
              userOptions,
              Patterns::create({"SumToAdd"}).enableRuntimeAsserts(false)});

  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Sum_8).size() == 0);
  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Add_7).size() == 1);
}

BOOST_AUTO_TEST_CASE(Expm1GradOpTest) {
  // Build an onnnx model
  auto builder = Builder::create();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};
  auto input = builder->addInputTensor(shape);

  auto output = builder->aiGraphcoreOpset1().expm1({input});
  auto l1     = builder->aiGraphcoreOpset1().l1loss({output}, 0.1);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto art      = AnchorReturnType("ALL");
  auto dataFlow = DataFlow(
      1, {{output, art}, {reservedGradientPrefix() + input, art}, {l1, art}});
  auto optimizer = ConstSGD(0.01);

  auto opts   = SessionOptions();
  auto device = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              l1,
              &optimizer,
              *device,
              opts,
              Patterns::create({"Expm1GradOp"}).enableRuntimeAsserts(false)});

  // Check the ir
  // Expm1Grad should have been replace with Mul and Add.
  BOOST_CHECK(ir.opsOfType(Onnx::GradOperators::Expm1Grad).size() == 0);
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Mul).size() == 1);
  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Add_7).size() == 1);
}

BOOST_AUTO_TEST_CASE(Log1pGradOpTest) {
  // Build an onnnx model
  auto builder = Builder::create();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};
  auto input = builder->addInputTensor(shape);

  auto output = builder->aiGraphcoreOpset1().log1p({input});
  auto l1     = builder->aiGraphcoreOpset1().l1loss({output}, 0.1);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto art      = AnchorReturnType("ALL");
  auto dataFlow = DataFlow(
      1, {{output, art}, {reservedGradientPrefix() + input, art}, {l1, art}});
  auto optimizer = ConstSGD(0.01);

  auto opts   = SessionOptions();
  auto device = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              l1,
              &optimizer,
              *device,
              opts,
              Patterns::create({"Log1pGradOp"}).enableRuntimeAsserts(false)});

  // Check the ir
  // Log1pGrad should have been replace with Div and Add.
  BOOST_CHECK(ir.opsOfType(Onnx::GradOperators::Log1pGrad).size() == 0);
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Div).size() == 1);
  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Add_7).size() == 1);
}
