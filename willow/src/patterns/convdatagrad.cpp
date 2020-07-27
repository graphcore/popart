// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/addbias.hpp>
#include <popart/op/conv.hpp>
#include <popart/patterns/convdatagrad.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/util.hpp>

namespace popart {

bool ConvDataGradPattern::matches(Op *op) const {
  return (op->opid == Onnx::GradOperators::ConvDataGrad ||
          op->opid == Onnx::GradOperators::MultiConvDataGrad);
}

std::vector<const Tensor *> ConvDataGradPattern::touches(Op *) const {
  return {};
}

bool ConvDataGradPattern::apply(Op *op) const {

  const auto gradOp = dynamic_cast<MultiConvDataGradBaseOp *>(op);

  OperatorIdentifier gradOpId = Onnx::CustomOperators::MultiConv_1;
  if (op->opid == Onnx::GradOperators::ConvDataGrad) {
    gradOpId = Onnx::Operators::Conv_1;
  }

  auto conv =
      dynamic_cast<MultiConvBaseOp *>(makeReplacementOpInIr(gradOpId, op));

  conv->setConvOptions(gradOp->getConvOptions());

  auto numConvs = gradOp->numConvs();
  for (int i = 0; i < numConvs; i++) {
    auto weights_in =
        gradOp->inTensor(MultiConvDataGradBaseOp::getWeightsInIndex(i));
    auto gradConvIn_out =
        gradOp->inTensor(MultiConvDataGradBaseOp::getGradConvolvedInIndex(i));
    auto grad_out = gradOp->outTensor(MultiConvDataGradBaseOp::getOutIndex(i));
    gradOp->disconnectInTensor(MultiConvDataGradBaseOp::getWeightsInIndex(i),
                               weights_in);
    gradOp->disconnectInTensor(
        MultiConvDataGradBaseOp::getGradConvolvedInIndex(i), gradConvIn_out);
    gradOp->disconnectOutTensor(grad_out);

    auto flip = dynamic_cast<ConvFlipWeightsOp *>(
        makeReplacementOpInIr(Onnx::CustomOperators::ConvFlipWeights, op));

    // Inherit the options from the forward op
    flip->setConvOptions(gradOp->getConvOptions());

    // Configure the flip weight op
    flip->connectInTensor(ConvFlipWeightsOp::getInIndex(), weights_in->id);
    flip->createAndConnectOutTensor(
        ConvFlipWeightsOp::getOutIndex(),
        weights_in->getIr().createIntermediateTensorId(weights_in->id));

    flip->setup();
    flip->setParameters(gradOp->getParameters(i));

    // Configure the conv op for the bwd pass
    conv->connectInTensor(
        MultiConvBaseOp::getWeightsInIndex(i),
        flip->outTensor(ConvFlipWeightsOp::getOutIndex())->id);
    conv->connectInTensor(MultiConvBaseOp::getDataInIndex(i),
                          gradConvIn_out->id);
    conv->connectOutTensor(MultiConvBaseOp::getOutIndex(i), grad_out->id);
  }

  conv->setupFromDataGradOp(gradOp);

  // Remove the MultiConvGradOp
  op->getGraph().eraseOp(op->id);

  return true;
}

namespace {
static PatternCreator<ConvDataGradPattern>
    convDataGradPattern(PreAliasPatternType::ConvDataGrad, "ConvDataGrad");
}

} // namespace popart
