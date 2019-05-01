#include <poponnx/graph.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/div.hpp>
#include <poponnx/op/scale.hpp>
#include <poponnx/op/sqrt.hpp>
#include <poponnx/patterns/pattern.hpp>
#include <poponnx/patterns/sqrtgradoppattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

bool SqrtGradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<SqrtGradOp>();
}

std::vector<const Tensor *> SqrtGradOpPattern::touches(Op *) const {
  return {};
}

bool SqrtGradOpPattern::apply(Op *op) const {
  auto grad_in  = op->inTensor(SqrtGradOp::getGradInIndex());
  auto fwd_out  = op->inTensor(SqrtGradOp::getFwdOutInIndex());
  auto grad_out = op->outTensor(SqrtGradOp::getOutIndex());

  // create the new ops
  auto scale = dynamic_cast<ScaleOp *>(
      makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Scale, op));
  scale->setScaleFactor(2.0f);
  auto div = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Div, op);

  // Remove the DivArg0GradOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->getGraph().eraseOp(op->id);

  // Connect up the new ops
  scale->connectInTensor(ScaleOp::getInIndex(), fwd_out->id);
  scale->createAndConnectOutTensor(ScaleOp::getOutIndex(),
                                   createIntermediateTensorId(fwd_out->id));
  scale->setup();

  div->connectInTensor(DivOp::getArg0InIndex(), grad_in->id);
  div->connectInTensor(DivOp::getArg1InIndex(),
                       scale->outTensor(ScaleOp::getOutIndex())->id);
  div->connectOutTensor(DivOp::getOutIndex(), grad_out->id);

  return true;
}

namespace {
static PatternCreator<SqrtGradOpPattern>
    SqrtGradOpPattern(PreAliasPatternType::SQRTGRADOP, "SqrtGradOp");
}

} // namespace poponnx
