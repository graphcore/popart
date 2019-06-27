#include <memory>
#include <poponnx/graph.hpp>
#include <poponnx/op/cos.hpp>
#include <poponnx/op/div.hpp>
#include <poponnx/op/sin.hpp>
#include <poponnx/op/tan.hpp>
#include <poponnx/patterns/tantosinovercospattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

bool TanToSinOverCosPattern::matches(Op *op) const {
  return op->isConvertibleTo<TanOp>();
}

std::vector<const Tensor *> TanToSinOverCosPattern::touches(Op *) const {
  return {};
}

// fwd_out = sin(fwd_in) / cos(fwd_in)
bool TanToSinOverCosPattern::apply(Op *op) const {
  auto fwd_in  = op->inTensor(TanOp::getInIndex());
  auto fwd_out = op->outTensor(TanOp::getOutIndex());

  // create the new ops
  auto sin = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Sin, op);
  auto cos = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Cos, op);
  auto div = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Div, op);

  // Remove the TanOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->getGraph().eraseOp(op->id);

  // Connect up the new ops
  sin->connectInTensor(SinOp::getInIndex(), fwd_in->id);
  sin->createAndConnectOutTensor(SinOp::getOutIndex(),
                                 createIntermediateTensorId(fwd_in->id));
  sin->setup();

  cos->connectInTensor(CosOp::getInIndex(), fwd_in->id);
  cos->createAndConnectOutTensor(CosOp::getOutIndex(),
                                 createIntermediateTensorId(fwd_in->id));
  cos->setup();

  div->connectInTensor(DivOp::getArg0InIndex(),
                       sin->outTensor(SinOp::getOutIndex())->id);
  div->connectInTensor(DivOp::getArg1InIndex(),
                       cos->outTensor(CosOp::getOutIndex())->id);
  div->connectOutTensor(DivOp::getOutIndex(), fwd_out->id);

  return true;
}

namespace {
static PatternCreator<TanToSinOverCosPattern>
    TanToSinOverCosPattern(PreAliasPatternType::TANTOSINOVERCOS,
                           "TanToSinOverCos");
}

} // namespace poponnx
