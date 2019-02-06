#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/addbias.hpp>
#include <poponnx/op/conv.hpp>
#include <poponnx/patterns/convbias.hpp>
#include <poponnx/patterns/patterns.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensors.hpp>

namespace poponnx {

bool ConvBiasPattern::matches(Op *op) const {
  return (op->opid == Onnx::Operators::Conv_1) && (op->input->n() == 3);
}

std::vector<const Tensor *> ConvBiasPattern::touches(Op *) const { return {}; }

bool ConvBiasPattern::apply(Op *op) const {
  const auto conv = dynamic_cast<ConvOp *>(op);

  auto add_bias = makeReplacementOpInIr(Onnx::CustomOperators::AddBias, op);

  const auto tmp_tensor_id = "prebias" + conv->output->id(0);
  op->getIr().getTensors().addActGrad(tmp_tensor_id);
  const auto b  = conv->input->tensor(ConvOp::getBiasInIndex());
  const auto t  = op->getIr().getTensors().get(tmp_tensor_id);
  const auto a1 = conv->output->tensor(ConvOp::getDataInIndex());

  t->info = a1->info;
  t->setProducer(conv);
  t->consumers.increment(add_bias);
  b->consumers.increment(add_bias);
  b->consumers.decrement(conv);
  a1->resetProducer(add_bias);

  conv->input->erase(ConvOp::getBiasInIndex());
  add_bias->input->insert(AddBiasOp::getDataInIndex(), t);
  add_bias->input->insert(AddBiasOp::getBiasInIndex(), b);

  conv->output->reset(0, t);
  add_bias->output->insert(0, a1);

  return true;
}

namespace {
static PatternCreator<ConvBiasPattern>
    convBiasPattern(PreAliasPatternType::SPLITCONVBIAS, "SplitConvBias");
}

} // namespace poponnx
