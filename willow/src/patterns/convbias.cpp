// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/graph.hpp>
#include <popart/op/addbias.hpp>
#include <popart/op/conv.hpp>
#include <popart/patterns/convbias.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensors.hpp>

namespace popart {

bool ConvBiasPattern::matches(Op *op) const {
  return ((op->opid == Onnx::Operators::Conv_1 ||
           op->opid == Onnx::Operators::Conv_11) &&
          (op->input->n() == 3));
}

std::vector<const Tensor *> ConvBiasPattern::touches(Op *) const { return {}; }

bool ConvBiasPattern::apply(Op *op) const {
  const auto conv = dynamic_cast<ConvOp *>(op);

  auto add_bias = makeReplacementOpInIr(Onnx::CustomOperators::AddBias, op);

  const auto tmp_tensor_id = "prebias" + conv->output->id(0);
  op->getGraph().getTensors().addActGrad(tmp_tensor_id);
  const auto b  = conv->input->tensor(ConvOp::getBiasInIndex());
  const auto t  = op->getGraph().getTensors().get(tmp_tensor_id);
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
    convBiasPattern(PreAliasPatternType::SplitConvBias, "SplitConvBias");
}

} // namespace popart
