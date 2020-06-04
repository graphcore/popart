// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <tuple>
#include <popart/graph.hpp>
#include <popart/op/flatten.hpp>
#include <popart/op/reshape.hpp>
#include <popart/op/squeeze.hpp>
#include <popart/op/unsqueeze.hpp>
#include <popart/patterns/optoreshapepattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

bool OpToReshapePattern::matches(Op *op) const {
  return (op->isConvertibleTo<SqueezeOp>() ||
          op->isConvertibleTo<UnsqueezeOp>() ||
          op->isConvertibleTo<FlattenOp>());
}

std::vector<const Tensor *> OpToReshapePattern::touches(Op *) const {
  return {};
}

// grad_out = grad_in / fwd_in1
bool OpToReshapePattern::apply(Op *op) const {
  // Get operation and operands.
  // auto squeezeOp = dynamic_cast<SqueezeOp *>(op);
  auto in  = op->inTensor(0);
  auto out = op->outTensor(SqueezeBaseOp::getOutIndex());

  // Create replacement op.
  auto reshape_op = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Reshape, op);

  // Preserve/transfer inplace priorities.
  for (auto inplacePriorityEntry : op->settings.inplacePriorityVeto) {
    std::string key;
    float priority;
    std::tie(key, priority) = inplacePriorityEntry;
    if ((key == "SqueezeInplace" && op->isConvertibleTo<SqueezeOp>()) ||
        (key == "UnsqueezeInplace" && op->isConvertibleTo<UnsqueezeOp>()) ||
        (key == "FlattenInplace" && op->isConvertibleTo<FlattenOp>())) {
      reshape_op->settings.inplacePriorityVeto.push_back(
          std::tuple<std::string, float>("ReshapeInplace", priority));
    }
  }

  // Disconnect op.
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->getGraph().eraseOp(op->id);

  // Connect reshape_op.
  reshape_op->connectInTensor(0, in->id);
  reshape_op->connectOutTensor(SqueezeBaseOp::getOutIndex(), out->id);

  // If we don't do this the ReshapeOp's outShape member will be wrong and mess
  // things up.
  auto reshape_base_op = dynamic_cast<popart::ReshapeBaseOp *>(reshape_op);
  reshape_base_op->setOutShape(out->info.shape());

  return true;
}

namespace {
static PatternCreator<popart::OpToReshapePattern>
    OpToReshapePattern(PreAliasPatternType::OpToReshape, "OpToReshape");
}

} // namespace popart
