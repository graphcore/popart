// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/gather.hpp>
#include <popart/op/groupnorm.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/pad.hpp>
#include <popart/op/reducesum.hpp>
#include <popart/op/scale.hpp>
#include <popart/op/subsample.hpp>
#include <popart/opmanager.hpp>
#include <popart/patterns/optoidentitypattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

bool OpToIdentityPattern::matches(Op *op) const {
  return op->canBeReplacedByIdentity();
}

std::vector<std::unique_ptr<Op>> OpToIdentityPattern::sequence(Op *op) const {
  std::vector<std::unique_ptr<Op>> seq;

  // For Ops that match this pattern, but have more than one input, leave only
  // the input connected that is mapped to the output by the identity function
  if (op->isConvertibleTo<GatherOp>()) {
    op->disconnectInTensor(GatherOp::indicesInIndex());
  }
  if (op->isConvertibleTo<GroupNormOp>()) {
    op->disconnectInTensor(GroupNormOp::getScaleInIndex());
    op->disconnectInTensor(GroupNormOp::getBInIndex());
  }

  // It should be an error to replace an op with multiple inputs.
  if (op->input->n() > 1) {
    throw error("Can not replace op with {} inputs with IdentityOp.",
                op->input->n());
  }

  seq.push_back(makeReplacementOp(Onnx::AiOnnx::OpSet9::Identity, op));

  return seq;
}

namespace {
static PatternCreator<OpToIdentityPattern> opToIdentityPattern("OpToIdentity");
}

} // namespace popart
