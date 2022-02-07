// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <popart/logging.hpp>
#include <popart/op/cast.hpp>
#include <popart/op/expand.hpp>
#include <popart/patterns/expandcast.hpp>

namespace popart {

bool ExpandCastPattern::matches(Op *op) const {
  if (!op->isConvertibleTo<ExpandOp>()) {
    return false;
  }

  Tensor *out = op->output->tensor(ExpandOp::getOutIndex());

  // Expand must be consumed just once
  if (out->consumers.getTotal() != 1) {
    return false;
  }

  Op *maybeCast = out->consumers.getOps()[0];
  if (!maybeCast->isConvertibleTo<CastOp>()) {
    return false;
  }

  // Do not swap if they are are on different IPUs
  if (op->getOptionalVGraphId() != maybeCast->getOptionalVGraphId()) {
    return false;
  }

  // Do not swap if output shape == input shape (a pointless expand)
  auto outElements = op->outInfo(ExpandOp::getOutIndex()).nelms();
  auto inElements = op->inInfo(ExpandOp::getInTensorIndex()).nelms();

  if (outElements == inElements) {
    return false;
  }

  return true;
}

std::vector<const Tensor *> ExpandCastPattern::touches(Op *op) const {
  Tensor *out = op->output->tensor(ExpandOp::getOutIndex());
  return {out,
          out->consumers.getOps()[0]->output->tensor(CastOp::getOutIndex())};
}

bool ExpandCastPattern::apply(Op *expandOp) const {
  Tensor *expandOpOut = expandOp->output->tensor(ExpandOp::getOutIndex());
  Op *castOp          = expandOpOut->consumers.getOps()[0];

  // Keep the expand input
  Tensor *overallIn = expandOp->input->tensor(ExpandOp::getInTensorIndex());

  // Keep the original intermediate tensor
  Tensor *intermediate = expandOp->output->tensor(ExpandOp::getOutIndex());

  // Keep the original output
  Tensor *overallOut = castOp->output->tensor(CastOp::getOutIndex());

  // Disconnect all
  expandOp->disconnectAllInputs();
  expandOp->disconnectAllOutputs();
  castOp->disconnectAllInputs();
  castOp->disconnectAllOutputs();

  // Connect in order cast -> expand -> out
  castOp->connectInTensor(CastOp::getInIndex(), overallIn->id);
  castOp->connectOutTensor(CastOp::getOutIndex(), intermediate->id);
  expandOp->connectInTensor(ExpandOp::getInTensorIndex(), intermediate->id);
  expandOp->connectOutTensor(ExpandOp::getOutIndex(), overallOut->id);

  // Call setup to fix the tensor details
  castOp->setup();
  expandOp->setup();

  return true;
}

namespace {
static PatternCreator<ExpandCastPattern>
    ExpandCastPattern("ExpandCast", true, false);
}

} // namespace popart
