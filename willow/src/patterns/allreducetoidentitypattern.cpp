// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/collectives/allreduce.hpp>
#include <popart/patterns/allreducetoidentitypattern.hpp>
#include <popart/tensorindex.hpp>

#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"

namespace popart {
class Tensor;

bool AllReduceToIdentityPattern::matches(Op *op) const {
  // Only match after autodiff for onnx models
  if (op->getIr().hasOnnxModel() && !op->getIr().hasConstructedBackwards()) {
    return false;
  }

  const AllReduceOp *op_ = dynamic_cast<const AllReduceOp *>(op);
  return (op_ != nullptr) && op_->getIdenticalInputs();
}

std::vector<const Tensor *> AllReduceToIdentityPattern::touches(Op *) const {
  return {};
}

bool AllReduceToIdentityPattern::apply(Op *op) const {
  AllReduceOp *op_ = dynamic_cast<AllReduceOp *>(op);

  auto numInputs = op_->input->n();
  auto inputs    = op_->input->tensorIdMap();
  auto outputs   = op_->output->tensorIdMap();
  auto ipus      = op_->getIpus();
  op_->disconnectAllInputs();
  op_->disconnectAllOutputs();

  for (int i = 0; i < numInputs; i++) {
    auto identity_op =
        makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Identity, op_);
    identity_op->connectInTensor(0, inputs[i]);
    identity_op->connectOutTensor(0, outputs[i]);
    identity_op->setVirtualGraphId(ipus[i]);
    identity_op->setup();
  }

  op_->getGraph().eraseOp(op_->id);

  return true;
}

namespace {
static PatternCreator<AllReduceToIdentityPattern>
    allReduceToIdentityPattern("AllReduceToIdentity", true, true);
}

} // namespace popart
