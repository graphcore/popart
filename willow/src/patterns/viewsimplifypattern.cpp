// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/reshape.hpp>
#include <popart/patterns/viewsimplifypattern.hpp>
#include <popart/tensor.hpp>
#include <popart/topocons.hpp>

#include "popart/names.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {

namespace {
std::pair<bool, InIndex> supportedViewOpAndInIndex(Op *op) {
  if (op->isConvertibleTo<ReshapeBaseOp>()) {
    return {true, ReshapeBaseOp::getInIndex()};
  }
  if (op->isConvertibleTo<IdentityOp>()) {
    return {true, IdentityOp::getInIndex()};
  }
  return {false, 0};
}

std::pair<Op *, InIndex> findSupportedProducer(Tensor *t) {
  if (t->hasProducer()) {
    auto prod            = t->getProducer();
    auto supported_index = supportedViewOpAndInIndex(prod);
    if (supported_index.first) {
      return {prod, supported_index.second};
    }
  }
  return {nullptr, 0};
}

bool hasImplicitRecomputeConflict(std::pair<Op *, InIndex> &producer_index) {
  return producer_index.first->settings.recomputeType ==
             RecomputeType::Checkpoint &&
         producer_index.first->inTensor(producer_index.second)
             ->isImplicitRecomputeTensor();
}
} // namespace

bool ViewSimplifyPattern::matches(Op *op) const {
  auto supported_index = supportedViewOpAndInIndex(op);
  if (supported_index.first) {
    auto producer_index =
        findSupportedProducer(op->inTensor(supported_index.second));
    if (producer_index.first) {

      if (hasImplicitRecomputeConflict(producer_index)) {
        // This is to catch assumptions made by implicit recompute.
        // If we have the program:
        //   y = a(x)
        //   z = b(y)
        // if 'a' is Checkpoint operation and 'x' is produced by a Recompute
        // operation then by modifying `b(y) -> b(x)` we will change how the
        // program is executed as getRequiredRecomputeOps in irlowering will not
        // stop at 'a'.
        // TODO T17663: Remove once explicit recompute is standard.
        return false;
      }

      return true;
    }
  }
  return false;
}

std::vector<const Tensor *> ViewSimplifyPattern::touches(Op *op) const {
  return {op->inTensor(supportedViewOpAndInIndex(op).second)};
}

bool ViewSimplifyPattern::apply(Op *op) const {
  auto &graph   = op->getGraph();
  auto in_index = supportedViewOpAndInIndex(op).second;
  auto producer_index =
      findSupportedProducer(op->inTensor(ReshapeBaseOp::getInIndex()));
  if (op->isConvertibleTo<IdentityOp>() &&
      producer_index.first->isConvertibleTo<ReshapeBaseOp>()) {
    // Identity(Reshape(x)) -> Reshape(x)
    auto replacement = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Reshape, op);
    auto out         = op->outTensor(IdentityOp::getOutIndex());

    op->disconnectAllInputs();
    op->disconnectAllOutputs();

    replacement->connectInTensor(
        in_index, producer_index.first->inId(producer_index.second));
    replacement->connectOutTensor(ReshapeBaseOp::getOutIndex(), out->id);
    dynamic_cast<ReshapeBaseOp *>(replacement)->setOutShape(out->info.shape());
    replacement->setup();

    graph.topoCons->transfer(op, replacement);

    graph.eraseOp(op->id);
    op = replacement;
  } else {
    // Identity(Identity(x)) -> Identity(x)
    // Reshape(Identity(x))  -> Reshape(x)
    // Reshape(Reshape(x)    -> Reshape(x)
    op->disconnectInTensor(in_index);
    op->connectInTensor(in_index,
                        producer_index.first->inId(producer_index.second));
  }

  // We have removed a data scheduling constraint. This replaces it with an
  // equivalent topological constraint.
  graph.topoCons->insert(producer_index.first, op);
  return true;
}

namespace {
// Not registering this pattern, as we want it to run at a special time
static AddPatternName<ViewSimplifyPattern> registerName("ViewSimplifyPattern");
} // namespace

} // namespace popart
