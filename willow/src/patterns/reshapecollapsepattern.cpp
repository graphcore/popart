// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/reshape.hpp>
#include <popart/patterns/reshapecollapsepattern.hpp>
#include <popart/tensor.hpp>
#include <popart/topocons.hpp>

namespace popart {

namespace {
Op *findReshapeProducer(Tensor *t) {
  if (t->hasProducer()) {
    auto prod = t->getProducer();
    if (prod->isConvertibleTo<ReshapeBaseOp>()) {
      return prod;
    }
  }
  return nullptr;
}
} // namespace

bool ReshapeCollapsePattern::matches(Op *op) const {
  auto &ir = op->getIr();
  // We want to run this pattern as late in the IR prepartion as possible
  // To ensure anchors constructed by using reservedPrefixes
  // are not removed.
  if (ir.canTrain() && !ir.hasDecomposedOptimizers()) {
    return false;
  }
  if (op->isConvertibleTo<ReshapeBaseOp>() &&
      findReshapeProducer(op->inTensor(ReshapeBaseOp::getInIndex()))) {
    return true;
  }
  return false;
}

std::vector<const Tensor *> ReshapeCollapsePattern::touches(Op *op) const {
  return {op->inTensor(ReshapeBaseOp::getInIndex())};
}

bool ReshapeCollapsePattern::apply(Op *op) const {
  auto prodReshape =
      findReshapeProducer(op->inTensor(ReshapeBaseOp::getInIndex()));
  op->disconnectInTensor(ReshapeBaseOp::getInIndex());
  op->connectInTensor(ReshapeBaseOp::getInIndex(),
                      prodReshape->inId(ReshapeBaseOp::getInIndex()));
  // We have removed a data scheduling constraint. This replaces it with an
  // equivilent topological constraint.
  op->getGraph().topoCons->insert(prodReshape, op);
  return true;
}

namespace {
static PatternCreator<ReshapeCollapsePattern>
    PreUniReplPattern("ReshapeCollapsePattern", true);
}

} // namespace popart
