// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ELEMENTWISEGRADOPPATTERN_HPP
#define GUARD_NEURALNET_ELEMENTWISEGRADOPPATTERN_HPP

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/mul.hpp>
#include <popart/patterns/pattern.hpp>

namespace popart {

// Generalized pattern to replace a GradOp with the Derivative of the Op and a
// Mul
template <class GRADOP, class DOP>
class ElementWiseGradOpPattern : public PreAliasPattern {
public:
  bool matches(Op *op) const override { return op->isConvertibleTo<GRADOP>(); }
  std::vector<const Tensor *> touches(Op *) const override { return {}; }
  bool apply(Op *op) const override {
    auto grad_in  = op->inTensor(GRADOP::getGradInIndex());
    auto fwd_in   = op->inTensor(GRADOP::getFwdArgInIndex());
    auto grad_out = op->outTensor(GRADOP::getOutIndex());

    // create the new ops
    auto d   = makeReplacementOpInIr(DOP::getOpId(op->getIr()), op);
    auto mul = makeReplacementOpInIr(MulOp::getOpId(op->getIr()), op);

    // Remove the GRADOP
    op->disconnectAllInputs();
    op->disconnectAllOutputs();
    op->getGraph().eraseOp(op->id);

    // Connect up the new ops
    d->connectInTensor(DOP::getInIndex(), fwd_in->id);
    d->createAndConnectOutTensor(
        DOP::getOutIndex(),
        grad_in->getIr().createIntermediateTensorId(grad_in->id));
    d->setup();

    mul->connectInTensor(MulOp::getArg0InIndex(), grad_in->id);
    mul->connectInTensor(MulOp::getArg1InIndex(),
                         d->outTensor(DOP::getOutIndex())->id);
    mul->connectOutTensor(MulOp::getOutIndex(), grad_out->id);

    return true;
  }
};
} // namespace popart

#endif
