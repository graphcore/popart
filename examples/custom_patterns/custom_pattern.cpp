// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/op/negate.hpp>
#include <popart/op/relu.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/patterns/patterns.hpp>

using namespace popart;

class ReplaceReluWithNeg : public PreAliasPattern {
public:
  bool matches(Op *op) const override { return op->isConvertibleTo<ReluOp>(); }

  std::vector<const Tensor *> touches(Op *) const override { return {}; }

  bool apply(Op *op) const override {
    logging::debug("ReplaceReluWithNeg::apply({})", op->debugName());
    op->setName("someReluOp");

    auto negOp =
        makeReplacementOpInIr(Onnx::Operators::Neg_6, op, "reluReplacement");

    auto inputId  = op->inId(ReluOp::getInIndex());
    auto outputId = op->outId(ReluOp::getOutIndex());
    op->disconnectAllInputs();
    op->disconnectAllOutputs();
    op->getGraph().eraseOp(op->id);

    negOp->connectInTensor(NegateOp::getInIndex(), inputId);
    negOp->connectOutTensor(NegateOp::getOutIndex(), outputId);
    negOp->setup();

    return true;
  }
};

static PatternCreator<ReplaceReluWithNeg> myPatternCreator("ReplaceReluWithNeg",
                                                           true);
