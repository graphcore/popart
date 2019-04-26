#include <poponnx/graph.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/abs.hpp>
#include <poponnx/op/cos.hpp>
#include <poponnx/op/mul.hpp>
#include <poponnx/op/sign.hpp>
#include <poponnx/op/sin.hpp>
#include <poponnx/patterns/pattern.hpp>
#include <poponnx/patterns/patterns.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

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
    d->createAndConnectOutTensor(DOP::getOutIndex(),
                                 createIntermediateTensorId(grad_in->id));
    d->setup();

    mul->connectInTensor(MulOp::getArg0InIndex(), grad_in->id);
    mul->connectInTensor(MulOp::getArg1InIndex(),
                         d->outTensor(DOP::getOutIndex())->id);
    mul->connectOutTensor(MulOp::getOutIndex(), grad_out->id);

    return true;
  }
};

namespace {
// Replace a AbsGradOp with
// (fwd_in) -> [Sign] -> (tmp1)
// {(tmp1), (grad_in)} -> [Mul] -> (grad_out)
static PatternCreator<ElementWiseGradOpPattern<AbsGradOp, SignOp>>
    sinGradOpPattern(PreAliasPatternType::ABSGRADOP, "AbsGradOp");

// Replace a SinGradOp with
// (fwd_in) -> [Cos] -> (tmp1)
// {(tmp1), (grad_in)} -> [Mul] -> (grad_out)
static PatternCreator<ElementWiseGradOpPattern<SinGradOp, CosOp>>
    absGradOpPattern(PreAliasPatternType::SINGRADOP, "SinGradOp");

} // namespace

} // namespace poponnx
