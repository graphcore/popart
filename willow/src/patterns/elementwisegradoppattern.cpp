#include <memory>
#include <popart/op/abs.hpp>
#include <popart/op/cos.hpp>
#include <popart/op/sign.hpp>
#include <popart/op/sin.hpp>
#include <popart/patterns/elementwisegradoppattern.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

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

} // namespace popart
