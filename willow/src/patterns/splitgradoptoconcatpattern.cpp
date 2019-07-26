#include <memory>
#include <popart/ir.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/gather.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/pad.hpp>
#include <popart/op/reducesum.hpp>
#include <popart/op/scale.hpp>
#include <popart/op/split.hpp>
#include <popart/op/subsample.hpp>
#include <popart/opmanager.hpp>
#include <popart/patterns/splitgradoptoconcatpattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

bool SplitGradOpToConcatPattern::matches(Op *op) const {
  return op->isConvertibleTo<SplitGradOp>();
}

std::vector<std::unique_ptr<Op>>
SplitGradOpToConcatPattern::sequence(Op *op) const {
  auto splitGradOp = dynamic_cast<SplitGradOp *>(op);

  std::vector<std::unique_ptr<Op>> seq;

  auto axis = splitGradOp->getAxis();
  seq.push_back(std::make_unique<ConcatOp>(
      Onnx::Operators::Concat_4, axis, op->getSettings()));

  return seq;
}

namespace {
static PatternCreator<SplitGradOpToConcatPattern>
    splitGradOpToConcatPattern(PreAliasPatternType::SPLITGRADOPTOCONCAT,
                               "SplitGradOpToConcat");
}

} // namespace popart
