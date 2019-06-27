#include <memory>
#include <poponnx/ir.hpp>
#include <poponnx/op/concat.hpp>
#include <poponnx/op/gather.hpp>
#include <poponnx/op/identity.hpp>
#include <poponnx/op/pad.hpp>
#include <poponnx/op/reducesum.hpp>
#include <poponnx/op/scale.hpp>
#include <poponnx/op/split.hpp>
#include <poponnx/op/subsample.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/patterns/splitgradoptoconcatpattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

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

} // namespace poponnx
