#include <poponnx/op/scale.hpp>
#include <poponnx/patterns/negativeonescalepattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

bool NegativeOneScalePattern::matches(Op *op) const {
  if (!(op->isConvertibleTo<ScaleOp>())) {
    return false;
  }

  return dynamic_cast<ScaleOp *>(op)->getScaleFactor() == -1.0f;
}

// output = neg(x)
std::vector<std::unique_ptr<Op>>
NegativeOneScalePattern::sequence(Op *op) const {
  std::vector<std::unique_ptr<Op>> seq;

  seq.push_back(makeReplacementOp(Onnx::AiOnnx::OpSet9::Neg, op, {}));

  return seq;
}

namespace {
static PatternCreator<NegativeOneScalePattern>
    negativeOneScalePatern(PreAliasPatternType::NEGATIVEONESCALE,
                           "NegativeOneScalePattern");
}

} // namespace poponnx
