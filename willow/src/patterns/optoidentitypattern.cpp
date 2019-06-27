#include <memory>
#include <poponnx/ir.hpp>
#include <poponnx/op/gather.hpp>
#include <poponnx/op/identity.hpp>
#include <poponnx/op/pad.hpp>
#include <poponnx/op/reducesum.hpp>
#include <poponnx/op/scale.hpp>
#include <poponnx/op/subsample.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/patterns/optoidentitypattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

bool OpToIdentityPattern::matches(Op *op) const {
  return op->canBeReplacedByIdentity();
}

std::vector<std::unique_ptr<Op>> OpToIdentityPattern::sequence(Op *op) const {
  std::vector<std::unique_ptr<Op>> seq;

  seq.push_back(makeReplacementOp(Onnx::AiOnnx::OpSet9::Identity, op, {}));

  return seq;
}

namespace {
static PatternCreator<OpToIdentityPattern>
    opToIdentityPattern(PreAliasPatternType::OPTOIDENTITY, "OpToIdentity");
}

} // namespace poponnx
