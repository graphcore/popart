#include <memory>
#include <popart/ir.hpp>
#include <popart/op/gather.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/pad.hpp>
#include <popart/op/reducesum.hpp>
#include <popart/op/scale.hpp>
#include <popart/op/subsample.hpp>
#include <popart/opmanager.hpp>
#include <popart/patterns/optoidentitypattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

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

} // namespace popart
