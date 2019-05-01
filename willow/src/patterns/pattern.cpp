#include <onnx/onnx_pb.h>
#include <spdlog/fmt/fmt.h>
#include <poponnx/error.hpp>
#include <poponnx/graph.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/op.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/patterns/pattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/util.hpp>

namespace poponnx {

int PreAliasPattern::tensor_counter = 0;

bool PreAliasPattern::touchesAnchored(Op *op) const {
  for (auto &tensor : touches(op)) {
    if (op->getIr().isAnchored(tensor->id)) {
      return true;
    }
  }
  return false;
};

TensorId PreAliasPattern::createIntermediateTensorId(TensorId base_id) {
  auto temp_id = fmt::format("t{}__{}", tensor_counter++, base_id);
  logging::pattern::trace("Generating tensor id {}", temp_id);
  return temp_id;
}

void Pattern::initialise(std::string pattern_name_) {
  pattern_name = pattern_name_;
}

void Pattern::transferBaseProperties(Op *from, Op *to) const {
  if (from->getVirtualGraphId()) {
    to->setVirtualGraphId(from->getVirtualGraphId());
  }

  if (from->getRecomputeOutput()) {
    to->setRecomputeOutput(from->getRecomputeOutput());
  }
}

std::unique_ptr<Op>
PreAliasPattern::makeReplacementOp(const OperatorIdentifier &operator_id,
                                   Op *oldOp,
                                   const Attributes &) const {

  // Create replacement Op with new attributes
  std::unique_ptr<Op> newOp = OpManager::createOp(
      operator_id, oldOp->getGraph(), getReplacementOpName(oldOp));

  if (newOp == nullptr) {
    throw error(
        "ILE : nullptr for newOp in makeReplacementOp, for op of type "
        "{} trying to make {}. Possibly need to 'register' the replacement? ",
        oldOp->str(),
        operator_id);
  }

  transferBaseProperties(oldOp, newOp.get());
  return newOp;
}

Op *PreAliasPattern::makeReplacementOpInIr(
    const OperatorIdentifier &operator_id,
    Op *oldOp,
    const Attributes &attr) const {
  // Create replacement Op with new attributes and
  // move into Ir
  std::unique_ptr<Op> newOpUp = makeReplacementOp(operator_id, oldOp, attr);
  Op *newOp                   = newOpUp.get();
  oldOp->getGraph().moveIntoGraph(std::move(newOpUp));

  return newOp;
}

const std::string &Pattern::getPatternName() const { return pattern_name; }

std::string Pattern::getReplacementOpName(Op *op) const {
  std::string replacementName;
  if (op->name() == "") {
    replacementName = "";
  } else {
    replacementName = op->name() + "_from_" + getPatternName();
  }
  return replacementName;
}

} // namespace poponnx
