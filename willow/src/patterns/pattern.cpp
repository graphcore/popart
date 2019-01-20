#include <onnx/onnx_pb.h>
#include <spdlog/fmt/fmt.h>
#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/op.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/patterns/pattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/util.hpp>

namespace poponnx {

int Pattern::tensor_counter = 0;

bool Pattern::touchesAnchored(Op *op) const {
  for (auto &tensor : touches(op)) {
    if (op->pir->isAnchored(tensor->id)) {
      return true;
    }
  }
  return false;
};

TensorId Pattern::createIntermediateTensorId(TensorId base_id) {
  auto temp_id = fmt::format("t{}__{}", tensor_counter++, base_id);
  logging::ir::trace("Generating tensor id {}", temp_id);
  return temp_id;
}

void Pattern::initialise(std::string pattern_name_) {
  pattern_name = pattern_name_;
}

std::unique_ptr<Op>
Pattern::makeReplacementOp(const OperatorIdentifier &operator_id,
                           Op *oldOp,
                           const Attributes &attr) const {
  // Context from original op to be transfered to replacement ops
  auto ir   = oldOp->pir;
  auto name = getReplacementOpName(oldOp);

  // Inherit some attributes from the op that the new op is replacing
  std::vector<std::string> inheritedAttributes = {sVirtualGraphAttribute,
                                                  sRecomputeOutputAttribute};

  Attributes newAttrs = attr;
  for (auto attr_name : inheritedAttributes) {
    auto _attr = oldOp->nAtts.filter(attr_name);
    newAttrs.takeAttribute(attr_name, _attr);
  }

  // Create replacement Op with new attributes
  return OpManager::createOp(operator_id, ir, name, newAttrs);
}

Op *Pattern::makeReplacementOpInIr(const OperatorIdentifier &operator_id,
                                   Op *oldOp,
                                   const Attributes &attr) const {
  auto ir = oldOp->pir;

  // Create replacement Op with new attributes and
  // move into Ir
  std::unique_ptr<Op> newOpUp = makeReplacementOp(operator_id, oldOp, attr);
  Op *newOp                   = newOpUp.get();
  ir->moveIntoIr(std::move(newOpUp));

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
