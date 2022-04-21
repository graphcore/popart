// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <onnx/onnx_pb.h>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>
#include <popart/util.hpp>

namespace popart {

int PreAliasPattern::tensor_counter = 0;

bool PreAliasPattern::touchesAnchored(Op *op) const {
  for (auto &tensor : touches(op)) {
    if (op->getIr().isAnchored(tensor->id)) {
      return true;
    }
  }
  return false;
}

void Pattern::transferBaseProperties(const Op *from, Op *to) const {
  if (from->hasVirtualGraphId()) {
    to->setVirtualGraphId(from->getVirtualGraphId());
  }
  if (from->hasExecutionPhase()) {
    to->setExecutionPhase(from->getExecutionPhase());
  }
  if (from->hasPipelineStage()) {
    to->setPipelineStage(from->getPipelineStage());
  }
  if (from->hasBatchSerializedPhase()) {
    to->setBatchSerializedPhase(from->getBatchSerializedPhase());
  }

  to->settings.scope            = from->settings.scope;
  to->settings.recomputeType    = from->settings.recomputeType;
  to->settings.tensorLocation   = from->settings.tensorLocation;
  to->fromLoss                  = from->fromLoss;
  to->toLoss                    = from->toLoss;
  to->settings.schedulePriority = from->settings.schedulePriority;
  to->settings.debugInfoId      = from->settings.debugInfoId;
}

std::unique_ptr<Op>
PreAliasPattern::makeReplacementOp(const OperatorIdentifier &operator_id,
                                   Op *oldOp,
                                   const std::string name) const {

  // Need to pass the debug if from old to new op in the attributes constructor
  // argument
  popart::Attributes attributes;
  // Warning : converting uint64_t to int64_t;
  Attributes::Int debugId = oldOp->settings.debugInfoId;
  attributes.setAttribute(sDebugInfoId, debugId);

  // Create replacement Op with new attributes
  std::unique_ptr<Op> newOp =
      OpManager::createOp(operator_id,
                          oldOp->getGraph(),
                          getReplacementOpName(oldOp, name),
                          attributes);

  if (newOp == nullptr) {
    throw internal_error(
        "nullptr for newOp in makeReplacementOp, for op of type "
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
    const std::string name) const {
  // Create replacement Op with new attributes and
  // move into Ir
  std::unique_ptr<Op> newOpUp = makeReplacementOp(operator_id, oldOp, name);
  Op *newOp                   = newOpUp.get();
  oldOp->getGraph().moveIntoGraph(std::move(newOpUp));
  transferBaseProperties(oldOp, newOp);
  return newOp;
}

const std::string &Pattern::getPatternName() const {
  return PatternNames::getName(typeid(*this));
}

std::string Pattern::getReplacementOpName(Op *op,
                                          const std::string name) const {
  std::string replacementName;
  if (op->name() == "") {
    replacementName = "";
  } else {
    replacementName = op->name() + ":" + getPatternName();
  }

  if (!name.empty()) {
    replacementName.append(std::string("_") + name);
  }

  return replacementName;
}

} // namespace popart
