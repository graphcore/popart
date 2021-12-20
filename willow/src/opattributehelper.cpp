// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <popart/graph.hpp>
#include <popart/graphutils.hpp>
#include <popart/ir.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/opattributehelper.hpp>
#include <popart/pointercomparators.hpp>
#include <popart/sessionoptions.hpp>

namespace popart {

namespace {
InheritOpAttributeHelper::InfoSortPriority
getExecutionContextPriority(ExecutionContext context) {
  switch (context) {
  case ExecutionContext::OptimizerFromHostFragment:
  case ExecutionContext::WeightsFromHostFragment:
    return 0;
  case ExecutionContext::Normal:
  case ExecutionContext::Subgraph:
    return 1;
  case ExecutionContext::AccumulateOuterFragment:
    return 2;
  case ExecutionContext::WeightsToHostFragment:
    return 3;
  default:
    return 4;
  }
}
} // namespace

std::ostream &
operator<<(std::ostream &os,
           InheritOpAttributeHelper::ConnectedOpRelation relation) {
  switch (relation) {
  case InheritOpAttributeHelper::ConnectedOpRelation::Producer:
    os << "Producer";
    break;
  case InheritOpAttributeHelper::ConnectedOpRelation::Consumer:
    os << "Consumer";
    break;
  case InheritOpAttributeHelper::ConnectedOpRelation::N:
    os << "N";
    break;
  }
  return os;
}

std::ostream &
operator<<(std::ostream &os,
           const InheritOpAttributeHelper::TraversalTensorInfo &info) {
  os << "TraversalTensorInfo(relation: " << info.relation
     << ", distance: " << info.distance << ")";
  return os;
}

void InheritOpAttributeHelper::apply(Op *op,
                                     bool inheritSerializations,
                                     AliasModel &aliasModel) {
  InheritOpAttributeHelper helper(op, inheritSerializations, aliasModel);
  helper.traverseGraphs();
  helper.setAttributes();
}

InheritOpAttributeHelper::InheritOpAttributeHelper(Op *op_,
                                                   bool inheritSerializations_,
                                                   AliasModel &aliasModel_)
    : op(op_), aliasModel(aliasModel_),
      inheritSerializations(inheritSerializations_), ir(op->getGraph().getIr()),
      opts(ir.getSessionOptions()) {

  // Extract settings
  vgraphed  = opts.virtualGraphMode != VirtualGraphMode::Off;
  pipelined = opts.enablePipelining;
  executionphased =
      opts.virtualGraphMode == VirtualGraphMode::ExecutionPhases &&
      opts.executionPhaseSettings.phases > 1;
}

std::vector<Tensor *> InheritOpAttributeHelper::getTraversalStartTensors() {
  std::vector<Tensor *> startTensors;

  // Start from all inputs to this Op
  for (auto inIndexAndTensor : op->input->tensorMap()) {
    tensorRelationMap[inIndexAndTensor.second] =
        TraversalTensorInfo{ConnectedOpRelation::Producer, 0};
    startTensors.push_back(inIndexAndTensor.second);
  }

  // Start from all outputs of this Op
  for (auto outIndexAndTensor : op->output->tensorMap()) {
    tensorRelationMap[outIndexAndTensor.second] =
        TraversalTensorInfo{ConnectedOpRelation::Consumer, 0};
    startTensors.push_back(outIndexAndTensor.second);
  }
  return startTensors;
}

void InheritOpAttributeHelper::addAttributeEntry(
    Op *visitedOp,
    Tensor *visitedTensor,
    const TraversalTensorInfo &info,
    bool isInput) {
  bool graphChanged = visitedOp->getGraph().id != op->getGraph().id;
  auto distance     = info.distance;

  // Sort descending for producer, ascending for consumer
  InfoSortPriority priMod =
      info.relation == ConnectedOpRelation::Producer ? -1 : 1;

  if (!stopExecutionContextSearch(graphChanged, distance)) {
    auto executionContext = visitedOp->settings.executionContext;
    executionContextSet.insert(
        {info,
         priMod * getExecutionContextPriority(executionContext),
         executionContext,
         visitedOp->id});
  }

  if (!stopVGIDSearch(graphChanged, distance)) {
    OptionalVGraphId vgid;
    if (isInput) {
      vgid = visitedOp
                 ->getIntrospectionInVirtualGraphId(
                     visitedOp->input->indices(visitedTensor).front())
                 .first;
    } else {
      vgid = visitedOp
                 ->getIntrospectionOutVirtualGraphId(
                     visitedOp->output->indices(visitedTensor).front())
                 .first;
    }
    if (vgid && vgid != unusedVGraphId) {
      vgidSet.insert({info,
                      priMod * static_cast<InfoSortPriority>(*vgid),
                      *vgid,
                      visitedOp->id});
    }
  }

  if (!stopPipelineStageSearch(graphChanged, distance)) {
    auto pipelineStage = visitedOp->settings.pipelineStage;
    if (pipelineStage && pipelineStage != unusedPipelineStage) {
      pipelineStageSet.insert(
          {info,
           priMod * static_cast<InfoSortPriority>(*pipelineStage),
           *pipelineStage,
           visitedOp->id});
    }
  }

  if (!stopExecutionPhaseSearch(graphChanged, distance)) {
    auto executionPhase = visitedOp->settings.executionPhase;
    if (executionPhase && executionPhase != unusedExecutionPhase) {
      if (IpuCopyOp *copyOp = dynamic_cast<IpuCopyOp *>(visitedOp)) {
        if (copyOp->getSourceIpu(
                isInput ? visitedTensor->id
                        : copyOp->inId(copyOp->outIndex(visitedTensor))) %
                    2 !=
                copyOp->getDestIpu() % 2 &&
            !isInput && visitedOp->hasExecutionPhase()) {
          // Inter-phase copy: Destination phase
          executionPhase = *executionPhase + 1;
        }
      }
      executionPhaseSet.insert(
          {info,
           priMod * static_cast<InfoSortPriority>(*executionPhase),
           *executionPhase,
           visitedOp->id});
    }
  }

  if (!stopBatchSerializedPhaseSearch(graphChanged, distance)) {
    auto batchSerializedPhase = visitedOp->settings.batchSerializedPhase;
    if (batchSerializedPhase &&
        batchSerializedPhase != unusedBatchSerializedPhase) {
      batchSerializedPhaseSet.insert(
          {info,
           priMod * static_cast<InfoSortPriority>(*batchSerializedPhase),
           *batchSerializedPhase,
           visitedOp->id});
    }
  }
}

void InheritOpAttributeHelper::traverseGraphs() {
  // Tensors to start breadth-first traversal from
  std::vector<Tensor *> startTensors = getTraversalStartTensors();

  int64_t stopDistance = std::numeric_limits<int64_t>::max();
  graphutils::traverse(
      startTensors,
      [this](Tensor *t) {
        auto info = tensorRelationMap.at(t);

        if (t->hasProducer()) {
          Op *producer = t->getProducer();
          if (producer != op) {
            auto producerInfo = info;
            if (producerInfo.relation == ConnectedOpRelation::Consumer) {
              // Count co-producers as one step further away
              producerInfo.distance++;
            }
            addAttributeEntry(producer, t, producerInfo, false);
          }
        }
        for (Op *consumer : t->consumers.getOps()) {
          if (consumer != op) {
            auto consumerInfo = info;
            if (consumerInfo.relation == ConnectedOpRelation::Producer) {
              // Count co-consumers as one step further away
              consumerInfo.distance++;
            }
            addAttributeEntry(consumer, t, consumerInfo, true);
          }
        }
        return true;
      },
      [this, &stopDistance](Op *top, Tensor *t0, Tensor *t1) {
        // Insert without overwriting
        auto info = tensorRelationMap.at(t0);
        info.distance++;
        tensorRelationMap.insert({t1, info});
        if (top->isIpuCopyOp()) {
          // Stop at IpuCopyOps because they signify that the placement has
          // changed
          return false;
        }
        bool graphChanged = t1->getGraph().id != op->getGraph().id;
        if (canStopTraversal(graphChanged, info.distance)) {
          // Finish after visiting the current distance from the Op
          if (info.distance > stopDistance) {
            return false;
          } else {
            stopDistance = info.distance;
          }
        }
        return true;
      },
      graphutils::TraversalType::BreadthFirst,
      graphutils::VisitType::Pre,
      graphutils::TraversalDirection::ForwardBackward);

  logging::op::trace("[InheritOpAttributeHelper::traverseGraphs] vgidSet: {}",
                     vgidSet);
  logging::op::trace(
      "[InheritOpAttributeHelper::traverseGraphs] pipelineStageSet: {}",
      pipelineStageSet);
  logging::op::trace(
      "[InheritOpAttributeHelper::traverseGraphs] executionPhaseSet: {}",
      executionPhaseSet);
  logging::op::trace(
      "[InheritOpAttributeHelper::traverseGraphs] executionContextSet: {}",
      executionContextSet);
  logging::op::trace(
      "[InheritOpAttributeHelper::traverseGraphs] batchSerializedPhaseSet: {}",
      batchSerializedPhaseSet);
}

bool InheritOpAttributeHelper::stopExecutionContextSearch(bool graphChanged,
                                                          int distance) {
  return graphChanged;
}

bool InheritOpAttributeHelper::stopVGIDSearch(bool graphChanged, int distance) {
  return !vgraphed;
}

bool InheritOpAttributeHelper::stopPipelineStageSearch(bool graphChanged,
                                                       int distance) {
  return graphChanged || !pipelined;
}

bool InheritOpAttributeHelper::stopExecutionPhaseSearch(bool graphChanged,
                                                        int distance) {
  return graphChanged || !executionphased;
}

bool InheritOpAttributeHelper::stopBatchSerializedPhaseSearch(bool graphChanged,
                                                              int distance) {
  return graphChanged || distance > 0 || !inheritSerializations;
}

bool InheritOpAttributeHelper::canStopTraversal(bool graphChanged,
                                                int distance) {
  // Sufficient criteria to stop searching further
  return (stopExecutionContextSearch(graphChanged, distance) ||
          !executionContextSet.empty()) &&
         (stopVGIDSearch(graphChanged, distance) || !vgidSet.empty()) &&
         (stopPipelineStageSearch(graphChanged, distance) ||
          !pipelineStageSet.empty()) &&
         (stopExecutionPhaseSearch(graphChanged, distance) ||
          !executionPhaseSet.empty()) &&
         (stopBatchSerializedPhaseSearch(graphChanged, distance) ||
          !batchSerializedPhaseSet.empty());
}

OptionalVGraphId InheritOpAttributeHelper::getRequiredVirtualGraphId() {
  OptionalVGraphId requiredVgid;

  for (auto inIndexAndTensor : op->input->tensorMap()) {

    std::set<Tensor *, PTensorCmp> associatedVariableTensors;

    if (inIndexAndTensor.second->tensorType() == TensorType::Variable) {
      associatedVariableTensors.insert(inIndexAndTensor.second);
    }

    // Work out aliases of input the input that are variables.
    Tensor *inputTensor = inIndexAndTensor.second;
    if (aliasModel.contains(inputTensor->id)) {
      for (auto &alias : aliasModel.allAliases(*inputTensor)) {
        if (alias->tensorType() == TensorType::Variable) {
          associatedVariableTensors.insert(alias);
          associatedVariableTensors.insert(inIndexAndTensor.second);
        }
      }
    } else {
      logging::debug("[InheritOpAttributeHelper::getRequiredVirtualGraphId] "
                     "Tensor '{}' not modelled "
                     "in provided AliasModel",
                     inputTensor->id);
      // There are historic code paths that call this function with an alias
      // model that does not model all tensor inputs of the op (for example: the
      // gradient inputs in GradGrowerSumOp::growGradSumOp). We therefore don't
      // currently error when this is the case.
    }

    for (Tensor *varTensor : associatedVariableTensors) {
      auto modifiedRegions = op->modifies(inIndexAndTensor.first);

      bool variableModOrAlias =
          std::any_of(modifiedRegions.begin(),
                      modifiedRegions.end(),
                      [](view::Region &r) { return !r.isEmpty(); });

      for (auto outIndexAndTensor : op->output->tensorMap()) {
        auto aliasedRegions =
            op->aliases(inIndexAndTensor.first, outIndexAndTensor.first);

        variableModOrAlias |=
            std::any_of(aliasedRegions.begin(),
                        aliasedRegions.end(),
                        [](view::Region &r) { return !r.isEmpty(); });
      }
      logging::op::trace("[InheritOpAttributeHelper::getRequiredVirtualGraphId]"
                         " Op {} consumes variable tensor {} ({}), touches: {}",
                         op->debugName(),
                         varTensor->id,
                         inIndexAndTensor.second->id,
                         variableModOrAlias ? "yes" : "no");
      if (variableModOrAlias) {
        // Variable tensors force the VGID to be such that the weight
        // is not modified or aliased on any other VGID than the one where
        // the weight is stored.
        for (Op *consumer : varTensor->consumers.getOps()) {
          if (consumer != op && consumer->hasVirtualGraphId()) {
            for (auto &indices : consumer->input->indicesMap()) {
              if (indices.first == inIndexAndTensor.second) {
                auto rvgid =
                    consumer
                        ->getIntrospectionInVirtualGraphId(indices.second[0])
                        .first;
                if (rvgid != unusedVGraphId) {
                  if (requiredVgid) {
                    requiredVgid = std::min(*requiredVgid, rvgid);
                  } else {
                    requiredVgid = rvgid;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  return requiredVgid;
}

template <typename T>
typename std::set<std::tuple<InheritOpAttributeHelper::TraversalTensorInfo,
                             InheritOpAttributeHelper::InfoSortPriority,
                             T,
                             OpId>>::iterator
InheritOpAttributeHelper::getBestSetElement(
    std::set<OpId> &preferredOpIds,
    const std::set<std::tuple<TraversalTensorInfo, InfoSortPriority, T, OpId>>
        &set,
    bool allowNewOpId) {

  auto begin = set.begin();
  auto it    = begin;

  std::set<OpId> newPreferredIds;
  // Iterate set by priority
  while (it != set.end()) {
    if (preferredOpIds.find(std::get<3>(*it)) != preferredOpIds.end()) {
      break;
    }
    it++;
  }
  if (it != set.end()) {
    // All OpIds providing the same value can stay preferred
    T value = std::get<2>(*it);
    for (auto &entry : set) {
      if (value == std::get<2>(entry)) {
        newPreferredIds.insert(std::get<3>(entry));
      }
    }

    std::set<OpId> intersectPreferredIds;

    std::set_intersection(
        preferredOpIds.begin(),
        preferredOpIds.end(),
        newPreferredIds.begin(),
        newPreferredIds.end(),
        std::inserter(intersectPreferredIds, intersectPreferredIds.end()));

    preferredOpIds.clear();
    preferredOpIds = intersectPreferredIds;
    return it;
  }

  if (allowNewOpId) {
    // Fallback to set beginning
    T value = std::get<2>(*begin);

    // All OpIds providing the same value can be added as preferred
    for (auto &entry : set) {
      if (value == std::get<2>(entry)) {
        preferredOpIds.insert(std::get<3>(entry));
      }
    }

    return begin;
  } else {
    // No result
    return set.end();
  }
}

void InheritOpAttributeHelper::setAttributes() {
  bool inherited = false;

  OptionalVGraphId requiredVgid = getRequiredVirtualGraphId();

  // Preferred Op to inherit from, if available
  std::set<OpId> preferredOpIds;

  // Inherit attributes in order. Each inherited
  // attribute adds to the preferredOpIds set, which will preferentially be
  // used when inheriting the other attributes.

  if (!executionContextSet.empty()) {
    auto it =
        getBestSetElement(preferredOpIds, executionContextSet, !inherited);
    if (it != executionContextSet.end()) {
      op->settings.executionContext = std::get<2>(*it);
      inherited                     = true;
    }
  }

  if (pipelined && !pipelineStageSet.empty()) {
    auto it = getBestSetElement(preferredOpIds, pipelineStageSet, !inherited);
    if (it != pipelineStageSet.end()) {
      op->setPipelineStage(std::get<2>(*it));
      inherited = true;
    }
  }

  if (executionphased && !executionPhaseSet.empty()) {
    auto it = getBestSetElement(preferredOpIds, executionPhaseSet, !inherited);
    if (it != executionPhaseSet.end()) {
      op->setExecutionPhase(std::get<2>(*it));
      inherited = true;
    }
  }

  if (inheritSerializations && !batchSerializedPhaseSet.empty()) {
    auto it =
        getBestSetElement(preferredOpIds, batchSerializedPhaseSet, !inherited);
    if (it != batchSerializedPhaseSet.end()) {
      op->setBatchSerializedPhase(std::get<2>(*it));
      inherited = true;
    }
  }

  if (vgraphed && !vgidSet.empty() && !op->isIpuCopyOp()) {
    auto it = getBestSetElement(preferredOpIds, vgidSet, true);
    if (it != vgidSet.end()) {
      op->setVirtualGraphId(std::get<2>(*it));
      inherited = true;
    }
  }

  // If inheritance did not yield the correct VGID, rectify
  // Example where this happens:
  //__________________________ phase 0, vgid 0
  // Var0 ------------ Op
  //  |                |
  //__|________________|______ phase 1, vgid 1
  //  |              OpGrad
  //  |                |
  //__|________________|______ phase 2, vgid 0
  //  `------------ VarUpdate <- will inherit wrong phase and vgid
  //
  if (requiredVgid &&
      (!op->hasVirtualGraphId() || op->getVirtualGraphId() != *requiredVgid)) {
    logging::op::debug("[InheritOpAttributeHelper::setAttributes] Changing Op "
                       "{} placement to required VGID: {}",
                       op->debugName(),
                       *requiredVgid);
    op->setVirtualGraphId(requiredVgid);
    if (op->hasExecutionPhase()) {
      op->setExecutionPhase(op->getExecutionPhase() + 1);
    }
    inherited = true;
  }

  if (!inherited) {
    logging::op::info("[InheritOpAttributeHelper::setAttributes] Could not "
                      "inherit placement attributes to Op {}",
                      op->debugName());
  } else {
    if (logging::shouldLog(logging::Module::op, logging::Level::Trace)) {
      std::vector<std::string> preferredOps;

      for (auto &opId : preferredOpIds) {
        preferredOps.push_back(ir.getOp(opId)->debugName());
      }

      logging::op::trace("[InheritOpAttributeHelper::setAttributes] Op {} "
                         "inherited attributes from Ops {}",
                         op->debugName(),
                         preferredOps);
    }
  }
}

} // namespace popart
