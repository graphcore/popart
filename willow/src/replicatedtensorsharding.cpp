// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <popart/op/collectives/collectives.hpp>
#include <popart/op/exchange/hostcopy.hpp>
#include <popart/op/exchange/multiexchange.hpp>
#include <popart/op/exchange/remote.hpp>
#include <popart/pointercomparators.hpp>
#include <popart/replicatedtensorsharding.hpp>

namespace popart {

bool ReplicatedTensorShardingOpInfo::operator<(
    ReplicatedTensorShardingOpInfo const &rhs) const {
  return std::make_tuple(this->id, this->inIndices, this->outIndices) <
         std::make_tuple(rhs.id, rhs.inIndices, rhs.outIndices);
}

std::ostream &operator<<(std::ostream &output,
                         const ReplicatedTensorShardingOpInfo &rtsOpId) {
  output << "opId=" << rtsOpId.id;
  output << "[inIndices=" << rtsOpId.inIndices << ", ";
  output << "outIndices=" << rtsOpId.outIndices << "]";
  return output;
}

std::ostream &operator<<(std::ostream &output,
                         const ReplicatedTensorShardingGroup &rtsGroup) {
  output << "RTSGroup[";
  if (rtsGroup.shape.has_value()) {
    output << "shape=" << *(rtsGroup.shape) << ", ";
  }
  if (rtsGroup.metaShape.has_value()) {
    output << "metaShape=" << *(rtsGroup.metaShape) << ", ";
  }
  output << "remoteTensorIds=" << rtsGroup.remoteTensorIds << ", ";
  output << "linkedTensorIds=" << rtsGroup.collectiveLinkedTensorIds << ", ";
  output << "shardedTensorIds=" << rtsGroup.shardedTensorIds << ", ";
  output << "collectiveOpIds=" << rtsGroup.collectiveOpIds << ", ";
  output << "remoteOpIds=" << rtsGroup.exchangeOpIds;
  output << "]";
  return output;
}

ReplicatedTensorShardingTracer::ReplicatedTensorShardingTracer(const Ir &ir_)
    : ir(ir_) {}

bool ReplicatedTensorShardingTracer::TraceHelper::traceVisitor(Tensor *t) {
  // Check if a restart is needed because the start tensors have changed
  if (restart) {
    return false;
  }

  // Same shape and meta shape implies connected RTS domain
  bool shapeMatch = false;
  if (group.shape.has_value() && group.metaShape.has_value() &&
      t->info.shape() == *(group.shape) &&
      t->info.metaShape() == *(group.metaShape)) {
    shapeMatch = true;
    group.shardedTensorIds.insert(t->id);
  }

  if (shapeMatch) {
    // Check if the tensor is associated with a remote buffer, and if so,
    // add all other related tensors to the start tensors from which to traverse
    // the graph
    auto it = tensorRemoteBufferMap.find(t);
    if (it != tensorRemoteBufferMap.end()) {
      for (RemoteBufferId rbid : it->second) {
        for (auto t : remoteBufferTensorMap.at(rbid)) {
          if (t->info.shape() == group.shape &&
              t->info.metaShape() == group.metaShape) {
            addStartTensor(t);
            // Add any remote variables to the group
            registerRemoteVariables(t->getGraph().getIr(), rbid);
          }
        }
      }
    }
  }

  auto checkCollectiveLinkedTensor = [this](CollectivesBaseOp *collectiveOp,
                                            Tensor *t) {
    if (collectiveOp->hasCorrespondingLinkedIndexTensor(t)) {
      auto link = collectiveOp->getCorrespondingLinkedIndexTensor(t);
      group.collectiveLinkedTensorIds.insert(link->id);
      for (auto linkRoot : graphutils::rootTensors(link)) {
        group.collectiveLinkedTensorIds.insert(linkRoot->id);
      }
    }
  };

  for (Op *c : t->consumers.getOps()) {
    if (CollectivesBaseOp *collectiveOp =
            dynamic_cast<CollectivesBaseOp *>(c)) {
      auto indices  = collectiveOp->input->indices(t);
      bool isLinked = collectiveOp->isCollectiveLinkedIndexTensor(t);
      if (shapeMatch || isLinked) {
        checkCollectiveLinkedTensor(collectiveOp, t);
        group.collectiveOpIds[collectiveOp->id].id = collectiveOp->id;
      }
      if (shapeMatch) {
        group.collectiveOpIds[collectiveOp->id].inIndices.insert(
            indices.begin(), indices.end());
      }
    }

    if (shapeMatch) {
      if (ExchangeBaseOp *exchangeOp = dynamic_cast<ExchangeBaseOp *>(c)) {
        auto indices                           = exchangeOp->input->indices(t);
        group.exchangeOpIds[exchangeOp->id].id = exchangeOp->id;
        group.exchangeOpIds[exchangeOp->id].inIndices.insert(indices.begin(),
                                                             indices.end());
      }
    }
  }

  if (shapeMatch) {
    if (t->hasProducer()) {
      Op *p = t->getProducer();
      if (CollectivesBaseOp *collectiveOp =
              dynamic_cast<CollectivesBaseOp *>(p)) {
        auto indices = collectiveOp->output->indices(t);
        checkCollectiveLinkedTensor(collectiveOp, t);
        group.collectiveOpIds[collectiveOp->id].id = collectiveOp->id;
        group.collectiveOpIds[collectiveOp->id].outIndices.insert(
            indices.begin(), indices.end());
      }
      if (ExchangeBaseOp *exchangeOp = dynamic_cast<ExchangeBaseOp *>(p)) {
        auto indices                           = exchangeOp->output->indices(t);
        group.exchangeOpIds[exchangeOp->id].id = exchangeOp->id;
        group.exchangeOpIds[exchangeOp->id].outIndices.insert(indices.begin(),
                                                              indices.end());
      }
    }
  }

  if (group.shape.has_value() && group.metaShape.has_value() &&
      t->info.shape() == *(group.shape) &&
      t->info.metaShape() != *(group.metaShape)) {
    logging::opx::warn(
        "[ReplicatedTensorShardingTracer::getCollectiveLinkedGroup] "
        "tensor {} matches in shape ({} vs. {}) but not "
        "meta-shape ({} vs. {})",
        t->id,
        t->info.shape(),
        *(group.shape),
        t->info.metaShape(),
        *(group.metaShape));
  }

  logging::opx::trace(
      "[ReplicatedTensorShardingTracer::getCollectiveLinkedGroup] "
      "visiting: {})",
      t->id);

  return true;
}

void ReplicatedTensorShardingTracer::TraceHelper::addStartTensor(
    Tensor *start) {
  auto oldSize = startTensors.size();
  startTensors.insert(start);
  if (startTensors.size() != oldSize) {
    restart = true;
  }
}

void ReplicatedTensorShardingTracer::TraceHelper::addStartTensors(
    const std::set<Tensor *, PTensorCmp> &start) {
  auto oldSize = startTensors.size();
  startTensors.insert(start.begin(), start.end());
  if (startTensors.size() != oldSize) {
    restart = true;
  }
}

std::vector<Tensor *>
ReplicatedTensorShardingTracer::TraceHelper::getStartTensors() const {
  std::vector<Tensor *> start;
  start.reserve(startTensors.size());
  start.insert(start.end(), startTensors.begin(), startTensors.end());
  return start;
}

void ReplicatedTensorShardingTracer::TraceHelper::registerRemoteBuffers(
    const Ir &ir) {
  auto allOps = ir.getAllOps();
  for (auto op : allOps) {
    if (auto exchangeOp = dynamic_cast<ExchangeBaseOp *>(op)) {
      for (int i = 0; i < exchangeOp->getNumExchanges(); ++i) {
        auto descriptor = exchangeOp->getExchangeDescriptor(i);
        if (descriptor.getRemoteBufferId() > -1) {
          auto inIndices      = exchangeOp->descriptorIndexToInIndices(i);
          auto outIndices     = exchangeOp->descriptorIndexToOutIndices(i);
          RemoteBufferId rbid = descriptor.getRemoteBufferId();
          for (auto in : inIndices) {
            tensorRemoteBufferMap[exchangeOp->inTensor(in)].insert(rbid);
            remoteBufferTensorMap[rbid].insert(exchangeOp->inTensor(in));
          }
          for (auto out : outIndices) {
            tensorRemoteBufferMap[exchangeOp->outTensor(out)].insert(rbid);
            remoteBufferTensorMap[rbid].insert(exchangeOp->outTensor(out));
          }
        }
      }
    }
  }
}

void ReplicatedTensorShardingTracer::TraceHelper::registerRemoteVariables(
    const Ir &ir,
    RemoteBufferId rbid) {
  for (auto &var :
       ir.getMainGraph().getTensors().getOfType(TensorType::Variable)) {
    if (var->tensorLocationInfo.isRemote()) {
      RemoteBufferId varRbid =
          var->tensorLocationInfo.getRemoteBufferInfo().first;
      if (varRbid == rbid) {
        group.remoteTensorIds.insert(var->id);
      }
    }
  }
}

bool ReplicatedTensorShardingTracer::TraceHelper::traceFilter(Op *op,
                                                              Tensor *tq,
                                                              Tensor *tn) {

  if (restart) {
    return false;
  }

  // Subgraph inputs/outputs should be traversed
  if (op->isConvertibleTo<SubgraphOp>()) {
    return true;
  }

  // Traverse ops based on whether tn and tq are linked according to RTS indices
  // or a collective linked tensor (for collective ops only)
  auto rtsIndices = op->getReplicatedTensorShardingIndices();

  std::vector<InIndex> tqIn;
  std::vector<InIndex> tnIn;
  std::vector<OutIndex> tqOut;
  std::vector<OutIndex> tnOut;

  if (op->input->contains(tq)) {
    tqIn = op->input->indices(tq);
  }
  if (op->input->contains(tn)) {
    tnIn = op->input->indices(tn);
  }
  if (op->output->contains(tq)) {
    tqOut = op->output->indices(tq);
  }
  if (op->output->contains(tn)) {
    tnOut = op->output->indices(tn);
  }

  for (auto rtsIndex : rtsIndices) {
    bool tqInSet = false;
    bool tnInSet = false;
    for (auto index : tqIn) {
      tqInSet |= rtsIndex.first.find(index) != rtsIndex.first.end();
    }
    for (auto index : tqOut) {
      tqInSet |= rtsIndex.second.find(index) != rtsIndex.second.end();
    }
    for (auto index : tnIn) {
      tnInSet |= rtsIndex.first.find(index) != rtsIndex.first.end();
    }
    for (auto index : tnOut) {
      tnInSet |= rtsIndex.second.find(index) != rtsIndex.second.end();
    }
    if (tqInSet && tnInSet) {
      // Input and output tensor are in the same RTS domain
      return true;
    } else if (CollectivesBaseOp *collectiveOp =
                   dynamic_cast<CollectivesBaseOp *>(op)) {
      // For collective ops, if the two tensors are not in the same RTS domain,
      // it could still be that one is a linked tensor for another
      if (tnInSet && collectiveOp->hasCorrespondingLinkedIndexTensor(tn)) {
        // tn is an RTS tensor and tq is the associated linked tensor
        if (collectiveOp->getCorrespondingLinkedIndexTensor(tn)->id == tq->id) {
          return true;
        }
      } else if (tqInSet &&
                 collectiveOp->hasCorrespondingLinkedIndexTensor(tq)) {
        // tq is an RTS tensor and tn is the associated linked tensor
        if (collectiveOp->getCorrespondingLinkedIndexTensor(tq)->id == tn->id) {
          return true;
        }
      }
    }
  }
  return false;
}

bool ReplicatedTensorShardingTracer::hasGroup(
    const ReplicatedTensorShardingOpInfo &opInfo) const {
  // A group contains a subset of input and output indices for a given op
  // hasGroup checks whether there already exists a group for a given opId which
  // exactly covers all the input and output indices of opInfo
  for (auto entry : opIdGroupMap) {
    ReplicatedTensorShardingOpInfo info = entry.first;
    if (info.id == opInfo.id) {
      bool allInputsAreIngroup  = std::includes(info.inIndices.begin(),
                                               info.inIndices.end(),
                                               opInfo.inIndices.begin(),
                                               opInfo.inIndices.end());
      bool allOutputsAreIngroup = std::includes(info.outIndices.begin(),
                                                info.outIndices.end(),
                                                opInfo.outIndices.begin(),
                                                opInfo.outIndices.end());
      if (allInputsAreIngroup && allOutputsAreIngroup) {
        return true;
      }
    }
  }
  return false;
}

bool ReplicatedTensorShardingTracer::hasGroup(const TensorId &tensorId) const {
  auto it = tensorIdGroupMap.find(tensorId);
  return it != tensorIdGroupMap.end();
}

const ReplicatedTensorShardingGroup &ReplicatedTensorShardingTracer::getGroup(
    const ReplicatedTensorShardingOpInfo &opInfo) const {
  // getGroup returns the group for a given opId which
  // exactly covers all the input and output indices of opInfo
  // if no such group exists an error is thrown
  for (auto entry : opIdGroupMap) {
    ReplicatedTensorShardingOpInfo info = entry.first;
    if (info.id == opInfo.id) {
      bool allInputsAreIngroup  = std::includes(info.inIndices.begin(),
                                               info.inIndices.end(),
                                               opInfo.inIndices.begin(),
                                               opInfo.inIndices.end());
      bool allOutputsAreIngroup = std::includes(info.outIndices.begin(),
                                                info.outIndices.end(),
                                                opInfo.outIndices.begin(),
                                                opInfo.outIndices.end());
      if (allInputsAreIngroup && allOutputsAreIngroup) {
        return groups.at(entry.second);
      }
    }
  }
  throw error(
      "[ReplicatedTensorShardingGroup "
      "&ReplicatedTensorShardingTracer::getGroup] OpInfo {} is not part "
      "of a replication tensor sharding (RTS) group",
      opInfo);
}

const ReplicatedTensorShardingGroup &
ReplicatedTensorShardingTracer::getGroup(const TensorId &tensorId) const {
  auto it = tensorIdGroupMap.find(tensorId);
  if (it == tensorIdGroupMap.end()) {
    throw error(
        "[ReplicatedTensorShardingGroup "
        "&ReplicatedTensorShardingTracer::getGroup] TensorId {} is not part "
        "of a replication tensor sharding (RTS) group",
        tensorId);
  } else {
    return groups.at(it->second);
  }
}

void ReplicatedTensorShardingTracer::trace(
    const std::set<Tensor *, PTensorCmp> &startTensors) {

  ReplicatedTensorShardingGroupId existingGroupIdForTensors = -1;

  bool allGroupsSet = std::all_of(
      startTensors.begin(),
      startTensors.end(),
      [this, &existingGroupIdForTensors](Tensor *t) {
        if (hasGroup(t->id)) {
          auto &group = getGroup(t->id);
          if (existingGroupIdForTensors == -1) {
            existingGroupIdForTensors = group.id;
            return true;
          } else {
            if (group.id == existingGroupIdForTensors) {
              return true;
            } else {
              throw internal_error(
                  "[ReplicatedTensorShardingTracer::trace] Tensors are already "
                  "in groups, but are conflicting {}==[ID: {}] vs. [ID: {}]",
                  t->id,
                  group.id,
                  existingGroupIdForTensors);
            }
          }
        }
        return false;
      });

  bool anyGroupsSet =
      std::any_of(startTensors.begin(), startTensors.end(), [this](Tensor *t) {
        return hasGroup(t->id);
      });

  if (anyGroupsSet && !allGroupsSet) {
    throw internal_error(
        "[ReplicatedTensorShardingTracer::trace] Some but not all "
        "tensors have a group set. This is unexpected.");
  }

  if (!anyGroupsSet) {

    TraceHelper helper;

    // Set the group shape from the start tensors
    for (auto t : startTensors) {
      if (!t->info.metaShape().empty()) {
        if (!helper.group.shape.has_value()) {
          helper.group.shape = t->info.shape();
        } else if (helper.group.shape != t->info.shape()) {
          throw internal_error(
              "[ReplicatedTensorShardingTracer::trace] The start tensors for "
              "the trace have incompatible shapes");
        }
        if (!helper.group.metaShape.has_value()) {
          helper.group.metaShape = t->info.metaShape();
        } else if (helper.group.metaShape != t->info.metaShape()) {
          throw internal_error(
              "[ReplicatedTensorShardingTracer::trace] The start tensors for "
              "the trace have incompatible metaShapes");
        }
      }
    }

    helper.registerRemoteBuffers(ir);

    helper.addStartTensors(startTensors);

    auto traceVisitorCallback = [&helper](Tensor *t) {
      return helper.traceVisitor(t);
    };
    auto traceFilterCallback = [&helper](Op *op, Tensor *tq, Tensor *tn) {
      return helper.traceFilter(op, tq, tn);
    };

    // Due to remote buffers, which are not captured in regular graph traversal,
    // have to restart until all remote buffers have been traversed too
    helper.restart = true;
    while (helper.restart) {
      helper.restart = false;
      graphutils::traverse(helper.getStartTensors(),
                           traceVisitorCallback,
                           traceFilterCallback,
                           graphutils::TraversalType::DepthFirst,
                           graphutils::VisitType::Pre,
                           graphutils::TraversalDirection::ForwardBackward,
                           graphutils::TraverseCallSites::All);
    }

    registerGroup(helper.group);
  }
}

void ReplicatedTensorShardingTracer::registerGroup(
    ReplicatedTensorShardingGroup &group) {
  auto index = groups.size();
  group.id   = index;
  groups.push_back(group);

  for (auto entry : group.remoteTensorIds) {
    tensorIdGroupMap[entry] = index;
  }
  for (auto entry : group.collectiveLinkedTensorIds) {
    tensorIdGroupMap[entry] = index;
  }
  for (auto entry : group.shardedTensorIds) {
    tensorIdGroupMap[entry] = index;
  }
  for (auto entry : group.collectiveOpIds) {
    opIdGroupMap[entry.second] = index;
  }
  for (auto entry : group.exchangeOpIds) {
    opIdGroupMap[entry.second] = index;
  }

  logging::trace(
      "[ReplicatedTensorShardingTracer::registerGroup] Registered group: {}",
      group);
}

} // namespace popart
