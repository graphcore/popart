// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/logging/timepartitionlogger.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <popart/alias/aliasmodel.hpp>
#include <popart/alias/aliasmodelgrower.hpp>
#include <popart/ces/constexpr.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/graphutils.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/exchange/hostcopy.hpp>
#include <popart/op/exchange/multiexchange.hpp>
#include <popart/op/init.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/loop.hpp>
#include <popart/op/restore.hpp>
#include <popart/pointercomparators.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensornames.hpp>
#include <popart/util.hpp>
#include <popart/variablesettings.hpp>

#include "popart/aliases.hpp"
#include "popart/basicoptionals.hpp"
#include "popart/chains.hpp"
#include "popart/debugcontext.hpp"
#include "popart/graphid.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op/exchange/exchange.hpp"
#include "popart/op/subgraph.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/region.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensorlocation.hpp"
#include "popart/tensors.hpp"
#include "popart/vertex.hpp"

namespace popart {

Ir &Tensor::getIr() { return getGraph().getIr(); }
const Ir &Tensor::getIr() const { return getGraph().getIr(); }

bool Tensor::consumersAllPreLoss() const {
  for (Op *consumer : consumers.getOps()) {
    if (consumer->scheduledPreLoss == ScheduledPreLoss::No) {
      return false;
    }
  }
  return true;
}

bool Tensor::isAliased() const {

  constexpr const char *const ctxt{"Tensor::isAliased"};
  logging::ir::trace("{} for Tensor {},", ctxt, str());
  auto scopedStopwatch = getIr().timePartitionLogger().scopedStopwatch(ctxt);

  for (Op *consumer : consumers.getOps()) {
    for (InIndex in : consumer->input->indices(graph.getTensors().get(id))) {
      for (auto outEntry : consumer->output->indicesMap()) {
        for (OutIndex out : outEntry.second) {
          auto regions = consumer->aliases(in, out);
          if (!std::all_of(regions.begin(),
                           regions.end(),
                           [](const view::Region &r) { return r.isEmpty(); })) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

bool Tensor::isModified(bool considerLoopInput) const {

  constexpr const char *const ctxt{"Tensor::isModified"};
  logging::ir::trace("{} for Tensor {},", ctxt, str());
  auto scopedStopwatch = getIr().timePartitionLogger().scopedStopwatch(ctxt);

  for (Op *consumer : consumers.getOps()) {
    for (InIndex in : consumer->input->indices(graph.getTensors().get(id))) {
      auto regions = consumer->modifies(in);
      if (!std::all_of(
              regions.begin(), regions.end(), [](const view::Region &r) {
                return r.isEmpty() ||
                       r.getAccessType() == view::AccessType::Read;
              })) {
        return true;
      }
    }
  }
  // All explicit loop inputs will be modified within the subgraph
  if (considerLoopInput && isExplicitLoopInput()) {
    return true;
  }
  return false;
}

view::Regions Tensor::modifiedRegionsByOps(std::vector<OpId> opIds,
                                           Aliases &aliases) const {
  std::vector<Op *> ops;
  ops.reserve(opIds.size());
  for (auto &opId : opIds) {
    ops.push_back(graph.getOp(opId));
  }
  return modifiedRegionsByOps(ops, aliases);
}

view::Regions Tensor::modifiedRegionsByOps(std::vector<Op *> ops,
                                           Aliases &aliases) const {

  constexpr const char *const ctxt{"Tensor::modifiedRegionsByOps"};
  logging::ir::trace("{} for Tensor {},", ctxt, str());
  auto scopedStopwatch = getIr().timePartitionLogger().scopedStopwatch(ctxt);

  // t0: non-const pointer to this
  Tensor *t0  = graph.getTensors().get(id);
  auto chains = aliases.aliasChainsFrom(t0);

  std::map<Op *, view::Regions> opToT0ReadRegions;
  std::map<Op *, view::Regions> opToT0ModifiedRegions;

  std::set<Tensor *, PTensorCmp> aliasedTensors;
  aliasedTensors.insert(t0);

  // All chains from t0
  for (auto &chain : chains) {
    // All aliases t1 of t0
    Tensor *t1 = chain.first;
    aliasedTensors.insert(t1);
  }

  for (auto t1 : aliasedTensors) {
    // All consumers of t1
    for (Op *consumer : t1->consumers.getOps()) {

      view::Regions subModifiedRegions1;
      view::Regions subModifiedRegions0;
      view::Regions subReadRegions1;
      view::Regions subReadRegions0;

      auto indices = consumer->input->indices(t1);

      for (InIndex index : indices) {
        // Any consumer modified region of t1
        auto mRegions = consumer->modifies(index);
        // Any consumer accessed region of t1
        auto rRegions = consumer->uses(index);
        // Any consumer accessed but not modified region of t1
        view::Regions dRegions;

        for (auto &r0 : rRegions) {
          auto rs = r0.sub(mRegions);
          dRegions.insert(dRegions.begin(), rs.begin(), rs.end());
        }
        dRegions = view::mergeRegions(dRegions);

        // Collect modified regions of t1
        subModifiedRegions1.insert(
            subModifiedRegions1.end(), mRegions.begin(), mRegions.end());

        // Collect read regions of t1
        subReadRegions1.insert(
            subReadRegions1.end(), dRegions.begin(), dRegions.end());
      }

      // Convert regions of t1 to regions of t0
      subModifiedRegions1 = view::mergeRegions(subModifiedRegions1);
      subReadRegions1     = view::mergeRegions(subReadRegions1);

      if (t0 == t1) {
        subModifiedRegions0 = subModifiedRegions1;
        subReadRegions0     = subReadRegions1;
      } else {
        for (auto &subModifiedRegion1 : subModifiedRegions1) {
          auto regions =
              aliases.getChainsFromTo(t1, t0).apply(subModifiedRegion1);
          subModifiedRegions0.insert(
              subModifiedRegions0.end(), regions.begin(), regions.end());
        }
        for (auto &subReadRegion1 : subReadRegions1) {
          auto regions = aliases.getChainsFromTo(t1, t0).apply(subReadRegion1);
          subReadRegions0.insert(
              subReadRegions0.end(), regions.begin(), regions.end());
        }
      }
      subModifiedRegions0 = view::mergeRegions(subModifiedRegions0);
      subReadRegions0     = view::mergeRegions(subReadRegions0);

      // Register what part of t0 the consumer of t1 modifies (indirectly)
      if (std::any_of(subModifiedRegions0.begin(),
                      subModifiedRegions0.end(),
                      [](const view::Region &r) { return !r.isEmpty(); })) {
        auto &regions = opToT0ModifiedRegions[consumer];
        regions.insert(regions.end(),
                       subModifiedRegions0.begin(),
                       subModifiedRegions0.end());
      }
      if (std::any_of(subReadRegions0.begin(),
                      subReadRegions0.end(),
                      [](const view::Region &r) { return !r.isEmpty(); })) {
        auto &regions = opToT0ReadRegions[consumer];
        regions.insert(
            regions.end(), subReadRegions0.begin(), subReadRegions0.end());
      }
    }
  }

  // Assemble all t0 modified regions
  view::Regions regionsReadUpUntilNow;
  view::Regions modifiedRegions;
  view::AccessType accessType = view::AccessType::None;

  // As soon as a consumer modified the whole input, we can stop
  for (Op *op : ops) {
    {
      auto it = opToT0ReadRegions.find(op);
      if (it != opToT0ReadRegions.end()) {
        auto opReadRegions = view::mergeRegions(it->second);
        regionsReadUpUntilNow.insert(regionsReadUpUntilNow.end(),
                                     opReadRegions.begin(),
                                     opReadRegions.end());
        regionsReadUpUntilNow = view::mergeRegions(regionsReadUpUntilNow);
      }
    }
    {
      auto it = opToT0ModifiedRegions.find(op);
      if (it != opToT0ModifiedRegions.end()) {
        auto opModifiedRegions = view::mergeRegions(it->second);
        modifiedRegions.insert(modifiedRegions.end(),
                               opModifiedRegions.begin(),
                               opModifiedRegions.end());
        modifiedRegions = view::mergeRegions(modifiedRegions);

        // If any newly modified region overlaps with a previously read region,
        // conservatively change access type to Read/ReadWrite
        if (std::any_of(opModifiedRegions.begin(),
                        opModifiedRegions.end(),
                        [&regionsReadUpUntilNow](view::Region &r0) {
                          return std::any_of(
                              regionsReadUpUntilNow.begin(),
                              regionsReadUpUntilNow.end(),
                              [&r0](view::Region &r1) {
                                return !r0.intersect(r1).isEmpty();
                              });
                        })) {
          accessType = view::combine({accessType, view::AccessType::Read});
        }

        for (auto &r : opModifiedRegions) {
          view::AccessType regionAccessType = r.getAccessType();
          if (!r.isEmpty() && (regionAccessType == view::AccessType::None ||
                               regionAccessType == view::AccessType::Read)) {
            throw error("Unexpected modified region access type None or Read");
          }
          accessType = view::combine({accessType, regionAccessType});
        }
        if (modifiedRegions.size() > 0 &&
            modifiedRegions.front() ==
                view::Region::getFull(t0->info.shape()) &&
            accessType == view::AccessType::Write) {
          // The whole input tensor has been touched, conclude
          //  If the whole tensor has been write-accessed first, we say that
          //  the list of ops passed to this method consume the tensor
          //  write-only. If any read access to the tensor happens before the
          //  write-only access, the modified tensor is read-write. Read-only
          //  does not make sense, since we ask about modified regions.
          // Examples:
          //  1.) VarUpdate will cause read-write access to modified input.
          //  2.) RemoteLoad will cause write-only access to modified input.
          break;
        }
      }
    }
  }

  // Update access type
  for (auto &r : modifiedRegions) {
    r.setAccessType(accessType);
  }

  // Return all regions touched
  return modifiedRegions;
}

std::set<Op *, POpCmp> Tensor::getInplaceModifiers() const {
  std::set<Op *, POpCmp> ops;
  anyAlias([&ops](Tensor *t) {
    auto consumers = t->consumers.getOps();
    for (auto c : consumers) {
      if (c->modifies()) {
        ops.insert(c);
      }
    }
    // Continue until all aliases have been visited
    return false;
  });
  return ops;
}

VGraphId Tensor::getVirtualGraphIdUnsafe() const {
  std::set<OpId> visited;
  return getVirtualGraphIdAndTileSetUnsafe(visited).first;
}

VGraphIdAndTileSet Tensor::getVirtualGraphIdAndTileSetUnsafe() const {
  std::set<OpId> visited;
  return getVirtualGraphIdAndTileSetUnsafe(visited);
}

VGraphIdAndTileSet
Tensor::getVirtualGraphIdAndTileSetUnsafe(std::set<OpId> &visited) const {

  constexpr const char *const ctxt{"Tensor::getVirtualGraphIdAndTileSetUnsafe"};
  logging::ir::trace("{} for Tensor {} (visited {}),", ctxt, str(), visited);
  auto scopedStopwatch = getIr().timePartitionLogger().scopedStopwatch(ctxt);

  VGraphIdAndTileSetSet vgidSet;

  // If this Tensor has a Producer, use its VirtualGraphId if it has one
  if (hasProducer()) {
    // special case of IPUCopy producer
    auto ipucopy = dynamic_cast<IpuCopyOp *>(getProducer());
    if (ipucopy) {
      vgidSet.insert({ipucopy->getDestIpu(), ipucopy->settings.tileSet});
    } else {
      if (visited.find(getProducer()->id) == visited.end()) {
        for (auto &indices : getProducer()->output->indicesMap()) {
          if (indices.first == this) {
            visited.insert(getProducer()->id);
            vgidSet.insert(getProducer()->getIntrospectionOutVirtualGraphId(
                indices.second[0], visited));
          }
        }
        visited.insert(producer->id);
      }
    }
  } else if (isGraphInput()) {
    // Graph input, derive from call site
    auto callSites = getGraph().getCallSiteOps();
    for (Op *callSite : callSites) {
      if (callSite->isConvertibleTo<SubgraphOp>() &&
          visited.find(callSite->id) == visited.end()) {
        visited.insert(callSite->id);
        auto opInIndex = callSite->subgraphInToOpInIndex(
            callSite->getCalledGraphIndex(getGraph().getGraphId()),
            getGraphInputIndex());
        if (callSite->hasInput(opInIndex)) {
          vgidSet.insert(callSite->input->tensor(opInIndex)
                             ->getVirtualGraphIdAndTileSetUnsafe(visited));
        } else {
          vgidSet.insert({callSite->hasVirtualGraphId()
                              ? callSite->getVirtualGraphId()
                              : unusedVGraphId,
                          callSite->settings.tileSet});
        }
      }
    }
  }

  if (vgidSet.empty() || vgidSet.begin()->first == unusedVGraphId) {

    // If a consumer is an IPUCopy, we can derive the virtual graph id from it.
    for (Op *consumer : consumers.getOps()) {
      auto ipucopy = dynamic_cast<IpuCopyOp *>(consumer);
      if (ipucopy) {
        if (visited.find(consumer->id) == visited.end()) {
          vgidSet.insert(
              {ipucopy->getSourceIpus().at(id), ipucopy->settings.tileSet});
          visited.insert(consumer->id);
        }
      }
    }

    // Try to get the virtual graph id from a consumer.
    for (Op *consumer : consumers.getOps()) {
      if (visited.find(consumer->id) == visited.end()) {
        for (auto &indices : consumer->input->indicesMap()) {
          if (indices.first->id == this->id) {
            vgidSet.insert(consumer->getIntrospectionInVirtualGraphId(
                indices.second[0], visited));
            visited.insert(consumer->id);
          }
        }
      }
      visited.insert(consumer->id);
    }
  }

  if (vgidSet.empty()) {
    // No virtual graph ID and tile set determined
    vgidSet.insert({unusedVGraphId, TileSet::Undefined});
  }

  logging::ir::trace("{} for Tensor {} (result {}),", ctxt, str(), vgidSet);

  return *vgidSet.begin();
}

VGraphIdAndTileSet
Tensor::getVirtualGraphIdAndTileSet(std::set<OpId> &visited) const {
  auto vgid = getVirtualGraphIdAndTileSetUnsafe(visited);
  if (vgid == VGraphIdAndTileSet(unusedVGraphId, TileSet::Undefined) ||
      vgid == VGraphIdAndTileSet(unusedVGraphId, TileSet::Compute)) {
    throw error("Invalid call to getVirtualGraphId, Tensor does not have one");
  }
  return vgid;
}

VGraphId Tensor::getVirtualGraphId() const {
  std::set<OpId> visited;
  auto vgid = getVirtualGraphIdAndTileSetUnsafe(visited);
  if (vgid == VGraphIdAndTileSet(unusedVGraphId, TileSet::Undefined) ||
      vgid == VGraphIdAndTileSet(unusedVGraphId, TileSet::Compute)) {
    throw error("Invalid call to getVirtualGraphId, Tensor does not have one");
  }
  return vgid.first;
}

bool Tensor::hasVirtualGraphId() const {
  return getVirtualGraphIdUnsafe() != unusedVGraphId;
}

std::vector<char> Tensor::getDataViaGraphTraversal() const {

  constexpr const char *const ctxt{"Tensor::getDataViaGraphTraversal"};
  logging::ir::trace("{} for Tensor {},", ctxt, str());
  auto scopedStopwatch = getIr().timePartitionLogger().scopedStopwatch(ctxt);

  Tensor *thisTensor = graph.getTensors().get(id);

  std::vector<Tensor *> start;
  start.push_back(thisTensor);

  std::vector<Tensor *> roots;
  std::set<Tensor *> tensorsOnPath;
  tensorsOnPath.insert(thisTensor);

  // Find roots
  graphutils::traverse(
      start,
      [&roots, &tensorsOnPath](Tensor *t) {
        tensorsOnPath.insert(t);
        if (t->hasTensorData()) {
          roots.push_back(t);
        }
        return true;
      },
      [](Op *op, Tensor *t0, Tensor *t1) { return true; },
      graphutils::TraversalType::DepthFirst,
      graphutils::VisitType::Pre,
      graphutils::TraversalDirection::Backward);

  logging::ir::trace("[Tensor::getDataViaGraphTraversal] Tensor {}: {} roots, "
                     "{} tensors on path",
                     id,
                     roots.size(),
                     tensorsOnPath.size());

  bool changed = true;
  while (changed) {
    changed = false;
    // Propagate const data
    graphutils::traverse(
        roots,
        [&changed, &tensorsOnPath](Tensor *t) {
          bool hasData = t->hasTensorData();
          if (!hasData) {
            if (tensorsOnPath.find(t) != tensorsOnPath.end()) {
              if (t->hasProducer()) {
                // Producer const expr resolving
                if (ConstExprOpManager::hasConstExprOp(t->getProducer())) {
                  bool allInputsResolved = true;
                  for (auto inTensor : t->getProducer()->input->tensors()) {
                    if (!inTensor->hasTensorData()) {
                      logging::ir::trace(
                          "[Tensor::getDataViaGraphTraversal] Tensor {}: no "
                          "tensor "
                          "data. Not all inputs resolved for tensor {}.",
                          inTensor->id,
                          t->id);
                      allInputsResolved = false;
                    }
                  }
                  if (allInputsResolved) {
                    auto ceOp =
                        ConstExprOpManager::createConstExprOp(t->getProducer());
                    t->setTensorData(t->info, ceOp->compute().data());
                    changed = true;
                  }
                }
              } else if (t->isGraphInput()) {
                // Caller const expr resolving
                auto callSites = t->getGraph().getCallSiteOps();

                bool callSitesAgree = true;

                std::vector<std::vector<char>> constData;

                for (Op *c : callSites) {
                  for (int i = 0; i < c->getCalledGraphs().size(); ++i) {
                    if (c->getCalledGraphs().at(i)->id == t->getGraph().id) {
                      InIndex index =
                          c->subgraphInToOpInIndex(i, t->getGraphInputIndex());
                      if (c->hasInput(index) &&
                          c->inTensor(index)->hasTensorData()) {
                        // Technically, if the graph is well-formed,
                        // we'd expect each call site to produce the
                        // same const value here, but it's not guaranteed.
                        const char *ptr = static_cast<const char *>(
                            c->inTensor(index)->tensorData()->data());
                        std::vector<char> data;
                        data.assign(ptr, ptr + t->info.nbytes());
                        constData.push_back(data);
                        callSitesAgree &= constData.at(0) ==
                                          constData.at(constData.size() - 1);
                      }
                    }
                  }
                }

                if (constData.size() > 0 && callSitesAgree) {
                  t->setTensorData(
                      t->info, static_cast<void *>(constData.front().data()));
                  changed = true;
                }
              }
            } else {
              // Not interested in propagating through this tensor
              // (was not on the backward path)
              return false;
            }
          }
          hasData = t->hasTensorData();

          logging::ir::trace(
              "[Tensor::getDataViaGraphTraversal] Tensor {} {} constant data.",
              t->id,
              hasData ? "has" : "does not have");

          if (hasData) {
            // Continue forward propagation
            return true;
          } else {
            // Stop forward propagation
            return false;
          }
        },
        [](Op *op, Tensor *t0, Tensor *t1) { return true; },
        graphutils::TraversalType::BreadthFirst,
        graphutils::VisitType::Pre,
        graphutils::TraversalDirection::Forward);
  }

  if (!hasTensorData()) {
    throw error("[Tensor::getDataViaGraphTraversal] Could not work out tensor "
                "data for {}.",
                id);
  } else {
    const char *ptr = static_cast<const char *>(tensorData()->data());
    std::vector<char> data;
    data.assign(ptr, ptr + info.nbytes());
    return data;
  }
}

void Tensor::setTensorLocationInfo(
    TensorLocation &tLocation,
    std::pair<RemoteBufferId, RemoteBufferIndex> &remoteBufferInfo) {

  tensorLocationInfo.setRemote(tLocation.isRemote());
  tensorLocationInfo.setSharded(tLocation.replicatedTensorSharding ==
                                ReplicatedTensorSharding::On);
  tensorLocationInfo.setRemoteBufferInfo(remoteBufferInfo.first,
                                         remoteBufferInfo.second);
}

std::set<PipelineStage> Tensor::getPipelineStages() const {
  auto result = consumers.getPipelineStages();
  if (hasProducer() && getProducer()->hasPipelineStage()) {
    auto ps = getProducer()->getPipelineStage();
    // An IpuCopyOp in pipeline stage N, produces a tensor ready to be consumed
    // in pipeline stage N+1.
    if (getProducer()->isConvertibleTo<IpuCopyOp>()) {
      ps++;
    }
    result.insert(ps);
  }
  return result;
}

int Tensor::getBatchAxisFromOp(Op *op,
                               bool isConsumer,
                               int proposedAxis) const {
  std::vector<int> indices;
  // All the input (output) indices relative to this tensor
  if (isConsumer) {
    indices = op->input->indices(graph.getTensors().get(id));
  } else {
    indices = op->output->indices(graph.getTensors().get(id));
  }
  for (int idx : indices) {
    int axis = isConsumer ? op->getInBatchAxis(idx) : op->getOutBatchAxis(idx);
    if (proposedAxis == -1) {
      // Not yet set
      proposedAxis = axis;
    } else if (axis != proposedAxis) {
      // Inconcistency between different indices
      std::stringstream ss;
      ss << "Batch axis inconsistent for tensor " << id;
      ss << ". It's set to both " << proposedAxis << " and " << axis;
      if (isConsumer) {
        ss << ". There may be an inconsistency between the consumer Ops.";
      } else {
        ss << " from producer Op " << op->opid << ".";
      }
      throw error(ss.str());
    }
  }
  // Sanity check the value
  if (proposedAxis >= info.rank()) {
    return -1;
  }
  return proposedAxis;
}

int Tensor::getBatchAxis() const {
  int proposedAxis = -1;
  // If this Tensor has a Producer, get the batch axis from it
  if (hasProducer()) {
    proposedAxis = getBatchAxisFromOp(getProducer(), false, proposedAxis);
    if (getProducer()->isConvertibleTo<InitOp>()) {
      // InitOp decides batch axis ultimately
      return proposedAxis;
    }
    if (proposedAxis > -1) {
      return proposedAxis;
    }
  }

  // Check the value of batch axis for this tensor from the consumers
  for (Op *consumer : consumers.getOps()) {
    proposedAxis = getBatchAxisFromOp(consumer, true, proposedAxis);
    if (proposedAxis > -1) {
      return proposedAxis;
    }
  }
  return proposedAxis;
}

std::ostream &operator<<(std::ostream &os, const TensorType &tt) {
  switch (tt) {
  case TensorType::ActGrad:
    os << "ActGrad";
    break;
  case TensorType::Const:
    os << "Const";
    break;
  case TensorType::Stream:
    os << "Stream";
    break;
  case TensorType::Unknown:
    os << "Unknown";
    break;
  case TensorType::Variable:
    os << "Variable";
    break;
  case TensorType::N:
  default:
    os << "Undefined";
    break;
  }

  return os;
}

std::unique_ptr<Tensor> Tensor::clone(Graph &graph_) const {
  if (tensorType() == TensorType::Variable) {
    std::unique_ptr<Tensor> theClone(
        new Tensor("clone_" + id, variableSettings, graph_, getDebugInfo()));
    theClone->info = info;
    return theClone;
  }
  {
    std::unique_ptr<Tensor> theClone(
        new Tensor("clone_" + id, tensorType(), graph_, getDebugInfo()));
    theClone->info = info;
    return theClone;
  }
}

Consumers::Consumers(Tensor *tensorConsumed_)
    : tensorConsumed(tensorConsumed_) {}

std::set<PipelineStage> Consumers::getPipelineStages() const {
  std::set<PipelineStage> stages;
  for (auto op : getOps()) {
    if (op->hasPipelineStage()) {
      stages.insert(op->getPipelineStage());
    }
  }

  return stages;
}

std::set<VGraphId> Consumers::getVirtualGraphIds() const {
  std::set<VGraphId> vgIDs;
  for (auto op : getOps()) {
    if (IpuCopyOp *ipuCopyOp = dynamic_cast<IpuCopyOp *>(op)) {
      auto &srcMap = ipuCopyOp->getSourceIpus();
      for (auto src : srcMap) {
        vgIDs.insert(src.second);
      }
    } else if (op->hasVirtualGraphId()) {
      vgIDs.insert(op->getVirtualGraphId());
    }
  }
  return vgIDs;
}

OptionalPipelineStage Consumers::findLowestPipelineStage() const {
  auto stages = getPipelineStages();

  if (stages.size() == 0) {
    return {};
  } else {
    return *std::min_element(stages.begin(), stages.end());
  }
}

OptionalPipelineStage Consumers::findHighestPipelineStage() const {
  auto stages = getPipelineStages();

  if (stages.size() == 0) {
    return {};
  } else {
    return *std::max_element(stages.begin(), stages.end());
  }
}

OptionalVGraphId Consumers::findLowestVirtualGraphID() const {
  auto vgIDs = getVirtualGraphIds();

  if (vgIDs.size() == 0) {
    return {};
  } else {
    return *std::min_element(vgIDs.begin(), vgIDs.end());
  }
}

int Consumers::n(Op *op) const {
  auto found = consumers_m.find(op);
  if (found == consumers_m.end()) {
    return 0;
  } else {
    return found->second;
  }
}

bool Tensor::hasTensorData() const {
  if (data_.get() == nullptr) {
    return false;
  }
  return true;
}

TensorData *Tensor::tensorData() {
  if (data_.get() == nullptr) {
    throw error("Data not set for " + id);
  }
  return data_.get();
}

const TensorData *Tensor::tensorData() const {
  if (data_.get() == nullptr) {
    throw error("Data not set for " + id);
  }
  return data_.get();
}

void Consumers::append(std::stringstream &ss) {
  std::string tab = "     ";

  ss << '\n';
  ss << "Consumer count of Tensor " << tensorConsumed->id << " : " << '\n';
  int max_length = 0;
  for (auto &op_count : getMap()) {
    max_length =
        std::max(max_length, static_cast<int>(op_count.first->str().size()));
  }

  for (auto &op_count : getMap()) {
    ss << padded(op_count.first->str(), max_length + 1) << " : "
       << op_count.second << '\n';
  }
  ss << "Total number of consumptions: " << getTotal();
}

const std::map<Op *, int, POpCmp> &Consumers::getMap() const {
  return consumers_m;
}

void Consumers::extend(const std::map<Op *, int, POpCmp> &m) {
  for (auto &op_count : m) {
    auto found = consumers_m.find(op_count.first);
    if (found != consumers_m.end()) {
      found->second += op_count.second;
    } else {
      consumers_m[op_count.first] = op_count.second;
    }
  }
}

void Tensor::setProducer(Op *op) {
  if (hasProducer()) {
    throw error("Cannot set a producer for Tensor " + id + " as already one");
  }
  producer = op;
}

void Tensor::resetProducer(Op *op) {
  if (!hasProducer()) {
    throw error("Cannot reset a producer for Tensor " + id +
                " as it does not already have one");
  }
  producer = op;
}

int Consumers::getTotal() const {
  //  using X = decltype(consumers_m.begin());
  //  return std::accumulate(consumers_m.begin(), consumers_m.end(), 0,
  //      [](const X & v1, const X & v2){return v1.second + v2.second;});
  int total = 0;
  for (auto &op_count : consumers_m) {
    total += op_count.second;
  }
  return total;
}

// using 'this' in a constructor list? Be careful.
// https://stackoverflow.com/questions/5058349

Tensor::Tensor(TensorId n,
               TensorType t,
               Graph &g,
               const DebugContext &debugContext)
    : Tensor(n, t, VariableSettings(), g, debugContext) {}

Tensor::Tensor(TensorId n,
               VariableSettings vs,
               Graph &g,
               const DebugContext &debugContext)
    : Tensor(n, TensorType::Variable, vs, g, debugContext) {}

Tensor::Tensor(TensorId n,
               TensorType t,
               VariableSettings vs,
               Graph &g,
               const DebugContext &debugContext)
    : Vertex(), id(n), consumers(this), graph(g), producer(nullptr),
      tensorType_(t), data_(nullptr), di(debugContext, n, t),
      variableUpdateType(VariableUpdateType::Gradient), variableSettings(vs) {}

void Consumers::decrement(Op *op) {
  auto found = consumers_m.find(op);
  if (found == consumers_m.end()) {
    throw error("cannot decrement non-existant consumer, " + op->debugName());
  }
  --(found->second);
  if (found->second == 0) {
    consumers_m.erase(op);
  }
}

Op *Tensor::getProducer() const {
  if (!hasProducer()) {
    throw error("No producer for tensor " + id + " to return");
  }
  return getProducerUnsafe();
}

Op *Tensor::getProducerUnsafe() const { return producer; }

bool Tensor::hasProducer() const { return producer != nullptr; }

bool Tensor::isGraphInput() const {
  return std::find(graph.getInputIds().begin(),
                   graph.getInputIds().end(),
                   id) != graph.getInputIds().end();
}

InIndex Tensor::getGraphInputIndex() const {
  auto it =
      std::find(graph.getInputIds().begin(), graph.getInputIds().end(), id);
  return std::distance(graph.getInputIds().begin(), it);
}

bool Tensor::isGraphOutput() const {
  return std::find(graph.getOutputIds().begin(),
                   graph.getOutputIds().end(),
                   id) != graph.getOutputIds().end();
}

InIndex Tensor::getGraphOutputIndex() const {
  auto it =
      std::find(graph.getOutputIds().begin(), graph.getOutputIds().end(), id);
  return std::distance(graph.getOutputIds().begin(), it);
}

bool Tensor::isLoopInput() const {
  if (isGraphInput()) {
    auto ops = graph.getCallSiteOps();
    for (Op *op : ops) {
      if (op->isConvertibleTo<LoopOp>()) {
        return true;
      }
    }
  }
  return false;
}

bool Tensor::isImplicitLoopInput() const {
  if (isGraphInput()) {
    auto ops = graph.getCallSiteOps();
    for (Op *op : ops) {
      if (LoopOp *loop = dynamic_cast<LoopOp *>(op)) {
        if (getGraphInputIndex() >= loop->getNumExplicitInputs()) {
          return true;
        }
      }
    }
  }
  return false;
}

bool Tensor::isExplicitLoopInput() const {
  return isLoopInput() && !isImplicitLoopInput();
}

bool Tensor::isLoopTripCounter() const {
  if (isGraphInput()) {
    auto ops = graph.getCallSiteOps();
    for (Op *op : ops) {
      if (LoopOp *loop = dynamic_cast<LoopOp *>(op)) {
        auto sgInIdx = getGraphInputIndex();
        if (sgInIdx == 0 && loop->subgraphInToOpInIndex(sgInIdx) ==
                                LoopOp::getMaximumTripCountInIndex()) {
          return true;
        }
      }
    }
  }
  return false;
}

bool Tensor::isUnmodifiable() const {
  return
      // Checkpoint tensors must not be modified by recompute Ops to ensure
      // the same value is used on first and second runs of the recompute Op
      isCheckpointTensor() ||
      // A simple (but overly strict) way to ensure that an op is not inplaced
      // if:
      // - its input, or a tensor it aliases, is restored inplace
      // - and its output, or a tensor that is an alias of it, is consumed
      //   by an ipucopy
      // TODO T19283: Make less strict once we can determine if any two
      // tensors are aliases of eachother
      isRestoreInplaceTensor() ||
      // Implicit loop counter tensors must not be modified, because each loop
      // iteration needs access to the unmodified original input.
      isImplicitLoopInput() ||
      // Anchor tensors must not be modified to ensure the correct values are
      // returned. Here we conservatively assume anchors are returned at the
      // very end of the computation
      isAnchored() ||
      // Graph output tensors must not be modified to ensure the correct value
      // is returned at the end of the computation
      isGraphOutput() ||
      // Variables and constants must not be modified by inplacing operations
      // (variables can still be updated by designated update operations)
      tensorType() == TensorType::Variable ||
      tensorType() == TensorType::Const ||
      // Optimiser tensors must not change during the training loop as they are
      // only streamed to the IPU when the user calls get ir::setOptimizer
      isOptimizerTensor();
}

bool Tensor::isCheckpointTensor() const {
  auto cops = consumers.getOps();
  return std::any_of(cops.begin(),
                     cops.end(),
                     [](const Op *op) {
                       return op->settings.recomputeType ==
                              RecomputeType::Recompute;
                     }) &&
         (!hasProducer() ||
          getProducer()->settings.recomputeType == RecomputeType::Checkpoint);
}

bool Tensor::isImplicitRecomputeTensor() const {
  return (hasProducer() &&
          getProducer()->settings.recomputeType == RecomputeType::Recompute);
}

bool Tensor::isRestoreInplaceTensor() const {
  auto cops = consumers.getOps();
  return std::any_of(cops.begin(), cops.end(), [](const Op *op) {
    return op->isConvertibleTo<RestoreInplaceOp>();
  });
}

bool Tensor::idIncludesPrefix(const std::vector<std::string> &prefixes) const {
  using boost::algorithm::starts_with;
  return std::any_of(
      prefixes.begin(), prefixes.end(), [this](const std::string &prefix) {
        return starts_with(id, prefix);
      });
}

bool Tensor::isOptimizerTensor() const {
  // TODO T11262 is to make an optimizer Tensor class, so that we don't need to
  // do these string comparisons
  return idIncludesPrefix(reservedOptimizerPrefixes());
}

bool Tensor::isRemoteArgTensor() const {
  return idIncludesPrefix({reservedRemoteArgPrefix()});
}

bool Tensor::isRandomSeedTensor() const {
  return idIncludesPrefix({reservedRandomSeedPrefix()});
}

bool Tensor::isOptimizerStateTensor() const {
  if (idIncludesPrefix(reservedOptimizerStatePrefixes()) ||
      id == reservedLossScaleUpdateFactorId()) {
    // sanity check that the accl tensor is of Variable type
    if (tensorType() != TensorType::Variable) {
      throw error(
          "Tensor {} has been identified as an Optimizer tensor, but it is "
          "not a Variable tensor.",
          id);
    }
    return true;
  }
  return false;
}

bool Tensor::isAccumulatorTensor() const {
  if (idIncludesPrefix(reservedAccumulatorPrefixes())) {
    // sanity check that the accl tensor is of Variable type
    if (tensorType() != TensorType::Variable) {
      throw error(
          "Tensor {} has been identified as an Accumulator tensor, but it is "
          "not a Variable tensor.",
          id);
    }
    return true;
  }
  return false;
}

bool Tensor::isHostLoadTensor() const {
  if (!hasProducer()) {
    return false;
  }
  if (getProducer()->isConvertibleTo<HostLoadOp>()) {
    return true;
  }
  if (MultiExchangeOp *producer =
          dynamic_cast<MultiExchangeOp *>(getProducer())) {
    auto outIndex   = producer->output->indices(getIr().getTensor(id)).front();
    auto descriptor = producer->getExchangeDescriptor(
        producer->outIndexToDescriptorIndex(outIndex).first);
    if (descriptor.isHostExchange() &&
        descriptor.getDirection() == ExchangeDirection::Load) {
      return true;
    }
  }
  return false;
}

bool Tensor::isWeightTensor() const {
  if (tensorType() != TensorType::Variable) {
    return false;
  }
  if (isAccumulatorTensor() || isOptimizerStateTensor()) {
    return false;
  }
  return true;
}

bool Tensor::isAnchored() const { return graph.getIr().isAnchored(id); }

bool Tensor::isRootAnchor() const { return graph.getIr().isRootAnchor(id); }

bool Tensor::anyAlias(std::function<bool(Tensor *)> predicate) const {

  constexpr const char *const ctxt{"Tensor::anyAlias"};
  logging::ir::trace("{} for Tensor {},", ctxt, str());

  auto scopedStopwatch = getIr().timePartitionLogger().scopedStopwatch(ctxt);

  // First check if this tensor itself satisfies the predicate. If so, we need
  // not bother constructing a poprithms graph to check for alias tensors.
  Tensor *t = graph.getTensors().get(id);
  if (predicate(t)) {
    return true;
  }

  // Build a poprithms::memory::inplace::Graph from the popart::Graph.
  AliasModel popMem;
  AliasModelGrower aliasModelGrower{popMem};
  aliasModelGrower.growPartialGraph(graph, id, DataDependenciesOnly::Yes);

  // Get the identifier used to represent this tensor in poprithms.
  auto poprithmsTensorId = popMem.getPoprithmsTensorId(id);

  // Iterate over all aliases as found by poprithms.
  for (const auto &poprithmsAliasId : popMem.g.allAliases(poprithmsTensorId)) {
    // All PopART tensors map to a Poprithms tensor, but not all Poprithms
    // tensors map to a PopART one. It is safe to only look at those Poprithms
    // aliases that have a corresponding PopART tensor.
    if (popMem.contains(poprithmsAliasId)) {
      // Translate back to PopART IDs.
      auto popartAliasId = popMem.getTensorId(poprithmsAliasId);
      auto popartAlias   = graph.getTensors().get(popartAliasId);
      if (predicate(popartAlias)) {
        // May as well return now.
        return true;
      }
    }
  }

  return false;
}

void Consumers::increment(Op *op) {
  auto found = consumers_m.find(op);
  if (found == consumers_m.end()) {
    consumers_m[op] = 1;
  } else {
    ++(found->second);
  }
}

std::vector<int64_t> Tensor::returnedShape(unsigned replicationFactor) {
  int64_t returned =
      variableSettings.numReplicasReturningVariable(replicationFactor);

  if (returned == 1) {
    return Shape(info.shape());
  }

  std::vector<int64_t> tensor_shape = info.shape();
  std::vector<int64_t> return_shape(tensor_shape.size() + 1);

  // Read in elements
  return_shape[0] = returned;
  for (auto i = 0; i < tensor_shape.size(); i++) {
    return_shape[i + 1] = tensor_shape[i];
  }

  return return_shape;
}

void Tensor::verifyMutableVoidInfo(const TensorInfo mutableVoidInfo,
                                   unsigned replicationFactor) {

  auto returShape = returnedShape(replicationFactor);

  bool correct = returShape.size() == mutableVoidInfo.shape().size();
  for (size_t i = 0; i < returShape.size(); i++) {
    correct = correct && (returShape[i] == mutableVoidInfo.shape()[i]);
  }
  if (!correct) {
    throw internal_error(
        "The shape of the MutableVoidData for IO does not not match the "
        "expected write region of this tensor. "
        "Tensor Write ({}): {} != MutableVoidData ({}): {}",
        info.shape().size(),
        returShape,
        mutableVoidInfo.shape().size(),
        mutableVoidInfo.shape());
  }
}

std::vector<Op *> Consumers::getOps() const {
  std::vector<Op *> ops;
  ops.reserve(consumers_m.size());
  for (auto &x : consumers_m) {
    ops.push_back(x.first);
  }
  return ops;
}

const std::map<Tensor *, std::vector<int>, PTensorCmp> &
TensorIndexMap::indicesMap() const {
  return indices_map;
}

TensorType Tensor::tensorType() const { return tensorType_; }

void Tensor::setTensorType(TensorType t) { tensorType_ = t; }

std::string Tensor::tensor_type() const {
  std::stringstream ss;
  ss << tensorType_;
  return ss.str();
}

std::vector<Op *> Tensor::associatedOps() const {
  std::vector<Op *> result = consumers.getOps();

  if (hasProducer()) {
    result.push_back(getProducer());
  }

  return result;
}

} // namespace popart
