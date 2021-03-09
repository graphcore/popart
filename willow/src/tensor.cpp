// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstring>
#include <string>
#include <popart/ces/constexpr.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/onnxutil.hpp>
#include <popart/op.hpp>
#include <popart/op/init.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/loop.hpp>
#include <popart/op/restore.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensornames.hpp>
#include <popart/util.hpp>

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

bool Tensor::isModified() const {
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
  if (isExplicitLoopInput()) {
    return true;
  }
  return false;
}

VGraphId Tensor::getVirtualGraphIdUnsafe() const {
  return getVirtualGraphIdAndTileSetUnsafe().first;
}

VGraphIdAndTileSet
Tensor::getVirtualGraphIdAndTileSetUnsafe(std::set<OpId> visited) const {

  // If this Tensor has a Producer, use its VirtualGraphId if it has one
  if (hasProducer()) {
    // special case of IPUCopy producer
    auto ipucopy = dynamic_cast<IpuCopyOp *>(getProducer());
    if (ipucopy) {
      return {ipucopy->getDestIpu(), ipucopy->settings.tileSet};
    } else if (getProducer()->hasVirtualGraphId()) {
      if (visited.find(getProducer()->id) == visited.end()) {
        for (auto &indices : getProducer()->output->indicesMap()) {
          if (indices.first == this) {
            return getProducer()->getIntrospectionOutVirtualGraphId(
                indices.second[0]);
          }
        }
      }
    }
  }

  // No producer with an id. Try to get the virtual graph id from a consumer.
  // Use the id of the first consumer with an id, if there is one
  for (Op *consumer : consumers.getOps()) {
    if (visited.find(consumer->id) == visited.end()) {
      if (consumer->hasVirtualGraphId()) {
        for (auto &indices : consumer->input->indicesMap()) {
          if (indices.first == this) {
            return consumer->getIntrospectionInVirtualGraphId(
                indices.second[0]);
          }
        }
      }
    }
  }

  // no consumers have virtual graph ids. If a consumer
  // is an IPUCopy, we can derive the virtual graph id from it.
  for (Op *consumer : consumers.getOps()) {
    auto ipucopy = dynamic_cast<IpuCopyOp *>(consumer);
    if (ipucopy) {
      return {ipucopy->getSourceIpus().at(id), ipucopy->settings.tileSet};
    }
  }

  // Graph input, derive from call site
  if (isGraphInput()) {
    auto callSites = getGraph().getCallSiteOps();
    if (callSites.size()) {
      Op *callSite = callSites.front();
      return {callSite->hasVirtualGraphId() ? callSite->getVirtualGraphId()
                                            : unusedVGraphId,
              callSite->settings.tileSet};
    }
  }

  // No virtual graph Id determined
  return {unusedVGraphId, TileSet::Compute};
}

VGraphIdAndTileSet
Tensor::getVirtualGraphIdAndTileSet(std::set<OpId> visited) const {
  auto vid = getVirtualGraphIdAndTileSetUnsafe(visited);
  if (vid == VGraphIdAndTileSet(unusedVGraphId, TileSet::Compute)) {
    throw error("Invalid call to getVirtualGraphId, Tensor does not have one");
  }
  return vid;
}

VGraphId Tensor::getVirtualGraphId() const {
  auto vid = getVirtualGraphIdAndTileSetUnsafe();
  if (vid == VGraphIdAndTileSet(unusedVGraphId, TileSet::Compute)) {
    throw error("Invalid call to getVirtualGraphId, Tensor does not have one");
  }
  return vid.first;
}

bool Tensor::hasVirtualGraphId() const {
  return getVirtualGraphIdUnsafe() != unusedVGraphId;
}

std::vector<char> Tensor::getDataViaRecursion() const {
  if (hasProducer()) {
    if (ConstExprOpManager::hasConstExprOp(producer)) {
      for (auto inTensor : producer->input->tensors()) {
        if (!inTensor->hasTensorData()) {
          auto outTemp = inTensor->getDataViaRecursion();
          inTensor->setTensorData(inTensor->info, outTemp.data());
        }
      }
      auto ceOp = ConstExprOpManager::createConstExprOp(producer);
      return ceOp->compute();
    } else {
      throw error("Recursing up the tree of producers for {}, the op {} was "
                  "found which has no const expr version.",
                  id,
                  producer->opid);
    }
  } else {
    throw error("Tensor {} has no producer, so can't work back to find data.",
                id);
  }
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
  std::unique_ptr<Tensor> theClone(
      new Tensor("clone_" + id, tensorType(), graph_, getDebugInfo()));
  theClone->info = info;
  return theClone;
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
    : Vertex(), id(n), consumers(this), graph(g), producer(nullptr),
      tensorTypeInfo(&getTensorTypeInfoMap().at(t)), data_(nullptr),
      di(debugContext, n, t) {}

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
        if (getGraphInputIndex() >= loop->numExplicitInputs()) {
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

bool Tensor::isOptimizerTensor() const {

  // TODO T11262 is to make an optimizer Tensor class, so that we don't need to
  // do these string comparisons
  for (auto optPref : reservedOptimizerPrefixes()) {
    std::size_t found = id.find(optPref);
    if (found != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool Tensor::isRemoteArgTensor() const {
  std::size_t found = id.find(reservedRemoteArgPrefix());
  if (found != std::string::npos) {
    return true;
  }
  return false;
}

bool Tensor::isRandomSeedTensor() const {
  std::size_t found = id.find(reservedRandomSeedPrefix());
  if (found != std::string::npos) {
    return true;
  }
  return false;
}

bool Tensor::isOptimizerStateTensor() const {
  auto states = reservedOptimizerStatePrefixes();
  if (std::any_of(
          states.begin(), states.end(), [this](const std::string state) {
            return id.find(state) != std::string::npos;
          })) {
    // sanity check that the accl tensor is of Variable type
    if (tensorType() != TensorType::Variable) {
      throw error("Tensor {} has been identified as an Accl tensor, but it is "
                  "not a Variable tensor.",
                  id);
    }
    return true;
  }
  return false;
}

bool Tensor::isAccumulatorTensor() const {
  auto states = reservedAccumulatorPrefixes();
  if (std::any_of(
          states.begin(), states.end(), [this](const std::string state) {
            return id.find(state) != std::string::npos;
          })) {
    // sanity check that the accl tensor is of Variable type
    if (tensorType() != TensorType::Variable) {
      throw error("Tensor {} has been identified as an Accl tensor, but it is "
                  "not a Variable tensor.",
                  id);
    }
    return true;
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

bool Tensor::isAnchored() const {
  auto &anchors = graph.getIr().getDataFlow().anchors();
  return std::find(anchors.begin(), anchors.end(), id) != anchors.end();
}

bool Tensor::anyAlias(std::function<bool(Tensor *)> predicate) const {
  // Fetch non-const pointer to "this"
  Tensor *t             = graph.getTensors().get(id);
  auto aliasedTensorMap = graph.getTensors().aliasChainsFrom(t);
  auto fullRegion       = view::Region::getFull(t->info.shape());

  bool tChecked = false;
  bool any      = false;

  for (auto &chain : aliasedTensorMap) {
    auto regions = chain.second.apply(fullRegion);
    if (nonEmptyRegion(regions)) {
      // We check if t itself is in the chain, if not we need to check it later.
      if (chain.first == t) {
        tChecked = true;
      }
      any |= predicate(chain.first);
    }
  }

  if (!tChecked) {
    any |= predicate(t);
  }
  return any;
}

void Consumers::increment(Op *op) {
  auto found = consumers_m.find(op);
  if (found == consumers_m.end()) {
    consumers_m[op] = 1;
  } else {
    ++(found->second);
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

const std::map<Tensor *, std::vector<int>, TensorIndexMap::TensorPtrComparator>
    &TensorIndexMap::indicesMap() const {
  return indices_map;
}

const std::map<TensorType, TensorTypeInfo> &getTensorTypeInfoMap() {
  static std::map<TensorType, TensorTypeInfo> M = initTensorTypeInfoMap();
  return M;
}

TensorType Tensor::tensorType() const { return tensorTypeInfo->type(); }

const std::string &Tensor::tensor_type() const {
  return tensorTypeInfo->type_s();
}

void Tensor::setTensorType(TensorType t) {
  tensorTypeInfo = &getTensorTypeInfoMap().at(t);
}

std::vector<Op *> Tensor::associatedOps() const {
  std::vector<Op *> result = consumers.getOps();

  if (hasProducer()) {
    result.push_back(getProducer());
  }

  return result;
}

TensorType TensorTypeInfo::type() const { return tensorType_; }

const std::string &TensorTypeInfo::type_s() const { return tensor_type_; }

TensorTypeInfo::TensorTypeInfo(TensorType t_, std::string ts_)
    : tensorType_(t_), tensor_type_(ts_) {}

std::map<TensorType, TensorTypeInfo> initTensorTypeInfoMap() {
  std::map<TensorType, TensorTypeInfo> tensor_types_m = {
      {TensorType::ActGrad, {TensorType::ActGrad, "ActGrad"}},
      {TensorType::Const, {TensorType::Const, "Const"}},
      {TensorType::Stream, {TensorType::Stream, "Stream"}},
      {TensorType::Unknown, {TensorType::Unknown, "Unknown"}},
      {TensorType::Variable, {TensorType::Variable, "Variable"}}};
  if (tensor_types_m.size() != static_cast<int64_t>(TensorType::N)) {
    throw error("missing element in TensorTypes");
  }
  return tensor_types_m;
}

VariableTensor::VariableTensor(TensorId n,
                               Graph &g,
                               const DebugContext &debugContext)
    : Tensor(n, TensorType::Variable, g, debugContext),
      variableUpdateType(VariableUpdateType::Gradient) {}

std::unique_ptr<Tensor> VariableTensor::clone(Graph &graph_) const {
  std::unique_ptr<Tensor> theClone(
      new VariableTensor("clone_" + id, graph_, getDebugInfo()));
  theClone->info = info;
  return theClone;
}

} // namespace popart
