// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <boost/optional/optional_io.hpp>
#include <onnx/onnx_pb.h>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/util.hpp>

// The layers:
#include <popart/op/elementwise.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/restore.hpp>
#include <popart/op/varupdate.hpp>

namespace {
using namespace popart;

// Shared implementation for Op::fwdRegMap and Op::bwdRegMap methods
view::RegMap defaultRegMapImpl(const Op &op,
                               InIndex i,
                               OutIndex o,
                               const std::string &methodName) {
  logging::op::trace("[{}] for OP {} index {}", methodName, op.debugName(), i);
  if (!op.input->hasIndex(i) || !op.output->hasIndex(o)) {
    throw error("invalid index in {}", methodName);
  } else if (!op.output->hasIndex(o)) {
    throw error("{} called for op with no zero output", methodName);
  } else if (op.inShape(i) != op.outShape(0)) {
    throw error("default {} not valid : should be specialised for {}",
                methodName,
                op.str());
  }
  return [](const view::Region &r) { return view::Regions(1, r); };
}

} // namespace

namespace popart {

GradInOutMapper::GradInOutMapper(int iG, int iNG, GradOpInType t)
    : iGrad(iG), iNonGrad(iNG), type(t) {}

bool GradInOutMapper::operator==(const GradInOutMapper &rhs) const {
  return (type == rhs.type) && (iGrad == rhs.iGrad) &&
         (iNonGrad == rhs.iNonGrad);
}

TensorInfo &Op::outInfo(OutIndex index) { return output->tensor(index)->info; }

const TensorInfo &Op::inInfo(InIndex index) const {
  return input->tensor(index)->info;
}

TensorInfo &Op::inInfo(InIndex index) { return input->tensor(index)->info; }

const TensorInfo &Op::outInfo(OutIndex index) const {
  return output->tensor(index)->info;
}

bool Op::isExcludedFromPattern(const Pattern *p) const {
  return settings.excludePatterns.find(p->getPatternName()) !=
         settings.excludePatterns.end();
}

Ir &Op::getIr() { return getGraph().getIr(); }
const Ir &Op::getIr() const { return getGraph().getIr(); }

bool Op::isElementWiseUnary() const {
  return isConvertibleTo<ElementWiseUnaryOp>();
}

view::Regions Op::uses(InIndex index) const {
  return view::Regions(1, view::Region::getFull(inShape(index)));
}

view::Regions Op::modifies(InIndex index) const {
  return view::Regions(1, view::Region::getEmpty(inRank(index)));
}

view::Regions Op::aliases(InIndex in, OutIndex) const {
  return view::Regions(1, view::Region::getEmpty(inRank(in)));
}

view::RegMap Op::fwdRegMap(InIndex i, OutIndex o) const {
  return defaultRegMapImpl(*this, i, o, "fwdRegMap");
}

view::RegMap Op::bwdRegMap(InIndex i, OutIndex o) const {
  return defaultRegMapImpl(*this, i, o, "bwdRegMap");
}

bool Op::isLossOp() const { return false; }
bool Op::isIpuCopyOp() const { return false; }
bool Op::copiesOptimizerTensors() const { return false; }
bool Op::isOptimizerOp() const { return false; }

bool Op::requiresRandomSeed() const { return false; }
InIndex Op::getSeedInIndex() const {
  throw error("Op {} does not have random seed input tensor", str());
}

Op::~Op() = default;

// return a vector of 1 or several OpAndTensorIds for
// obtaining the gradient of the inputs of this Op.
// The Op in the OpAndTensorIds is the gradient op, and
// the TensorIds are the input indices of input of this
// Op for which the gradient is computed
std::vector<std::unique_ptr<Op>> Op::getGradOps() { return {}; }

void Op::setup() { throw error("No setup() for {}", opid); }

void Op::defaultConnectInTensor(InIndex inIndex, TensorId tenId) {
  if (input->hasIndex(inIndex)) {
    throw internal_error(
        "error connecting input tensor '{}', {} already has an "
        "input at index {}",
        tenId,
        debugName(),
        inIndex);
  }

  Tensor *ptensor = getGraph().getTensors().get(tenId);
  input->insert(inIndex, ptensor);
  ptensor->consumers.increment(this);

  // Inherit fromLoss from the input tensor
  if (ptensor->fromLoss == PathFromLoss::Yes) {
    fromLoss = PathFromLoss::Yes;
  }
}

void Op::connectInTensor(InIndex inIndex, TensorId tenId) {
  defaultConnectInTensor(inIndex, tenId);
}

void Op::connectOutTensor(OutIndex outIndex, TensorId tenId) {
  if (output->hasIndex(outIndex)) {
    throw internal_error(
        "error connecting output tensor '{}', {} already has an "
        "output at index {}",
        tenId,
        debugName(),
        outIndex);
  }

  Tensor *ptensor = getGraph().getTensors().get(tenId);

  if (ptensor->hasProducer()) {
    throw internal_error("error connecting output tensor '{}' to {}, tensor "
                         "already has a producer",
                         tenId,
                         debugName());
  }

  output->insert(outIndex, ptensor);
  ptensor->setProducer(this);

  // Output tensor takes fromLoss from op
  ptensor->fromLoss = fromLoss;
}

void Op::disconnectInTensor(Tensor *tensor) {
  for (auto i : input->indicesMap().at(tensor)) {
    disconnectInTensor(i, tensor);
  }
}

void Op::disconnectInTensor(InIndex inIndex, Tensor *tensor) {
  if (inTensor(inIndex) != tensor) {
    throw internal_error("error disconnecting input tensor '{}', tensor is not "
                         "input {} of {}",
                         tensor->id,
                         inIndex,
                         debugName());
  }

  tensor->consumers.decrement(this);

  input->erase(inIndex);
}

void Op::disconnectOutTensor(Tensor *tensor) {
  for (auto idx : output->indices(tensor)) {
    if (tensor->hasProducer() && tensor->getProducer() == this) {
      tensor->resetProducer(nullptr);
    } else {
      throw internal_error(
          "error disconnecting output, tensor is not produced by this op");
    }

    output->erase(idx);
  }
}

void Op::disconnectAllInputs() {
  auto inputs = input->tensors();
  for (auto i : inputs) {
    disconnectInTensor(i);
  }
  if (input->n() != 0) {
    throw internal_error("Failed to disconnect all inputs from {}",
                         debugName());
  }
}

void Op::disconnectAllOutputs() {
  auto tensors = output->tensors();
  for (auto tensor : tensors) {
    disconnectOutTensor(tensor);
  }
  if (output->n() != 0) {
    throw internal_error("Failed to disconnect all outputs from {}",
                         debugName());
  }
}

void Op::createAndConnectOutTensor(OutIndex outIndex, TensorId tenId) {
  if (output->hasIndex(outIndex)) {
    throw internal_error(
        "error connecting output tensor '{}', {} already has an "
        "output at index {}",
        tenId,
        debugName(),
        outIndex);
  }

  tenId = (getScope() / tenId).str();

  getGraph().getTensors().addActGrad(tenId);
  Tensor *ptensor = getGraph().getTensors().get(tenId);
  output->insert(outIndex, ptensor);
  ptensor->setProducer(this);
}

std::string Op::getSubgraphEquivId() const {

  // Are any of the inputs aliased by output?
  bool noAliases = true;
  for (auto &in_tensor : input->tensorMap()) {
    for (auto &out_tensor : output->tensorMap()) {
      auto regions = aliases(in_tensor.first, out_tensor.first);
      noAliases    = noAliases && std::all_of(regions.begin(),
                                           regions.end(),
                                           [](const view::Region &r) {
                                             return r.isEmpty();
                                           });
    }
  }

  // Of all aliasing Ops, we only allow the VarUpdateOp to be outlined.
  // This partially resolves the failure to the propagate inplace modifications
  // through calls, T8604.

  /*
  bool aliasAndNotVarUpdate =
      !noAliases && !(dynamic_cast<const VarUpdateOp *>(this));
  */

  std::stringstream ss;
  if (isOutlineable()) { // && !aliasAndNotVarUpdate) {
    OpEquivIdCreator os(this);
    // TODO: Figure out which attributes really are relevant to outlining!
    // Certainly, not all are, and this makes subgraph outlining ineffective.
    appendOutlineAttributes(os);
    ss << os.str();
  } else {
    // in the case where the op is not outlineable, we return a unique string
    // to guarantee that it does not appear in any outline matches.
    ss << str() << "_uid_" << id;
  }

  logging::trace(
      "[Op::getSubgraphEquivId] Op: {} Id: {}", debugName(), ss.str());

  return ss.str();
}

void Op::append(std::stringstream &ss) const {
  OpSerialiser os(this, ss);

  appendAttributes(os);
  appendMore(os);
}

void Op::toJSON(std::stringstream &ss) const { OpJsonSerialiser os(this, ss); }

// The appendMore attributes appear in the log but are not used
// in the outlining algorithm
void Op::appendMore(OpSerialiserBase &os) const {
  os.appendAttribute("schedulePriority",
                     static_cast<float>(settings.schedulePriority));
}

int Op::getNonGradInIndex(int gradOpOutIndex) const {
  return gradOutToNonGradIn().at(gradOpOutIndex);
}

const std::vector<GradInOutMapper> &Op::gradInputInfo() const {
  throw error("Op {} cannot get `grad input info'", opid);
}

const std::map<int, int> &Op::gradOutToNonGradIn() const {
  throw error("Op {} cannot get `grad out to non grad in'", opid);
}

std::vector<std::tuple<OperatorIdentifier, float>>
Op::inplacePriorityDefault() const {
  return {};
}

std::unique_ptr<Op>
Op::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  throw error("Op {} cannot return inplace variant {} ", opid, operator_id);
}

bool Op::readyToCreateGradients(std::set<int> &s) const {
  return s.size() == nEdgesToLoss;
}

int64_t Op::memOfOutputs() const {
  int64_t mem = 0;
  for (auto &t_inds : output->indicesMap()) {
    mem += t_inds.first->info.nbytes();
  }
  return mem;
}

void Op::appendAttributes(OpSerialiserBase &os) const {
  appendOutlineAttributes(os);
  os.appendAttribute(sPingPongPhaseAttribute, settings.pingPongPhase);
  os.appendAttribute(sPipelineStageAttribute, settings.pipelineStage);
  os.appendAttribute("scope", getScope());
}

void Op::appendOutlineAttributes(OpSerialiserBase &os) const {
  std::string recomputeString =
      settings.recomputeType == RecomputeType::Recompute ? "YES" : "NO";
  os.appendAttribute("recompute", recomputeString);
  os.appendAttribute(sVirtualGraphAttribute, getOptionalVGraphId());
  os.appendAttribute("useIoTiles", settings.useIoTiles);
  for (auto attribute : settings.extraOutlineAttributes) {
    os.appendAttribute(attribute.first,
                       attribute.first + ":" + attribute.second);
  }
}

std::vector<const Graph *> Op::getCalledGraphs() const { return {}; }

std::vector<TensorId> Op::getInputsForGraph(const Graph &) const {
  throw error("Op does not call any graphs");
}

Shape Op::prettyNpOut(const Shape &s0, const Shape &s1) const {
  std::stringstream ss;
  ss << "Op " << str();

  return npOut(s0, s1, ss.str());
}

TensorInfo Op::prettyNpOut(const TensorInfo &i0, const TensorInfo &i1) const {
  std::stringstream ss;
  ss << "Op " << str();

  return npOut(i0, i1, ss.str());
}

const std::string &Op::name() const { return getName(); }

Op::Op(const Op &op)
    : Vertex(op), input(new TensorIndexMap), output(new TensorIndexMap),
      id(op.settings.graph.get().getIr().getAndIncrOpsCounter()), opid(op.opid),
      settings(op.settings) {
  // input, output: empty.
}

bool Op::hasInput(InIndex index) const { return input->hasIndex(index); }

Tensor *Op::inTensor(InIndex index) { return input->tensor(index); }
const Tensor *Op::inTensor(InIndex index) const { return input->tensor(index); }
Tensor *Op::outTensor(OutIndex index) { return output->tensor(index); }
const Tensor *Op::outTensor(OutIndex index) const {
  return output->tensor(index);
}

size_t Op::inTensorCount() const { return input->n(); }
size_t Op::outTensorCount() const { return output->n(); }

TensorId Op::inId(InIndex index) { return inTensor(index)->id; }
const TensorId Op::inId(InIndex index) const { return inTensor(index)->id; }
TensorId Op::outId(OutIndex index) { return outTensor(index)->id; }
const TensorId Op::outId(OutIndex index) const { return outTensor(index)->id; }

Op::Op(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : input(new TensorIndexMap), output(new TensorIndexMap),
      // the id
      id(settings_.graph.get().getIr().getAndIncrOpsCounter()), opid(_opid),
      // the Attributes
      settings(settings_) {}

Ir &Op::Op::Settings::getIr() const { return graph.get().getIr(); }

std::ostream &operator<<(std::ostream &ost, const RecomputeType &rt) {
  switch (rt) {
  case (RecomputeType::Recomputed): {
    ost << "Recomputed";
    break;
  }
  case (RecomputeType::Recompute): {
    ost << "Recompute";
    break;
  }
  case (RecomputeType::Checkpoint): {
    ost << "Checkpoint";
    break;
  }
  case (RecomputeType::Undefined): {
    ost << "Undefined";
    break;
  }
  }
  return ost;
}

void Op::Op::Settings::setFromAttributes(const Attributes &attributes) {

  if (attributes.hasAttribute(sPingPongPhaseAttribute)) {
    int64_t value;
    attributes.set(value, sPingPongPhaseAttribute);
    pingPongPhase = value;
  }

  if (attributes.hasAttribute(sVirtualGraphAttribute)) {
    int64_t value;
    attributes.set(value, sVirtualGraphAttribute);
    vgraphId = value;
  }

  if (attributes.hasAttribute(sPipelineStageAttribute)) {
    int64_t value;
    attributes.set(value, sPipelineStageAttribute);
    pipelineStage = value;
  }

  if (attributes.hasAttribute(sRecomputeOutputAttribute)) {
    int64_t recomputeTypeTmp;

    attributes.set(recomputeTypeTmp, sRecomputeOutputAttribute);
    // reversing the static_cast<int64_t> used to insert value into ONNX proto
    recomputeType = static_cast<RecomputeType>(recomputeTypeTmp);
  }

  if (attributes.hasAttribute(sCacheOutputAttribute)) {
    int64_t cacheTypeTmp;
    attributes.set(cacheTypeTmp, sCacheOutputAttribute);
    cacheType = static_cast<CacheType>(cacheTypeTmp);
  }

  if (attributes.hasAttribute(sSchedulePriority)) {
    float schedule_priority;
    attributes.set(schedule_priority, sSchedulePriority);
    schedulePriority = static_cast<double>(schedule_priority);
  }

  if (attributes.hasAttribute(sIOTilesAttribute)) {
    int64_t useIoTilesTmp;
    attributes.set(useIoTilesTmp, sIOTilesAttribute);
    useIoTiles = static_cast<IsIoTile>(useIoTilesTmp);
  }

  bool hasNamesAtt = attributes.hasAttribute(sInplaceOpNames);
  // either both or neither inplace attributes must be provided
  if (hasNamesAtt != attributes.hasAttribute(sInplaceOpPriorities)) {
    throw error("Either BOTH or NEITHER of the fields {} and {} must be set, "
                "but only {} is set",
                sInplaceOpNames,
                sInplaceOpPriorities,
                hasNamesAtt ? sInplaceOpNames : sInplaceOpPriorities);
  }

  // if both are provided,
  if (hasNamesAtt) {
    std::vector<std::string> names;
    attributes.set(names, sInplaceOpNames);
    std::vector<float> priorities;
    attributes.set(priorities, sInplaceOpPriorities);

    if (names.size() != priorities.size()) {
      throw error("For fields {} and {}, the number of elements must be the "
                  "same [ {} != {} ]",
                  sInplaceOpPriorities,
                  sInplaceOpNames,
                  priorities.size(),
                  names.size());
    }

    for (int i = 0; i < names.size(); ++i) {
      inplacePriorityVeto.push_back({names[i], priorities[i]});
    }
  }

  if (attributes.hasAttribute(sExcludePatternsAttribute)) {
    std::vector<std::string> names;
    attributes.set(names, sExcludePatternsAttribute);
    excludePatterns.insert(names.begin(), names.end());

    // Check the names in excludePatterns are valid.
    for (const auto &patternName : excludePatterns) {
      if (!PatternNames::contains(patternName)) {
        throw error("Invalid pattern name '{}' in Op::excludePatterns",
                    patternName);
      }
    }
  }

  if (attributes.hasAttribute(sOutlineAttribute)) {
    std::vector<std::string> outlineAttributes;
    attributes.set(outlineAttributes, sOutlineAttribute);
    for (size_t i = 0; i < outlineAttributes.size(); i += 2) {
      extraOutlineAttributes.insert(
          {outlineAttributes[i], outlineAttributes[i + 1]});
    }
  }
}

void Op::setVirtualGraphId(const OptionalVGraphId value) {
  settings.vgraphId = value;
}

const OptionalVGraphId Op::getOptionalVGraphId() const {
  return settings.vgraphId;
}

VGraphId Op::getVirtualGraphId() const {
  if (!hasVirtualGraphId()) {
    throw error(
        "Cannot return vGraphId for Op {}. It has not had this attribute set",
        debugName());
  }
  return *(settings.vgraphId);
}

VGraphIdAndIoTile Op::getIntrospectionInVirtualGraphId(InIndex) const {
  return {getVirtualGraphId(), settings.useIoTiles};
}

VGraphIdAndIoTile Op::getIntrospectionOutVirtualGraphId(OutIndex) const {
  return {getVirtualGraphId(), settings.useIoTiles};
}

bool Op::hasVirtualGraphId() const {
  if (settings.vgraphId) {
    return true;
  } else {
    return false;
  }
}

const OptionalPingPongPhase Op::getOptionalPingPongPhase() const {
  return settings.pingPongPhase;
}

void Op::setPingPongPhase(const OptionalPingPongPhase value) {
  settings.pingPongPhase = value;
}

PingPongPhase Op::getPingPongPhase() const {
  if (!hasPingPongPhase()) {
    throw error("Cannot return PingPongPhase for Op {}. "
                "It has not had this attribute set",
                debugName());
  }
  return *(settings.pingPongPhase);
}

bool Op::hasPingPongPhase() const {
  if (settings.pingPongPhase) {
    return true;
  } else {
    return false;
  }
}

const OptionalBatchSerializedPhase Op::getOptionalBatchSerializedPhase() const {
  return settings.batchSerializedPhase;
}

void Op::setBatchSerializedPhase(const OptionalBatchSerializedPhase value) {
  settings.batchSerializedPhase = value;
}

BatchSerializedPhase Op::getBatchSerializedPhase() const {
  if (!hasBatchSerializedPhase()) {
    throw error("Cannot return BatchSerializedPhase for Op {}. "
                "It has not had this attribute set",
                debugName());
  }
  return *(settings.batchSerializedPhase);
}

bool Op::hasBatchSerializedPhase() const {
  if (settings.batchSerializedPhase) {
    return true;
  } else {
    return false;
  }
}

OptionalPipelineStage Op::getOptionalPipelineStage() const {
  return settings.pipelineStage;
}

void Op::setPipelineStage(OptionalPipelineStage value) {
  settings.pipelineStage = value;
}

bool Op::hasPipelineStage() const { return bool(settings.pipelineStage); }

PipelineStage Op::getPipelineStage() const {
  if (!hasPipelineStage()) {
    throw error("Cannot return pipelineStage for Op {}. It has not had this "
                "attribute set.",
                debugName());
  }
  return *(settings.pipelineStage);
}

void Op::inheritPlacementAttributes(bool inheritSerializations) {
  const Ir &ir = getGraph().getIr();

  enum ConnectedOpRelation {
    // The op from which to inherit is a producer of an input to this op
    Producer = 0,
    // The op from which to inherit is a consumer of an output to this op
    Consumer
  };

  auto getOpVGID = [](Op *op, ConnectedOpRelation rel) {
    OptionalVGraphId vgid;
    if (op->isIpuCopyOp()) {
      IpuCopyOp *copyOp = dynamic_cast<IpuCopyOp *>(op);
      // If the lhsOp is a producer to the current op, the DestIpu is relevant
      // otherwise, the source IPU is relevant
      vgid = rel == ConnectedOpRelation::Producer ? copyOp->getDestIpu()
                                                  : copyOp->getSourceIpu();
    } else {
      vgid = op->getOptionalVGraphId();
    }
    return vgid;
  };

  auto getOpPingPongPhase = [](Op *op, ConnectedOpRelation rel) {
    OptionalPingPongPhase phase;
    if (op->isIpuCopyOp()) {
      IpuCopyOp *copyOp = dynamic_cast<IpuCopyOp *>(op);
      if (copyOp->getSourceIpu() % 2 != copyOp->getDestIpu() % 2 &&
          rel == ConnectedOpRelation::Producer && op->hasPingPongPhase()) {
        // Inter-phase copy: Destination phase
        phase = op->getPingPongPhase() + 1;
        return phase;
      }
    }
    phase = op->getOptionalPingPongPhase();
    return phase;
  };

  OptionalVGraphId requiredVgid;
  std::vector<std::pair<Op *, ConnectedOpRelation>> connectedOps;

  // Scan if the current operation modifies any weights. This modification
  // may occur through an inplace aliasing of the weight, e.g. with MatMul
  // serialization, therefore we have to search through the aliasing chains
  // to find any directly (or through an alias) modified weights, and ensure
  // the modifying op is placed on the virtual graph that owns the weight.
  for (auto inIndexAndTensor : input->tensorMap()) {

    std::set<Tensor *, PTensorCmp> associatedVariableTensors;

    if (inIndexAndTensor.second->getTensorTypeInfo()->type() ==
        TensorType::Variable) {
      associatedVariableTensors.insert(inIndexAndTensor.second);
    }

    auto aliasedTensorMap =
        getGraph().getTensors().aliasChainsFrom(inIndexAndTensor.second);
    auto fullRegion =
        view::Region::getFull(inIndexAndTensor.second->info.shape());
    for (const auto &chain : aliasedTensorMap) {
      auto regions = chain.second.apply(fullRegion);
      bool nonEmptyAlias =
          std::any_of(regions.begin(), regions.end(), [](view::Region &r) {
            return !r.isEmpty();
          });
      if (nonEmptyAlias &&
          chain.first->getTensorTypeInfo()->type() == TensorType::Variable) {
        associatedVariableTensors.insert(chain.first);
        associatedVariableTensors.insert(inIndexAndTensor.second);
      }
    }

    for (Tensor *varTensor : associatedVariableTensors) {
      auto modifiedRegions = modifies(inIndexAndTensor.first);

      bool variableModOrAlias =
          std::any_of(modifiedRegions.begin(),
                      modifiedRegions.end(),
                      [](view::Region &r) { return !r.isEmpty(); });

      for (auto outIndexAndTensor : output->tensorMap()) {
        auto aliasedRegions =
            aliases(inIndexAndTensor.first, outIndexAndTensor.first);

        variableModOrAlias |=
            std::any_of(aliasedRegions.begin(),
                        aliasedRegions.end(),
                        [](view::Region &r) { return !r.isEmpty(); });
      }
      logging::op::trace("Op {} consumes variable tensor {} ({}), touches: {}",
                         debugName(),
                         varTensor->id,
                         inIndexAndTensor.second->id,
                         variableModOrAlias ? "yes" : "no");
      if (variableModOrAlias) {
        // Variable tensors force the VGID to be such that the weight
        // is not modified or aliased on any other VGID than the one where
        // the weight is stored.
        for (Op *consumer : varTensor->consumers.getOps()) {
          if (consumer != this && consumer->hasVirtualGraphId()) {
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

  bool pipeline = ir.getSessionOptions().enablePipelining;
  bool pingpong = ir.getSessionOptions().pingPongPhases > 1;
  bool vgraphs =
      ir.getSessionOptions().virtualGraphMode != VirtualGraphMode::Off;

  // Sort function to find the Op from which to inherit attributes
  // Sorting criteria:
  // - Producers of inputs before consumers of outputs
  // - Producers in descending order of
  //    - PipelineStage
  //    - PingPongPhase
  //    - VGID
  //    - BatchSerializedPhase
  // - Consumers in ascending order of
  //    - PipelineStage
  //    - PingPongPhase
  //    - VGID
  //    - BatchSerializedPhase
  auto opSorter = [pipeline,
                   pingpong,
                   vgraphs,
                   &getOpVGID,
                   &getOpPingPongPhase](
                      const std::pair<Op *, ConnectedOpRelation> &lhs,
                      const std::pair<Op *, ConnectedOpRelation> &rhs) {
    Op *lhsOp                  = lhs.first;
    ConnectedOpRelation lhsRel = lhs.second;
    bool lhsProducer           = lhsRel != ConnectedOpRelation::Consumer;
    bool lhsPipeline           = pipeline && lhsOp->hasPipelineStage();
    bool lhsPingpong           = pingpong && lhsOp->hasPingPongPhase();
    bool lhsVirtual =
        vgraphs && (lhsOp->hasVirtualGraphId() || lhsOp->isIpuCopyOp());
    Op *rhsOp                  = rhs.first;
    ConnectedOpRelation rhsRel = rhs.second;
    bool rhsProducer           = rhsRel != ConnectedOpRelation::Consumer;
    bool rhsPipeline           = pipeline && rhsOp->hasPipelineStage();
    bool rhsPingpong           = pingpong && rhsOp->hasPingPongPhase();
    bool rhsVirtual =
        vgraphs && (rhsOp->hasVirtualGraphId() || rhsOp->isIpuCopyOp());

    std::tuple<bool,
               PipelineStage,
               PingPongPhase,
               VGraphId,
               BatchSerializedPhase,
               OpId>
        lhsTuple(
            lhsProducer,
            (lhsProducer ? 1 : -1) *
                (lhsPipeline ? lhsOp->getPipelineStage() : unusedPipelineStage),
            (lhsProducer ? 1 : -1) * (lhsPingpong
                                          ? *getOpPingPongPhase(lhsOp, lhsRel)
                                          : unusedPingPongPhase),
            (lhsProducer ? 1 : -1) *
                (lhsVirtual ? *getOpVGID(lhsOp, lhsRel) : unusedVGraphId),
            (lhsProducer ? 1 : -1) * (lhsOp->hasBatchSerializedPhase()
                                          ? lhsOp->getBatchSerializedPhase()
                                          : unusedBatchSerializedPhase),
            lhsOp->id);

    std::tuple<bool,
               PipelineStage,
               PingPongPhase,
               VGraphId,
               BatchSerializedPhase,
               OpId>
        rhsTuple(
            rhsProducer,
            (rhsProducer ? 1 : -1) *
                (rhsPipeline ? rhsOp->getPipelineStage() : unusedPipelineStage),
            (rhsProducer ? 1 : -1) * (rhsPingpong
                                          ? *getOpPingPongPhase(rhsOp, rhsRel)
                                          : unusedPingPongPhase),
            (rhsProducer ? 1 : -1) *
                (rhsVirtual ? *getOpVGID(rhsOp, rhsRel) : unusedVGraphId),
            (rhsProducer ? 1 : -1) * (rhsOp->hasBatchSerializedPhase()
                                          ? rhsOp->getBatchSerializedPhase()
                                          : unusedBatchSerializedPhase),
            rhsOp->id);
    return lhsTuple > rhsTuple;
  };

  for (auto inIndexAndTensor : input->tensorMap()) {
    if (inIndexAndTensor.second->hasProducer()) {
      connectedOps.emplace_back(inIndexAndTensor.second->getProducer(),
                                ConnectedOpRelation::Producer);
    }
  }

  for (auto outIndexAndTensor : output->tensorMap()) {
    for (Op *consumer : outIndexAndTensor.second->consumers.getOps()) {
      connectedOps.emplace_back(consumer, ConnectedOpRelation::Consumer);
    }
  }

  bool inherited = false;
  if (!connectedOps.empty()) {
    auto connectedOp =
        *std::min_element(connectedOps.begin(), connectedOps.end(), opSorter);
    Op *op                  = connectedOp.first;
    ConnectedOpRelation rel = connectedOp.second;

    if (pipeline) {
      setPipelineStage(op->getOptionalPipelineStage());
    }

    if (pingpong) {
      setPingPongPhase(getOpPingPongPhase(op, rel));
    }

    if (vgraphs && !isIpuCopyOp()) {
      setVirtualGraphId(getOpVGID(op, rel));
    }

    if (inheritSerializations) {
      setBatchSerializedPhase(op->getOptionalBatchSerializedPhase());
    }

    inherited = true;
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
      (!hasVirtualGraphId() || getVirtualGraphId() != *requiredVgid)) {
    logging::op::debug("Changing Op {} placement to required VGID: {}",
                       debugName(),
                       *requiredVgid);
    setVirtualGraphId(requiredVgid);
    if (hasPingPongPhase()) {
      setPingPongPhase(getPingPongPhase() + 1);
    }
    inherited = true;
  }

  if (!inherited) {
    logging::op::warn("Could not inherit placement attributes to Op {}",
                      debugName());
  }
}

const Shape &Op::inShape(InIndex index) const {
  return inTensor(index)->info.shape();
}

const Shape &Op::outShape(OutIndex index) const {
  return outTensor(index)->info.shape();
}

int Op::inRank(InIndex index) const { return inTensor(index)->info.rank(); }

int Op::outRank(InIndex index) const { return outTensor(index)->info.rank(); }

OutIndex Op::outIndex(Tensor *tensor) const {
  std::vector<OutIndex> outIndices = output->indices(tensor);
  if (outIndices.size() == 0) {
    throw internal_error("Cannot find output index of tensor {} for {}",
                         tensor->id,
                         debugName());
  } else if (outIndices.size() > 1) {
    throw internal_error("Tensor {} is an output of {} at more than one index",
                         tensor->id,
                         debugName());
  }
  return outIndices.at(0);
}

std::string Op::str() const {
  std::stringstream ss;
  ss << id << " (" << opid << ")";
  return ss.str();
}

std::string Op::debugName() const {
  std::string debug_id;
  if (!getName().empty()) {
    debug_id = getName();
  } else {
    std::stringstream ss;
    ss << opid;
    debug_id = ss.str();
  }

  std::vector<TensorId> in_ids;
  for (auto i : input->tensorIdMap()) {
    in_ids.push_back(i.second);
  }

  std::vector<TensorId> out_ids;
  for (auto i : output->tensorIdMap()) {
    out_ids.push_back(i.second);
  }

  return logging::format("Op({}, inputs=[{}], outputs=[{}])",
                         debug_id,
                         logging::join(in_ids.begin(), in_ids.end(), ", "),
                         logging::join(out_ids.begin(), out_ids.end(), ", "));
}

bool Op::isNorm() const { return false; }

// By default an operation cannot be replaced
bool Op::canBeReplacedByIdentity() { return false; }

std::map<fwtools::subgraph::InIndex, Op::SubgraphInSig>
Op::getSubgraphInputs() const {
  std::map<fwtools::subgraph::InIndex, Op::SubgraphInSig> ins;
  for (auto &index_tensor : input->tensorMap()) {
    auto inIndex       = index_tensor.first;
    auto tensor        = index_tensor.second;
    Op *unsafeProducer = tensor->getProducerUnsafe();
    // tensorflow will need some way if distinguishing
    // between tensors without producers
    fwtools::subgraph::OutIndex outIndex = -1;
    if (unsafeProducer) {
      outIndex = unsafeProducer->output->indicesMap().at(tensor).at(0);
    }
    ins[inIndex] = SubgraphInSig(unsafeProducer, outIndex, tensor->id);
  }
  return ins;
}

std::map<fwtools::subgraph::OutIndex, std::set<Op *>>
Op::getSubgraphOutputs() const {

  std::map<fwtools::subgraph::OutIndex, std::set<Op *>> cmap;

  for (auto &index_tensor : output->tensorMap()) {
    auto out_index  = index_tensor.first;
    auto out_tensor = index_tensor.second;
    std::set<Op *> consumers;
    if (settings.graph.get().getIr().isAnchored(out_tensor->id)) {
      consumers.insert(
          &settings.graph.get().getIr().getSubgraphAnchorPlaceholder());
    }
    for (auto &op : out_tensor->consumers.getOps()) {
      consumers.insert(op);
    }
    cmap[out_index] = consumers;
  }
  return cmap;
}

bool Op::isOutlineable() const { return true; }

void Op::getInTensorData(TensorId tensorId,
                         std::vector<int64_t> &data,
                         std::vector<DataType> dataTypes) {

  // check 1 : that there is already a tensor with the shape tensor's name
  if (!getGraph().getTensors().contains(tensorId)) {
    throw error("the tensor `" + tensorId + "` is not defined");
  }

  Tensor *tensor = getGraph().getTensors().get(tensorId);

  // check 2 : that the tensor has data
  if (!tensor->hasTensorData()) {
    throw error("the tensor `" + tensorId + "` does not have data");
  }

  TensorData *tensorData = tensor->tensorData();

  // check 3 : that the data is the expected type
  bool validType = false;
  for (auto dt : dataTypes) {
    if (tensor->info.dataType() == dt) {
      validType = true;
      break;
    }
  }

  if (!validType) {
    throw error("the tensor `" + tensorId +
                "` is not the correct type, it is " + tensor->info.data_type());
  }

  // check 5 : that is is rank 0 or rank 1
  if (tensor->info.rank() > 1) {
    throw error("the rank of tensor `" + tensorId + "` is greater than 1");
  }

  if (tensor->info.dataType() == DataType::INT32) {
    int32_t *pdata = static_cast<int32_t *>(tensorData->data());
    for (int i = 0; i < tensor->info.nelms(); ++i) {
      data.push_back(pdata[i]);
    }
  } else if (tensor->info.dataType() == DataType::INT64) {
    int64_t *pdata = static_cast<int64_t *>(tensorData->data());
    for (int i = 0; i < tensor->info.nelms(); ++i) {
      data.push_back(pdata[i]);
    }
  } else {
    throw error("unsupported data type {} for tensor `{}`",
                tensor->info.data_type(),
                tensorId);
  }
}

std::ostream &operator<<(std::ostream &ss, const GradInOutMapper &g) {
  ss << logging::format("GradInOutMapper(iGrad: {}, iNonGrad: {}, type: {})",
                        g.iGrad,
                        g.iNonGrad,
                        g.type);
  return ss;
}

std::ostream &operator<<(std::ostream &ss, const GradOpInType &t) {
  switch (t) {
  case GradOpInType::In: {
    ss << "GradOpInType::IN";
    break;
  }
  case GradOpInType::Out: {
    ss << "GradOpInType::OUT";
    break;
  }
  case GradOpInType::GradOut: {
    ss << "GradOpInType::GRADOUT";
    break;
  }
  default:
    ss << logging::format("GradOpInType::UNDEFINED({})", static_cast<int>(t));
    break;
  }
  return ss;
}

bool Op::consumesAnchor() const {
  for (auto tensor : input->tensors()) {
    if (getIr().isAnchored(tensor->id)) {
      return true;
    }
  }
  return false;
}

bool Op::producesAnchor() const {
  for (auto tensor : output->tensors()) {
    if (getIr().isAnchored(tensor->id)) {
      return true;
    }
  }
  return false;
}

bool Op::consumesCheckpointAndIsRecompute() const {
  if (settings.recomputeType == RecomputeType::Recompute) {
    for (auto &index_tensor : input->tensorMap()) {
      auto inTensor = index_tensor.second;
      // Tensors without producers are effectively Checkpointed
      if (!inTensor->hasProducer() ||
          (inTensor->hasProducer() &&
           inTensor->getProducer()->settings.recomputeType ==
               RecomputeType::Checkpoint)) {
        return true;
      }
    }
  }
  return false;
}

bool Op::consumesImplicitLoopInput() const {
  for (auto &index_tensor : input->tensorMap()) {
    auto inTensor = index_tensor.second;
    if (inTensor->isImplicitLoopInput()) {
      return true;
    }
  }
  return false;
}

bool Op::consumesRestoredInplaceTensor() const {
  for (auto tensor : input->tensors()) {
    for (auto consumer : tensor->consumers.getOps()) {
      if (consumer->isConvertibleTo<RestoreInplaceOp>()) {
        return true;
      }
    }
  }
  return false;
}

bool Op::consumesGraphOutput() const {

  const auto graphOutputs = getGraph().getOutputIds();

  const auto opInTensors = input->tensors();
  return std::any_of(opInTensors.cbegin(),
                     opInTensors.cend(),
                     [graphOutputs](const Tensor *inTensor) {
                       return std::find(graphOutputs.cbegin(),
                                        graphOutputs.cend(),
                                        inTensor->id) != graphOutputs.cend();
                     });
}

bool Op::producesGraphOutput() const {

  const auto graphOutputs = getGraph().getOutputIds();

  const auto opOutTensors = output->tensors();

  return std::any_of(opOutTensors.cbegin(),
                     opOutTensors.cend(),
                     [graphOutputs](const Tensor *outTensor) {
                       return std::find(graphOutputs.cbegin(),
                                        graphOutputs.cend(),
                                        outTensor->id) != graphOutputs.cend();
                     });
}

std::string Op::getInputsUnmodifiableString() const {
  std::ostringstream oss;
  oss << "([produces anchor ? " << producesAnchor() << "], consumes anchor ? "
      << consumesAnchor() << ", consumes checkpoint and is recompute ? "
      << consumesCheckpointAndIsRecompute()
      << ", consumes implicit loop input ? " << consumesImplicitLoopInput()
      << ", consumes graph output ? " << consumesGraphOutput() << ')';
  return oss.str();
}

bool Op::modifies() const {
  for (const auto &index_tensor : input->tensorMap()) {
    auto index = index_tensor.first;
    for (auto reg : modifies(index)) {
      if (!reg.isEmpty()) {
        return true;
      }
    }
  }

  return false;
}

bool Op::inputsUnmodifiable() const {
  return

      // Anchor tensors must not be modified to ensure the correct values are
      // returned. Here we conservatively assume anchors are returned at the
      // very end of the computation
      consumesAnchor()

      // Checkpoint tensors must not be modified by recompute Ops to ensure
      // the same value is used on first and second runs of the recompute Op
      || consumesCheckpointAndIsRecompute()

      // Implicit loop counter tensors must not be modified, because each loop
      // iteration needs access to the unmodified original input.
      || consumesImplicitLoopInput()

      // A simple (but overly strict) way to ensure that an op is not inplaced
      // if:
      // - its input, or a tensor it aliases, is restored inplace
      // - and its output, or a tensor that is an alias of it, is consumed
      //   by an ipucopy
      // TODO T19283: Make less strict once we can determine if any two tensors
      // are aliases of eachother
      || consumesRestoredInplaceTensor()

      // Graph output tensors must not be modified to ensure the correct value
      // is returned at the end of the computation
      || consumesGraphOutput();
}

bool Op::canShard() const { return false; }

std::map<TensorId, std::vector<TensorId>>
Op::shard(const std::map<TensorId, std::vector<TensorId>> &inputs) {
  std::map<TensorId, std::vector<TensorId>> outputs;
  size_t num_shards = 1;
  for (auto &idkv : inputs) {
    num_shards = std::max(num_shards, idkv.second.size());
  }

  auto &graph = getGraph();
  std::vector<Op *> cloneOps;
  for (size_t b = 0; b < num_shards; ++b) {
    auto clonedOpUp = clone();
    auto cloneId    = graph.moveIntoGraph(std::move(clonedOpUp));
    Op *clonedOp    = graph.getOp(cloneId);
    clonedOp->disconnectAllInputs();
    clonedOp->disconnectAllOutputs();
    for (const auto &in : input->tensorMap()) {
      auto serializedTensor = inputs.find(in.second->id);
      if (serializedTensor == inputs.end()) {
        // Tensors not split
        clonedOp->connectInTensor(in.first, in.second->id);
      } else {
        if (serializedTensor->second.size() == num_shards) {
          // Tensors split dimension
          clonedOp->connectInTensor(in.first, serializedTensor->second[b]);
        } else if (serializedTensor->second.size() == 1) {
          // Tensors not split
          clonedOp->connectInTensor(in.first, serializedTensor->second[0]);
        } else {
          throw error("[Op] Number of input tensors must be 1 or match the "
                      "serialziation factor {}",
                      num_shards);
        }
      }
    }
    for (const auto &out : output->tensorMap()) {
      TensorId sliceId =
          getIr().createBatchSliceTensorId(out.second->id,
                                           static_cast<unsigned>(b),
                                           static_cast<unsigned>(b + 1));
      clonedOp->createAndConnectOutTensor(out.first, sliceId);
      outputs[out.second->id].push_back(sliceId);
    }
    configureShardedOp(clonedOp, b);
    clonedOp->setup();

    logging::op::trace("[Op::shard] Cloned op {} {} -> {}",
                       clonedOp->opid,
                       clonedOp->input->getIndexShapeMap(),
                       clonedOp->output->getIndexShapeMap());
  }
  graph.topoCons->transferToMultiple(this, cloneOps);
  return outputs;
}

void Op::configureShardedOp(Op *const shardOp, int shardIndex) const {
  shardOp->setBatchSerializedPhase(shardIndex);
}

} // namespace popart
