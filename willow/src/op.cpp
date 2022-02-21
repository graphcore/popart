// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <boost/optional/optional_io.hpp>
#include <onnx/onnx_pb.h>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/mean.hpp>
#include <popart/op/sum.hpp>
#include <popart/opattributehelper.hpp>
#include <popart/opdebuginfo.hpp>
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

#include <sstream>
#include <poprithms/memory/inplace/crosslink.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <popart/alias/aliasmodel.hpp>

namespace {
using namespace popart;

// Shared implementation for Op::fwdRegMap and Op::bwdRegMap methods
view::RegMap defaultRegMapImpl(const Op &op,
                               InIndex i,
                               OutIndex o,
                               const std::string &methodName) {
  logging::op::trace(
      "[{}] for OP {} index {} -> {}", methodName, op.debugName(), i, o);
  if (!op.input->hasIndex(i) || !op.output->hasIndex(o)) {
    throw error("invalid index in {}", methodName);
  } else if (!op.output->hasIndex(o)) {
    throw error("{} called for op with no zero output", methodName);
  } else if (op.inShape(i) != op.outShape(o)) {
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

std::tuple<ReplEqOutputMap, ReplEqModifiedInputMap>
Op::fwdPropagateIsReplicaEqual(const AliasModel &aliasModel,
                               const ReplEqInputMap &inputMap,
                               ReplicaEqualAnalysisProxy &proxy) const {

  // Return a mapping where each output tensor is mapped to the logical
  // conjunction of the value assigned to input tensors. That is, by default
  // outputs are considered replica equal if *all* input tensors are.

  auto value = true;

  for (auto &input : input->tensorMap()) {
    // Four-valued equivalent of a logical and over inputs.
    value = value && inputMap.at(input.first);
  }

  // Prepare result map.
  ReplEqOutputMap outputMap;
  for (auto &output : output->tensorMap()) {
    outputMap[output.first] = value;
  }

  return {outputMap, proxy.getModifiedInputMapFromAliases(this, outputMap)};
}

bool Op::isLossOp() const { return false; }
bool Op::isIpuCopyOp() const { return false; }
bool Op::copiesOptimizerTensors() const { return false; }
bool Op::isOptimizerOp() const { return settings.optimizerOp; }
bool Op::isGradientClippingOp() const { return settings.gradientClippingOp; }

bool Op::requiresRandomSeed() const { return false; }
InIndex Op::getSeedInIndex() const {
  throw error("Op {} does not have random seed input tensor", str());
}

Op::~Op() = default;

void Op::setCalledSubgraphGradInfo(const FwdGraphToBwdGraphInfo &info_) {
  // do nothing.
}

// return a vector of 1 or several OpAndTensorIds for
// obtaining the gradient of the inputs of this Op.
// The Op in the OpAndTensorIds is the gradient op, and
// the TensorIds are the input indices of input of this
// Op for which the gradient is computed
std::vector<std::unique_ptr<Op>> Op::getGradOps() { return {}; }

void Op::setup() { throw error("No setup() for {}", opid); }

void Op::finalizeDebugInfo() { debugInfo.finalize(); }

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

void Op::connectInTensorLike(const Op *other, InIndex index, TensorId tenId) {
  IpuCopyOp *dstOp       = dynamic_cast<IpuCopyOp *>(this);
  const IpuCopyOp *srcOp = dynamic_cast<const IpuCopyOp *>(other);
  if (srcOp && dstOp) {
    if (!srcOp->hasInput(index)) {
      throw error("[Op::connectInTensorLike] Op {} has no input {}.",
                  srcOp->debugName(),
                  index);
    }
    TensorId srcTensorId = srcOp->input->tensor(index)->id;
    dstOp->connectInTensor(index, tenId, srcOp->getSourceIpu(srcTensorId));
  } else if (dstOp) {
    throw error(
        "[Op::connectInTensorLike] Op {} is an IpuCopyOp but {} is not.",
        dstOp->debugName(),
        other->debugName(),
        index);
  } else {
    connectInTensor(index, tenId);
  }
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
  std::vector<int> indices = input->indicesMap().at(tensor);
  for (auto i : indices) {
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

void Op::disconnectInTensor(InIndex inIndex) {
  if (!hasInput(inIndex)) {
    throw internal_error("error disconnecting tensor at index {} of Op {}. "
                         "There is no tensor at this index to disconnect.",
                         inIndex,
                         debugName());
  }
  disconnectInTensor(inIndex, inTensor(inIndex));
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

  // Avoid double scoping (could be improved by making TensorId a class T33644)
  if (tenId.find(getScope().str()) == std::string::npos) {
    tenId = (getScope() / tenId).str();
  }

  getGraph().getTensors().addActGrad(tenId, getDebugInfo());
  Tensor *ptensor = getGraph().getTensors().get(tenId);
  output->insert(outIndex, ptensor);
  ptensor->setProducer(this);
}

std::string Op::getSubgraphEquivId(
    const std::map<std::string, popart::any> &externalAttrs) const {
  std::stringstream ss;
  if (isOutlineable()) { // && !aliasAndNotVarUpdate) {
    OpEquivIdCreator os(this);
    OpSerialiserBase &opb = os;

    for (const auto &externalAttr : externalAttrs) {
      const std::string &key          = externalAttr.first;
      const any &value                = externalAttr.second;
      const std::type_info &valueType = value.type();

      if (valueType == typeid(float)) {
        opb.appendAttribute(key, any_cast<float>(value));
      } else if (valueType == typeid(double)) {
        opb.appendAttribute(key, any_cast<double>(value));
      } else if (valueType == typeid(int)) {
        opb.appendAttribute(key, any_cast<int>(value));
      } else if (valueType == typeid(int64_t)) {
        opb.appendAttribute(key, any_cast<int64_t>(value));
      } else if (valueType == typeid(uint32_t)) {
        opb.appendAttribute(key, any_cast<uint32_t>(value));
      } else if (valueType == typeid(uint64_t)) {
        opb.appendAttribute(key, any_cast<uint64_t>(value));
      } else if (valueType == typeid(std::string)) {
        opb.appendAttribute(key, any_cast<std::string>(value));
      } else if (valueType == typeid(std::vector<float>)) {
        opb.appendAttribute(key, any_cast<std::vector<float>>(value));
      } else if (valueType == typeid(std::vector<double>)) {
        opb.appendAttribute(key, any_cast<std::vector<double>>(value));
      } else if (valueType == typeid(std::vector<int64_t>)) {
        opb.appendAttribute(key, any_cast<std::vector<int64_t>>(value));
      } else if (valueType == typeid(Scope)) {
        opb.appendAttribute(key, any_cast<Scope>(value));
      } else if (valueType == typeid(bool)) {
        opb.appendAttribute(key, any_cast<bool>(value));
      } else if (valueType == typeid(nonstd::optional<int64_t>)) {
        opb.appendAttribute(key, any_cast<nonstd::optional<int64_t>>(value));
      } else if (valueType == typeid(nonstd::optional<float>)) {
        opb.appendAttribute(key, any_cast<nonstd::optional<float>>(value));
      } else if (valueType == typeid(nonstd::optional<double>)) {
        opb.appendAttribute(key, any_cast<nonstd::optional<double>>(value));
      } else if (valueType == typeid(std::map<TensorId, uint64_t>)) {
        opb.appendAttribute(key, any_cast<std::map<TensorId, uint64_t>>(value));
      } else {
        throw error("[Op::getSubgraphEquivId] Unsupported attribute type for"
                    "attribute '{}' ({})",
                    key,
                    valueType.name());
      }
    }

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

int64_t Op::memOfOutputs() const {
  int64_t mem = 0;
  for (auto &t_inds : output->indicesMap()) {
    mem += t_inds.first->info.nbytes();
  }
  return mem;
}

void Op::appendAttributes(OpSerialiserBase &os) const {
  std::ostringstream executionContextSs;
  executionContextSs << settings.executionContext;
  appendOutlineAttributes(os);
  os.appendAttribute(sExecutionPhaseAttribute, settings.executionPhase);
  os.appendAttribute(sExecutionContextAttribute, executionContextSs.str());
  os.appendAttribute(sPipelineStageAttribute, settings.pipelineStage);
  os.appendAttribute("scope", getScope());

  // Can not add debugInfoId to the attributes as this will break
  // cache executables - debugInfoId would be part of the hash.
  // The id will be different each time the `same` graph is built.
  //
  // os.appendAttribute(sDebugInfoId, settings.debugInfoId);
}

void Op::appendOutlineAttributes(OpSerialiserBase &os) const {
  std::string recomputeString =
      settings.recomputeType == RecomputeType::Recompute ? "YES" : "NO";
  os.appendAttribute("recompute", recomputeString);
  os.appendAttribute(sVirtualGraphAttribute, getOptionalVGraphId());
  os.appendAttribute("tileSet", static_cast<int64_t>(settings.tileSet));
  for (auto attribute : settings.extraOutlineAttributes) {
    os.appendAttribute(attribute.first,
                       attribute.first + ":" + attribute.second);
  }
}

std::vector<const Graph *> Op::getCalledGraphs() const { return {}; }

std::vector<GraphId> Op::getCalledGraphIds() const {
  std::vector<GraphId> graphIds;
  for (auto graph : getCalledGraphs()) {
    graphIds.push_back(graph->id);
  }
  return graphIds;
}

SubgraphIndex Op::getCalledGraphIndex(const GraphId &id) const {
  auto calledGraphs = getCalledGraphIds();
  auto it           = std::find(calledGraphs.begin(), calledGraphs.end(), id);
  if (it == calledGraphs.end()) {
    throw error("[getCalledGraphIndex] Op {} does not call graph {}.",
                debugName(),
                id.str());
  } else {
    return std::distance(calledGraphs.begin(), it);
  }
}

Shape Op::prettyNpOut(const Shape &s0, const Shape &s1) const {
  std::stringstream ss;
  ss << "Op " << str();

  return npOut(s0, s1, ss.str());
}

TensorInfo Op::prettyNpOut(const TensorInfo &i0,
                           const TensorInfo &i1,
                           bool checkDataType) const {
  std::stringstream ss;
  ss << "Op " << str();

  return npOut(i0, i1, checkDataType, ss.str());
}

InIndex Op::opInToSubgraphInIndex(SubgraphIndex subgraphIndex,
                                  InIndex inIndex) const {
  throw error("Op {} has no subgraphs", debugName());
}

InIndex Op::subgraphInToOpInIndex(SubgraphIndex subgraphIndex,
                                  InIndex inIndex) const {
  throw error("Op {} has no subgraphs", debugName());
}

OutIndex Op::opOutToSubgraphOutIndex(SubgraphIndex subgraphIndex,
                                     OutIndex outIndex) const {
  throw error("Op {} has no subgraphs", debugName());
}

OutIndex Op::subgraphOutToOpOutIndex(SubgraphIndex subgraphIndex,
                                     OutIndex outIndex) const {
  throw error("Op {} has no subgraphs", debugName());
}

std::set<OutIndex> Op::opInToOpOutIndex(InIndex in) const {
  std::set<OutIndex> indices;
  // By default, traverse every output
  for (auto out : output->tensorMap()) {
    indices.insert(out.first);
  }
  return indices;
}

std::set<InIndex> Op::opOutToOpInIndex(OutIndex out) const {
  std::set<OutIndex> indices;
  // By default, traverse every input
  for (auto in : input->tensorMap()) {
    indices.insert(in.first);
  }
  return indices;
}

const std::string &Op::name() const { return getName(); }

std::string idStr(Op &op) {
  if (!op.name().empty()) {
    return op.name() + sNameDelimiter + std::to_string(op.id);
  } else {
    return std::to_string(op.id);
  }
}

Op::Op(const Op &op)
    : Vertex(op), input(new TensorIndexMap), output(new TensorIndexMap),
      id(op.settings.graph.get().getIr().getAndIncrOpsCounter()), opid(op.opid),
      settings(op.settings),
      debugInfo(
          {DebugNameAndId(std::to_string(id), op.settings.debugInfoId, name())},
          *this) {
  // input, output: empty.
}

bool Op::hasInput(InIndex index) const { return input->hasIndex(index); }
bool Op::hasOutput(InIndex index) const { return output->hasIndex(index); }

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
      settings(settings_),
      debugInfo(
          {DebugNameAndId(std::to_string(id), settings.debugInfoId, name())},
          *this) {}

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

std::ostream &operator<<(std::ostream &ost, const ExecutionContext &ec) {
  switch (ec) {
  case (ExecutionContext::Normal): {
    ost << "Normal";
    break;
  }
  case (ExecutionContext::AccumulateOuterFragment): {
    ost << "AccumulateOuterFragment";
    break;
  }
  case (ExecutionContext::WeightsFromHostFragment): {
    ost << "WeightsFromHostFragment";
    break;
  }
  case (ExecutionContext::WeightsToHostFragment): {
    ost << "WeightsToHostFragment";
    break;
  }
  case (ExecutionContext::OptimizerFromHostFragment): {
    ost << "OptimizerFromHostFragment";
    break;
  }
  case (ExecutionContext::Subgraph): {
    ost << "Subgraph";
    break;
  }
  }
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const ReductionType &rt) {
  switch (rt) {
  case (ReductionType::NoReduction): {
    ost << "NoReduction";
    break;
  }
  case (ReductionType::Sum): {
    ost << "Sum";
    break;
  }
  case (ReductionType::Mean): {
    ost << "Mean";
    break;
  }
  case (ReductionType::N): {
    throw error("[Op::operator<<] Unsupported reduction type");
    break;
  }
  }
  return ost;
}

void Op::Op::Settings::setFromAttributes(const Attributes &attributes) {

  if (attributes.hasAttribute(sExecutionPhaseAttribute)) {
    int64_t value;
    attributes.set(value, sExecutionPhaseAttribute);
    executionPhase = value;
  }

  if (attributes.hasAttribute(sExecutionContextAttribute)) {
    int64_t value;
    attributes.set(value, sExecutionContextAttribute);
    executionContext = static_cast<ExecutionContext>(value);
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

  if (attributes.hasAttribute(sOutputTensorLocationAttribute)) {
    std::vector<int64_t> tensorLocationTmp;
    attributes.set(tensorLocationTmp, sOutputTensorLocationAttribute);
    tensorLocation = TensorLocation(tensorLocationTmp);
  }

  if (attributes.hasAttribute(sSchedulePriority)) {
    float schedule_priority;
    attributes.set(schedule_priority, sSchedulePriority);
    schedulePriority = static_cast<double>(schedule_priority);
  }

  if (attributes.hasAttribute(sTileSetAttribute)) {
    int64_t tileSetTmp;
    attributes.set(tileSetTmp, sTileSetAttribute);
    tileSet = static_cast<TileSet>(tileSetTmp);
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

  if (attributes.hasAttribute(sDebugInfoId)) {
    int64_t debug_info_id;
    attributes.set(debug_info_id, sDebugInfoId);
    debugInfoId = debug_info_id;
  }
}

// Return suitable settings for an Op inserted before the input to an existing
// Op
Op::Settings Op::getInSettings(InIndex index) const {
  Op::Settings inSettings = getSettings();
  auto vs                 = getIntrospectionInVirtualGraphId(index);
  if (vs.first != unusedVGraphId) {
    inSettings.vgraphId = vs.first;
  }
  inSettings.tileSet = vs.second;
  return inSettings;
}

// Return suitable settings for an Op inserted after the output to an existing
// Op
Op::Settings Op::getOutSettings(OutIndex index) const {
  Op::Settings outSettings = getSettings();
  auto vs                  = getIntrospectionOutVirtualGraphId(index);
  if (vs.first != unusedVGraphId) {
    outSettings.vgraphId = vs.first;
  }
  outSettings.tileSet = vs.second;
  return outSettings;
}

// Adjust the settings to be suitable as input at InIndex
Op::Settings Op::adjustInSettings(InIndex index, Op::Settings settings_) const {
  Op::Settings inSettings = getInSettings(index);
  if (inSettings.vgraphId != settings.vgraphId) {
    settings_.vgraphId = inSettings.vgraphId;
  }
  if (inSettings.tileSet != settings.tileSet) {
    settings_.tileSet = inSettings.tileSet;
  }
  return settings_;
}

// Adjust the settings to be suitable as output at OutIndex
Op::Settings Op::adjustOutSettings(OutIndex index,
                                   Op::Settings settings_) const {
  Op::Settings outSettings = getOutSettings(index);
  if (outSettings.vgraphId != settings.vgraphId) {
    settings_.vgraphId = outSettings.vgraphId;
  }
  if (outSettings.tileSet != settings.tileSet) {
    settings_.tileSet = outSettings.tileSet;
  }
  return settings_;
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

VGraphIdAndTileSet Op::getIntrospectionInVirtualGraphId(InIndex index) const {
  std::set<OpId> visited;
  return getIntrospectionInVirtualGraphId(index, visited);
}

VGraphIdAndTileSet Op::getIntrospectionOutVirtualGraphId(OutIndex index) const {
  std::set<OpId> visited;
  return getIntrospectionOutVirtualGraphId(index, visited);
}

VGraphIdAndTileSet
Op::getIntrospectionInVirtualGraphId(InIndex, std::set<OpId> &visited) const {
  return {hasVirtualGraphId() ? getVirtualGraphId() : unusedVGraphId,
          settings.tileSet};
}

VGraphIdAndTileSet
Op::getIntrospectionOutVirtualGraphId(OutIndex, std::set<OpId> &visited) const {
  return {hasVirtualGraphId() ? getVirtualGraphId() : unusedVGraphId,
          settings.tileSet};
}

bool Op::hasVirtualGraphId() const {
  if (settings.vgraphId) {
    return true;
  } else {
    return false;
  }
}

const OptionalExecutionPhase Op::getOptionalExecutionPhase() const {
  return settings.executionPhase;
}

void Op::setExecutionPhase(const OptionalExecutionPhase value) {
  settings.executionPhase = value;
}

ExecutionPhase Op::getExecutionPhase() const {
  if (!hasExecutionPhase()) {
    throw error("Cannot return ExecutionPhase for Op {}. "
                "It has not had this attribute set",
                debugName());
  }
  return *(settings.executionPhase);
}

bool Op::hasExecutionPhase() const {
  if (settings.executionPhase) {
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

const OptionalStochasticRoundingMethod
Op::getOptionalStochasticRoundingMethod() const {
  return settings.stochasticRoundingMethod;
}

void Op::setStochasticRoundingMethod(
    const OptionalStochasticRoundingMethod value) {
  settings.stochasticRoundingMethod = value;
}

StochasticRoundingMethod Op::getStochasticRoundingMethod() const {
  if (!hasStochasticRoundingMethod()) {
    throw error("Cannot return StochasticRoundingMethod for Op {}. "
                "It has not had this attribute set",
                debugName());
  }
  return *(settings.stochasticRoundingMethod);
}

bool Op::hasStochasticRoundingMethod() const {
  if (settings.stochasticRoundingMethod) {
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

void Op::inheritPlacementAttributes(bool inheritSerializations,
                                    AliasModel &aliasModel) {
  InheritOpAttributeHelper::apply(this, inheritSerializations, aliasModel);
}

const Shape &Op::inShape(InIndex index) const {
  return inTensor(index)->info.shape();
}

const Shape &Op::outShape(OutIndex index) const {
  return outTensor(index)->info.shape();
}

int Op::inRank(InIndex index) const { return inTensor(index)->info.rank(); }

int Op::outRank(InIndex index) const { return outTensor(index)->info.rank(); }

InIndex Op::inIndex(Tensor *tensor) const {
  if (!input->contains(tensor)) {
    throw internal_error(
        "Cannot find input index of tensor {} for {}", tensor->id, debugName());
  }
  auto inIndices = input->indices(tensor);
  if (inIndices.size() > 1) {
    throw internal_error("Tensor {} is an input of {} at more than one index",
                         tensor->id,
                         debugName());
  }
  return inIndices.at(0);
}

OutIndex Op::outIndex(Tensor *tensor) const {
  if (!output->contains(tensor)) {
    throw internal_error("Cannot find output index of tensor {} for {}",
                         tensor->id,
                         debugName());
  }
  auto outIndices = output->indices(tensor);
  if (outIndices.size() > 1) {
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
  std::stringstream ss;
  if (!getName().empty()) {
    ss << getName();
    ss << " (" << opid << ")";
  } else {
    ss << opid;
  }
  debug_id = ss.str();

  std::vector<TensorId> in_ids;
  for (auto i : input->tensorIdMap()) {
    in_ids.push_back(i.second);
  }

  std::vector<TensorId> out_ids;
  for (auto i : output->tensorIdMap()) {
    out_ids.push_back(i.second);
  }

  std::vector<std::string> subgraphs;
  for (auto &g : getCalledGraphs()) {
    std::stringstream ss;
    ss << g->id.str();
    ss << "(";
    ss << logging::join(g->getInputIds().begin(), g->getInputIds().end(), ", ");
    ss << "; ";
    ss << logging::join(
        g->getOutputIds().begin(), g->getOutputIds().end(), ", ");
    ss << ")";
    subgraphs.push_back(ss.str());
  }

  std::string subgraph_fmt =
      logging::format("subgraphs=[{}], ",
                      logging::join(subgraphs.begin(), subgraphs.end(), ", "));

  return logging::format("Op({}, {}inputs=[{}], outputs=[{}])",
                         debug_id,
                         subgraphs.empty() ? "" : subgraph_fmt,
                         logging::join(in_ids.begin(), in_ids.end(), ", "),
                         logging::join(out_ids.begin(), out_ids.end(), ", "));
}

bool Op::isNorm() const { return false; }

// By default an operation cannot be replaced
bool Op::canBeReplacedByIdentity() const { return false; }

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

float Op::calcAutoVirtualGraphCost(std::set<int> &inputs_seen) {

  std::set<int> outputs_seen;
  float total = 0.0f;

  // Check if backwards pass
  for (auto &gradOp : getGradOps()) {
    for (auto &inOutMapper : gradOp->gradInputInfo()) {
      int indexFwd      = inOutMapper.iNonGrad;
      GradOpInType type = inOutMapper.type;
      // the input at index 'indexGrad' to gradOp is
      switch (type) {
      // An input to the fwd Op. Ignore weights seen previously.
      case GradOpInType::In: {
        bool exists = inputs_seen.insert(indexFwd).second;
        if (exists) {
          // This will need checking
          total += static_cast<float>(input->tensor(indexFwd)->info.nbytes());
        }
        break;
      }

      //  An output from the fwd Op.
      case GradOpInType::Out: {
        bool exists = outputs_seen.insert(indexFwd).second;
        if (exists) {
          total += static_cast<float>(output->tensor(indexFwd)->info.nbytes());
        }
        break;
      }

      // This is the data that passes through the backwards pass.
      // Unless the VarUpdate is done as a single compute_set
      // This input can be ignored as not 'always live'
      case GradOpInType::GradOut: {
        break;
      }
      }
    }
  }

  return total;
}

bool Op::isOutlineable() const { return true; }

bool Op::hasSideEffect() const { return false; }

bool Op::canRecompute() const { return !hasSideEffect(); }

void Op::getInTensorData(TensorId tensorId,
                         std::vector<int64_t> &data,
                         std::vector<DataType> dataTypes) {

  // check 1 : that there is already a tensor with the shape tensor's name
  if (!getGraph().getTensors().contains(tensorId)) {
    throw error("the tensor `" + tensorId + "` is not defined");
  }

  Tensor *tensor = getGraph().getTensors().get(tensorId);

  std::vector<char> tensorData = tensor->getDataViaGraphTraversal();
  void *ptr                    = static_cast<void *>(tensorData.data());

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
    int32_t *pdata = static_cast<int32_t *>(ptr);
    for (int i = 0; i < tensor->info.nelms(); ++i) {
      data.push_back(pdata[i]);
    }
  } else if (tensor->info.dataType() == DataType::INT64) {
    int64_t *pdata = static_cast<int64_t *>(ptr);
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

bool Op::inputUnmodifiable(InIndex in) const {
  auto t = input->tensor(in);
  // If the input tensor itself, or any of it's aliases, are unmodifiable
  return t->anyAlias([](Tensor *tensor) { return tensor->isUnmodifiable(); });
}

bool Op::hasAliasedModifiers(OutIndex out) const {
  auto t = output->tensor(out);

  auto checkConsumers = [](Tensor *t_in) {
    for (Op *consumer : t_in->consumers.getOps()) {
      for (InIndex in : consumer->input->indices(t_in)) {
        auto regions = consumer->modifies(in);
        if (regionsModified(regions)) {
          return true;
        }
      }
    }
    return false;
  };

  return t->anyAlias(checkConsumers);
}

bool Op::isParentOf(const Op *op) const {
  // We're a parent of op if and only if op is our child.
  return op->isChildOf(this);
}

bool Op::isChildOf(const Op *op) const {
  // We're a direct child of op if we consume any of the tensors that op
  // produces.
  const auto opOutTensors = op->output->tensors();
  const auto inTensors    = input->tensors();
  std::vector<TensorId> inTensorIds;
  inTensorIds.reserve(inTensors.size());
  for (const auto inTensor : inTensors) {
    inTensorIds.push_back(inTensor->id);
  }
  return std::any_of(opOutTensors.cbegin(),
                     opOutTensors.cend(),
                     [&inTensorIds](const Tensor *outTensor) {
                       return std::find(inTensorIds.cbegin(),
                                        inTensorIds.cend(),
                                        outTensor->id) != inTensorIds.cend();
                     });
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

bool Op::modifiesIndex(InIndex in) const {
  auto region = modifies(in);
  if (regionsModified(region)) {
    return true;
  }
  return false;
}

bool Op::overwritesTensor(Tensor *t) const {
  auto consumers = t->consumers.getOps();
  if (std::find(consumers.begin(), consumers.end(), this) == consumers.end()) {
    return false;
  }

  auto indices = input->indices(t);

  bool overwrite = !indices.empty();

  for (auto index : indices) {
    auto regions = modifies(index);

    if (regions.size() > 0 &&
        regions.front() ==
            view::Region::getFull(t->info.shape(), view::AccessType::Write)) {
      overwrite &= true;
    }

    if ((regions.size() == 0) ||
        (std::any_of(regions.begin(), regions.end(), [](const view::Region &r) {
          return r.getAccessType() == view::AccessType::Read ||
                 r.getAccessType() == view::AccessType::ReadWrite ||
                 r.isEmpty();
        }))) {
      // Consumer without modification, or modifications after reading,
      // we assume the tensor is read and therefore not purely overwritten
      overwrite &= false;
    }
  }
  return overwrite;
}

bool Op::modifiesTensor(Tensor *t) const {
  auto consumers = t->consumers.getOps();
  if (std::find(consumers.begin(), consumers.end(), this) == consumers.end()) {
    return false;
  }

  auto indices = input->indices(t);

  for (auto index : indices) {
    if (modifiesIndex(index)) {
      return true;
    }
  }
  return false;
}

bool Op::inputsUnmodifiable() const {
  for (auto &in : input->tensorIdMap()) {
    if (inputUnmodifiable(in.first)) {
      return true;
    }
  }
  return false;
}

ReplicatedTensorShardingIndices Op::getReplicatedTensorShardingIndices() const {
  return {};
}

void Op::configureForReplicatedTensorSharding(
    ReplicatedTensorShardingIndices indices,
    CommGroup shardingDomain) {
  setup();
}

void Op::growAliasModelMulti(AliasModel &m) const {

  // poprithms::memory::inplace::Ops have contiguous input indices. PopART Ops
  // do not.
  //
  // We therefore remove the gaps in the PopART Tensor's input indices, back
  // packing contiguously (compactly). For example, if a PopART Op has inputs at
  // indices (0,2,3) the poprithms equivalent will have inputs at (0,1,2): the
  // "gap" at 1 is removed, and all the subsequent inputs shift down one index.
  //
  // The same is true for the outputs: all gaps are removed in for the poprithms
  // Op.

  std::vector<poprithms::memory::inplace::TensorId> inIds;
  for (const auto &index_tensor : input->tensorMap()) {
    inIds.push_back(m.getPoprithmsTensorId(index_tensor.second->id));
  }

  poprithms::memory::inplace::Shapes outShapes;
  for (const auto &kv : output->tensorMap()) {
    outShapes.push_back(kv.second->info.shape());
  }

  // Use PopART Ops' virtual methods to determine where the aliases lie between
  // inputs and outputs, and which of the are modifying (as opposed to just pure
  // "identity" aliases).

  std::vector<poprithms::memory::inplace::CrossLink> crossAliases;
  uint64_t poprInIndex{0};

  for (const auto &in_index_tensor : input->tensorMap()) {
    uint64_t poprOutIndex{0};
    const auto inIndex = in_index_tensor.first;
    for (const auto &out_index_tensor : output->tensorMap()) {

      const auto outIndex = out_index_tensor.first;

      // If there is an alias between the PopART Op from inIndex->outIndex, then
      // the corresponding poprithms Op must have an alias from
      // poprInIndex->poprOutIndex. If the PopART alias is modifying, then so
      // too must the poprithms Op be.

      if (doesAlias(inIndex, outIndex)) {
        auto m = modifiesIndex(inIndex)
                     ? poprithms::memory::inplace::CrossLink::modifies(
                           poprInIndex, poprOutIndex)
                     : poprithms::memory::inplace::CrossLink::pureAliases(
                           poprInIndex, poprOutIndex);
        crossAliases.push_back(m);
      }
      ++poprOutIndex;
    }
    ++poprInIndex;
  }

  const auto opId = m.g.multi(inIds, outShapes, crossAliases);

  uint64_t outIndex{0};
  for (const auto &kv : output->tensorMap()) {
    m.insertTensor({opId, outIndex}, *kv.second);
    ++outIndex;
  }
  // This still required in case no outputs
  m.insertOp(opId, id);
}

bool Op::doesAlias(InIndex inIndex, OutIndex outIndex) const {
  const auto regions = aliases(inIndex, outIndex);
  return std::any_of(regions.cbegin(),
                     regions.cend(),
                     [](const view::Region &r) { return !r.isEmpty(); });
}

bool Op::doesAlias() const {
  for (const auto &x : input->tensorMap()) {
    for (const auto &y : output->tensorMap()) {
      if (doesAlias(x.first, y.first)) {
        return true;
      }
    }
  }
  return false;
}

poprithms::memory::inplace::Proposal
Op::mapInplaceProposal(const AliasModel &aliasModel,
                       OperatorIdentifier opId) const {
  if (!isOutplace()) {
    throw error(
        "Invalid call to mapInplaceProposal for {}, as it is not outplace.",
        str());
  }

  throw error("mapInplaceProposal not implemented for {}", str());
}

void Op::growAliasModel(AliasModel &m) const {
  if (doesAlias()) {
    throw error("Ops which alias must implement growAliasModel, this for {} ",
                this->str());
  }

  if (!inplacePriorityDefault().empty()) {
    throw error(
        "Ops with inplace variants must implement growAliasModel, this for {} ",
        this->str());
  }
  growAliasModelMulti(m);
}

poprithms::memory::inplace::Proposal
Op::mapInplaceProposalGate0(const AliasModel &aliasModel,
                            OperatorIdentifier opId) const {

  if (!isOutplace()) {
    std::ostringstream oss;
    oss << "Inplacing logic error: "
        << "Ops which are not outplace should never set Proposals. "
        << "This for Op " << str() << ". ";
    throw error(oss.str());
  }

  return {aliasModel.getGate(id), 0};
}

void Op::transferBaseProperties(Op *to) {
  if (hasVirtualGraphId()) {
    to->setVirtualGraphId(getVirtualGraphId());
  }
  if (hasExecutionPhase()) {
    to->setExecutionPhase(getExecutionPhase());
  }
  if (hasPipelineStage()) {
    to->setPipelineStage(getPipelineStage());
  }
  if (hasBatchSerializedPhase()) {
    to->setBatchSerializedPhase(getBatchSerializedPhase());
  }

  to->settings.scope            = settings.scope;
  to->settings.recomputeType    = settings.recomputeType;
  to->settings.tensorLocation   = settings.tensorLocation;
  to->fromLoss                  = fromLoss;
  to->toLoss                    = toLoss;
  to->settings.schedulePriority = settings.schedulePriority;
  to->settings.debugInfoId      = settings.debugInfoId;
}

Op *Op::getPrecedingOp(InIndex inIndex) {
  return inTensor(inIndex)->getProducer();
}

Op *Op::getFollowingOp(OutIndex outIndex) {
  auto t = outTensor(outIndex);
  if (t->consumers.getTotal() == 1) {
    return t->consumers.getOps().at(0);
  } else if (t->consumers.getTotal() == 0) {
    throw internal_error(
        "There are no ops following {} out index {}", debugName(), outIndex);
  } else {
    throw internal_error("There are multiple ops following {} out index {}.",
                         debugName(),
                         outIndex);
  }
}

std::vector<Op *> Op::getFollowingOps(OutIndex outIndex) {
  auto t = outTensor(outIndex);
  return t->consumers.getOps();
}

} // namespace popart
