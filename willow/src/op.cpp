#include <onnx/onnx_pb.h>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/util.hpp>

// The layers:
#include <popart/op/elementwise.hpp>
#include <popart/op/varupdate.hpp>

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

view::Region Op::uses(InIndex index) const {
  return view::Region::getFull(inShape(index));
}

view::Region Op::modifies(InIndex index) const {
  return view::Region::getEmpty(inRank(index));
};

view::Region Op::aliases(InIndex index) const {
  return view::Region::getEmpty(inRank(index));
};

view::RegMap Op::fwdRegMap(InIndex i) const {
  // TODO : merge these errors with those in bwdRegMap (T7107)
  if (!input->hasIndex(i)) {
    throw error("invalid index in fwdRegMap");
  } else if (!output->hasIndex(0)) {
    throw error("fwdMapReg called for op with no zero output");
  } else if (inShape(i) != outShape(0)) {
    throw error("default fwdRegMap not valid : should be specialised for {}",
                str());
  }

  return [](const view::Region &r) { return r; };
}

view::RegMap Op::bwdRegMap(InIndex i) const {
  if (!input->hasIndex(i)) {
    throw error("invalid index in bwdRegMap");
  } else if (!output->hasIndex(0)) {
    throw error("bwdMapReg called for op with no zero output");
  } else if (inShape(i) != outShape(0)) {
    throw error("default bwdRegMap not valid : should be specialised");
  }

  return [](const view::Region &r) { return r; };
}

bool Op::isLossOp() const { return false; }
bool Op::isIpuCopyOp() const { return false; }
bool Op::copiesOptimizerTensors() const { return false; }

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
  Tensor *ptensor = getGraph().getTensors().get(tenId);
  output->insert(outIndex, ptensor);
  if (ptensor->hasProducer()) {
    ptensor->resetProducer(this);
  } else {
    ptensor->setProducer(this);
  }

  // Output tensor takes fromLoss from op
  ptensor->fromLoss = fromLoss;
}

void Op::disconnectInTensor(Tensor *tensor) {
  for (auto i : input->indicesMap().at(tensor)) {
    disconnectInTensor(i, tensor);
  }
}

void Op::disconnectInTensor(InIndex inIndex, Tensor *tensor) {
  tensor->consumers.decrement(this);

  input->erase(inIndex);
}

void Op::disconnectOutTensor(Tensor *tensor) {
  for (auto idx : output->indices(tensor)) {
    // Trying to reset the producer here when the producer is not `this' should
    // probably be an error
    if (tensor->hasProducer() && tensor->getProducer() == this) {
      tensor->resetProducer(nullptr);
    }
    output->erase(idx);
  }
}

void Op::disconnectAllInputs() {
  for (auto entry : input->tensorMap()) {
    auto tensor = entry.second;
    tensor->consumers.decrement(this);
  }
  input->clear();
}

void Op::disconnectAllOutputs() {
  for (auto entry : output->tensorMap()) {
    auto tensor = entry.second;
    tensor->resetProducer(nullptr);
  }
  output->clear();
}

void Op::createAndConnectOutTensor(OutIndex outIndex, TensorId tenId) {
  tenId = (getScope() / tenId).str();

  getGraph().getTensors().addActGrad(tenId);
  Tensor *ptensor = getGraph().getTensors().get(tenId);
  output->insert(outIndex, ptensor);
  ptensor->setProducer(this);
}

std::string Op::getSubgraphEquivId() const {

  // Are any of the inputs aliased by output?
  bool noAliases = true;
  for (auto &index_tensor : input->tensorMap()) {
    noAliases = noAliases && aliases(index_tensor.first).isEmpty();
  }

  // Of all aliasing Ops, we only allow the VarUpdateOp to be outlined.
  // This partially resolves the failure to the propogate inplace modifications
  // through calls, T8604.
  bool aliasAndNotVarUpdate =
      !noAliases && !(dynamic_cast<const VarUpdateOp *>(this));

  if (isOutlineable() && settings.recomputeType != RecomputeType::RECOMPUTE &&
      !aliasAndNotVarUpdate) {
    OpEquivIdCreator os(this);
    appendAttributes(os);
    return os.str();
  }

  // in the case where the op is not outlineable, we return a unique string
  // to guarantee that it does not appear in any outline matches.
  std::stringstream ss;
  ss << str() << "_uid_" << id;
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
  os.appendAttribute("priority", static_cast<float>(priority));
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
  std::stringstream ss;
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

  std::string recomputeString =
      settings.recomputeType == RecomputeType::RECOMPUTE ? "YES" : "NO";
  os.appendAttribute("recompute", recomputeString);

  os.appendAttribute(sVirtualGraphAttribute, getOptionalVirtualGraphId());
  os.appendAttribute(sPipelineStageAttribute, settings.pipelineStage);
  os.appendAttribute("scope", getScope());
}

std::vector<const Graph *> Op::getCalledGraphs() const { return {}; }

std::vector<TensorId> Op::getInputsForGraph(const Graph &) const {
  throw error("Op does not call any graphs");
}

const std::string &Op::name() const { return getName(); }

Op::Op(const Op &op)
    : Vertex(op), input(new TensorIndexMap), output(new TensorIndexMap),
      priority(op.priority),
      id(op.settings.graph.get().getIr().getAndIncrOpsCounter()), opid(op.opid),
      settings(op.settings) {
  // input, output: empty.
}

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
    : input(new TensorIndexMap), output(new TensorIndexMap), priority(0.0),
      // the id
      id(settings_.graph.get().getIr().getAndIncrOpsCounter()), opid(_opid),
      // the Attributes
      settings(settings_) {}

Ir &Op::Op::Settings::getIr() const { return graph.get().getIr(); }

void Op::Op::Settings::setFromAttributes(const Attributes &attributes) {

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
    int64_t value;
    attributes.set(value, sRecomputeOutputAttribute);
    recomputeType =
        value ? RecomputeType::RECOMPUTE : RecomputeType::CHECKPOINT;
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
}

void Op::setVirtualGraphId(const boost::optional<int64_t> value) {
  settings.vgraphId = value;
}

const boost::optional<VGraphId> Op::getOptionalVirtualGraphId() const {
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

VGraphId Op::getIntrospectionInVirtualGraphId(InIndex) const {
  return getVirtualGraphId();
}

VGraphId Op::getIntrospectionOutVirtualGraphId(OutIndex) const {
  return getVirtualGraphId();
}

bool Op::hasVirtualGraphId() const {
  if (settings.vgraphId) {
    return true;
  } else {
    return false;
  }
}

boost::optional<PipelineStage> Op::getOptionalPipelineStage() const {
  return settings.pipelineStage;
}

void Op::setPipelineStage(boost::optional<PipelineStage> value) {
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

const Shape &Op::inShape(InIndex index) const {
  return inTensor(index)->info.shape();
}

const Shape &Op::outShape(OutIndex index) const {
  return outTensor(index)->info.shape();
}

int Op::inRank(InIndex index) const { return inTensor(index)->info.rank(); }

int Op::outRank(InIndex index) const { return outTensor(index)->info.rank(); }

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

  std::vector<TensorId> out_ids;
  for (auto i : output->tensorIdMap()) {
    out_ids.push_back(i.second);
  }

  return logging::format("Op({}, outputs=[{}])",
                         debug_id,
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
  case GradOpInType::IN: {
    ss << "GradOpInType::IN";
    break;
  }
  case GradOpInType::OUT: {
    ss << "GradOpInType::OUT";
    break;
  }
  case GradOpInType::GRADOUT: {
    ss << "GradOpInType::GRADOUT";
    break;
  }
  default:
    ss << logging::format("GradOpInType::UNDEFINED({})", static_cast<int>(t));
    break;
  }
  return ss;
}

} // namespace popart
