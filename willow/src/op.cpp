#include <onnx/onnx_pb.h>
#include <spdlog/fmt/fmt.h>
#include <poponnx/ir.hpp>
#include <poponnx/region.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensors.hpp>
#include <poponnx/util.hpp>

// The layers:
#include <poponnx/op.hpp>
#include <poponnx/opmanager.hpp>

namespace poponnx {

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
  // TODO : merge these errors with those in bwdRegMap (T6707)
  if (!input->hasIndex(i)) {
    throw error("invalid index in fwdRegMap");
  } else if (!output->hasIndex(0)) {
    throw error("fwdMapReg called for op with no zero output");
  } else if (inShape(i) != outShape(0)) {
    throw error("default fwdRegMap not valid : should be specialised");
  }

  return [](const view::Region &r) { return r; };
}

view::RegMap Op::bwdRegMap(InIndex i) const {
  if (!input->hasIndex(i)) {
    throw error("invalid index in fwdRegMap");
  } else if (!output->hasIndex(0)) {
    throw error("fwdMapReg called for op with no zero output");
  } else if (inShape(i) != outShape(0)) {
    throw error("default fwdRegMap not valid : should be specialised");
  }

  return [](const view::Region &r) { return r; };
}

bool Op::isLossOp() const { return false; }

std::unique_ptr<Op> Op::clone() const {
  throw error("No clone implemented for {}", opid);
}

Op::~Op() = default;

// return a vector of 1 or several OpAndTensorIds for
// obtaining the gradient of the inputs of this Op.
// The Op in the OpAndTensorIds is the gradient op, and
// the TensorIds are the input indices of input of this
// Op for which the gradient is computed
std::vector<std::unique_ptr<Op>> Op::getGradOps() {
  throw error("Cannot get gradients for {}", opid);
}

void Op::setup() { throw error("No setup() for {}", opid); }

void Op::defaultConnectInTensor(InIndex inIndex, TensorId tenId) {
  Tensor *ptensor = getIr().getTensors().get(tenId);
  input->insert(inIndex, ptensor);
  ptensor->consumers.increment(this);
}

void Op::connectInTensor(InIndex inIndex, TensorId tenId) {
  defaultConnectInTensor(inIndex, tenId);
}

void Op::connectOutTensor(OutIndex outIndex, TensorId tenId) {
  Tensor *ptensor = getIr().getTensors().get(tenId);
  output->insert(outIndex, ptensor);
  if (ptensor->hasProducer()) {
    ptensor->resetProducer(this);
  } else {
    ptensor->setProducer(this);
  }
}

void Op::disconnectInTensor(InIndex inIndex, Tensor *tensor) {
  tensor->consumers.decrement(this);

  input->erase(inIndex);
}

void Op::disconnectOutTensor(Tensor *tensor) {
  for (auto idx : output->indices(tensor)) {
    tensor->resetProducer(nullptr);
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
  getIr().getTensors().addActGrad(tenId);
  Tensor *ptensor = getIr().getTensors().get(tenId);
  output->insert(outIndex, ptensor);
  ptensor->setProducer(this);
}

void Op::append(std::stringstream &ss) const {
  appendIO(ss);
  ss << '\n';
  appendMore(ss);
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
  return s.size() == nPathsToLoss();
}

int64_t Op::memOfOutputs() const {
  int64_t mem = 0;
  for (auto &t_inds : output->indicesMap()) {
    mem += t_inds.first->info.nbytes();
  }
  return mem;
}

void Op::appendIO(std::stringstream &ss) const {
  static std::string tab = "    ";

  ss << '\n' << "Op ";
  if (!getName().empty()) {
    ss << '"' << getName() << "\", ";
  }
  ss << id << " of type " << opid << '\n';

  int max_id_length = std::max(input->maxIdLength(), output->maxIdLength());

  ss << tab << "inputs" << '\n';
  input->append(ss, tab + tab, max_id_length);

  ss << '\n' << tab << "outputs" << '\n';
  output->append(ss, tab + tab, max_id_length);

  ss << '\n' << tab << "attributes" << '\n';
  appendAttributes(ss, tab + "    ");
}

void Op::appendAttributes(std::stringstream &ss, const std::string &tab) const {
  if (getRecomputeOutput())
    ss << tab << sRecomputeOutputAttribute << ":" << *(getRecomputeOutput())
       << '\n';

  if (getVirtualGraphId())
    ss << tab << sVirtualGraphAttribute << ":" << *(getVirtualGraphId())
       << '\n';
}

const std::string &Op::name() const { return getName(); }

Op::Op(const Op &op)
    : Vertex(op), input(new TensorIndexMap), output(new TensorIndexMap),
      priority(op.priority), id(op.settings.ir.getAndIncrOpsCounter()),
      opid(op.opid), settings(op.settings) {
  // input, output: empty.
}

Tensor *Op::inTensor(InIndex index) { return input->tensor(index); }
const Tensor *Op::inTensor(InIndex index) const { return input->tensor(index); }
Tensor *Op::outTensor(OutIndex index) { return output->tensor(index); }
const Tensor *Op::outTensor(OutIndex index) const {
  return output->tensor(index);
}

TensorId Op::inId(InIndex index) { return inTensor(index)->id; }
const TensorId Op::inId(InIndex index) const { return inTensor(index)->id; }
TensorId Op::outId(OutIndex index) { return outTensor(index)->id; }
const TensorId Op::outId(OutIndex index) const { return outTensor(index)->id; }

Op::Op(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : input(new TensorIndexMap), output(new TensorIndexMap), priority(0.0),
      // the id
      id(settings_.ir.getAndIncrOpsCounter()), opid(_opid),
      // the Attributes
      settings(settings_) {}

void Op::Op::Settings::setFromAttributes(const Attributes &attributes) {

  if (attributes.hasAttribute(sVirtualGraphAttribute)) {
    int64_t value;
    attributes.set(value, sVirtualGraphAttribute);
    vgraphId = value;
  }

  if (attributes.hasAttribute(sRecomputeOutputAttribute)) {
    int64_t value;
    attributes.set(value, sRecomputeOutputAttribute);
    recomputeOutput = value;
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

  return fmt::format("Op({}, outputs=[{}])",
                     debug_id,
                     fmt::join(out_ids.begin(), out_ids.end(), ", "));
}

// By default an operation can not be replaced
bool Op::canBeReplacedByIdentity() { return false; }

} // namespace poponnx
