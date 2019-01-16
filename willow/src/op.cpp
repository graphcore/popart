#include <onnx/onnx_pb.h>
#include <spdlog/fmt/fmt.h>
#include <poponnx/ir.hpp>
#include <poponnx/tensor.hpp>

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

bool Op::modifies(InIndex) const {
  // default for ops is: No, it does not modify the input
  return false;
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
  Tensor *ptensor = pir->getTensors().get(tenId);
  input->insert(inIndex, ptensor);
  ptensor->consumers.increment(this);
}

void Op::connectInTensor(InIndex inIndex, TensorId tenId) {
  defaultConnectInTensor(inIndex, tenId);
}

void Op::connectOutTensor(OutIndex outIndex, TensorId tenId) {
  Tensor *ptensor = pir->getTensors().get(tenId);
  output->insert(outIndex, ptensor);
  ptensor->setProducer(this);
}

void Op::disconnectInTensor(InIndex inIndex, Tensor *tensor) {
  tensor->consumers.decrement(this);

  input->erase(inIndex);
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
  pir->getTensors().addActGrad(tenId);
  Tensor *ptensor = pir->getTensors().get(tenId);
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

bool Op::hasInplaceVariant(InIndex) const { return false; }

bool Op::hasInplaceVariant(const std::vector<InIndex> &) const { return false; }

std::unique_ptr<Op> Op::getInplaceVariant(InIndex) {
  throw error("Op {} cannot get an inplace Op", opid);
}

std::unique_ptr<Op> Op::getInplaceVariant(const std::vector<InIndex> &) {
  throw error("Op {} cannot get an inplace Op", opid);
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
  if (!_name.empty()) {
    ss << '"' << _name << "\", ";
  }
  ss << id << " of type " << opid << '\n';

  int max_id_length = std::max(input->maxIdLength(), output->maxIdLength());

  ss << tab << "inputs" << '\n';
  input->append(ss, tab + tab, max_id_length);

  ss << '\n' << tab << "outputs" << '\n';
  output->append(ss, tab + tab, max_id_length);

  ss << '\n' << tab << "attributes" << '\n';
  nAtts.append(ss, tab + tab);
}

const std::string &Op::name() const { return _name; }

Op::Op(const Op &op)
    : Vertex(op), input(new TensorIndexMap), output(new TensorIndexMap),
      priority(op.priority), pir(op.pir), id(pir->getAndIncrOpsCounter()),
      opid(op.opid), nAtts(op.nAtts), _name(op._name) {
  // input, output: empty.
}

Op::Op(const OperatorIdentifier &_opid,
       Ir *_ir,
       const std::string &_name_,
       const Attributes &_attributes)
    : input(new TensorIndexMap), output(new TensorIndexMap), priority(0.0),
      // the Ir
      pir(_ir),
      // the id
      id(_ir->getAndIncrOpsCounter()), opid(_opid),
      // the Attributes
      nAtts(_attributes), _name(_name_) {}

int Ir::getOpSetVersionFromModel(const std::string &node_domain) {

  // If the node.domain is blank it means the default ai.onnx
  auto domain = node_domain;
  if (domain == "") {
    domain = Domain::ai_onnx;
  }

  // Get the version of the opset from the model based on the domain
  int version    = 0;
  auto opsetList = getModel().opset_import();
  for (auto &opset : opsetList) {

    std::string opset_domain;
    if (opset.has_domain() == false || opset.domain() == "") {
      opset_domain = Domain::ai_onnx;
    } else {
      opset_domain = opset.domain();
    }

    if (domain == opset_domain) {

      auto opset_version = static_cast<int>(opset.version());

      // If the same domain is mentioned multiple times find the largest
      if (opset_version > version)
        version = opset_version;
    }
  }

  // If the version has not be set throw an exception
  if (version == 0) {
    throw error("No opset version defined for domain \'{}\'", domain);
  }

  return version;
}

std::unique_ptr<Op> Ir::addOp(const Node &node) {

  int version = getOpSetVersionFromModel(node.domain());

  std::unique_ptr<Op> p = OpManager::createOp(node.domain(),
                                              node.op_type(),
                                              version,
                                              this,
                                              node.name(),
                                              node.attribute());
  if (p != nullptr)
    return p;
  else {
    if (node.op_type() == Onnx::AiOnnx::OpSet9::Constant.type) {
      throw error("ILE. Constant Ops are not to be added");
    } else {
      throw error("No class for {}.{}:{}",
                  (node.domain() == "" ? Domain::ai_onnx : node.domain()),
                  node.op_type(),
                  version);
    }
  }
}

std::string Op::str() const {
  std::stringstream ss;
  ss << id << "(" << opid << ")";
  return ss.str();
}

std::string Op::debugName() const {
  std::string debug_id;
  if (!_name.empty()) {
    debug_id = _name;
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

} // namespace poponnx
