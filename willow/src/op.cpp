#include <onnx/onnx_pb.h>
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
  ss << '\n' << "Op " << id << " of type " << opid << '\n';

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

std::unique_ptr<Op> Ir::addOp(const Node &node) {

  // For now we need to look at the domain and set the version. We really should
  // get the version from the model
  unsigned version = 9;
  if (node.domain() == Domain::ai_graphcore) {
    version = 1;
  }

  OperatorIdentifier opid(node.domain(), node.op_type(), version);

  std::unique_ptr<Op> p =
      OpManager::createOp(opid, this, node.name(), node.attribute());
  if (p != nullptr)
    return p;
  else {
    if (opid == Onnx::Operators::Constant) {
      throw error("ILE. Constant Ops are not to be added");
    } else {
      throw error("No class for {}", opid);
    }
  }
}

std::string Op::str() const {
  std::stringstream ss;
  ss << id << "(" << opid << ")";
  return ss.str();
}

} // namespace poponnx
