#include <onnx/onnx_pb.h>
#include <poponnx/ir.hpp>
#include <poponnx/tensor.hpp>

// The layers:
#include <poponnx/op/add.hpp>
#include <poponnx/op/averagepool.hpp>
#include <poponnx/op/batchnorm.hpp>
#include <poponnx/op/conv.hpp>
#include <poponnx/op/cos.hpp>
#include <poponnx/op/cosh.hpp>
#include <poponnx/op/div.hpp>
#include <poponnx/op/exp.hpp>
#include <poponnx/op/gemm.hpp>
#include <poponnx/op/identity.hpp>
#include <poponnx/op/matmul.hpp>
#include <poponnx/op/maxpool.hpp>
#include <poponnx/op/mul.hpp>
#include <poponnx/op/negate.hpp>
#include <poponnx/op/pad.hpp>
#include <poponnx/op/reciprocal.hpp>
#include <poponnx/op/reducesum.hpp>
#include <poponnx/op/relu.hpp>
#include <poponnx/op/sigmoid.hpp>
#include <poponnx/op/sin.hpp>
#include <poponnx/op/softmax.hpp>
#include <poponnx/op/sqrt.hpp>
#include <poponnx/op/square.hpp>
#include <poponnx/op/squeeze.hpp>
#include <poponnx/op/subsample.hpp>
#include <poponnx/op/subtract.hpp>
#include <poponnx/op/sum.hpp>
#include <poponnx/op/tan.hpp>
#include <poponnx/op/tanh.hpp>
#include <poponnx/op/transpose.hpp>
#include <poponnx/op/varupdate.hpp>

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
  throw error("No clone implemented for " + op_type());
}

Op::~Op() = default;

// return a vector of 1 or several OpAndTensorIds for
// obtaining the gradient of the inputs of this Op.
// The Op in the OpAndTensorIds is the gradient op, and
// the TensorIds are the input indices of input of this
// Op for which the gradient is computed
std::vector<std::unique_ptr<Op>> Op::getGradOps() {
  throw error("Cannot get gradients for " + op_type());
}

void Op::setup() { throw error("No setup() for " + op_type()); }

void Op::connectInTensor(InIndex inIndex, TensorId tenId) {
  Tensor *ptensor = pir->getTensors().get(tenId);
  input->insert(inIndex, ptensor);
  ptensor->consumers.increment(this);
}

void Op::connectOutTensor(OutIndex outIndex, TensorId tenId) {
  Tensor *ptensor = pir->getTensors().get(tenId);
  output->insert(outIndex, ptensor);
  ptensor->setProducer(this);
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
  throw error("Op " + op_type() + " cannot get `grad input info'");
}

const std::map<int, int> &Op::gradOutToNonGradIn() const {
  throw error("Op " + op_type() + " cannot get `grad out to non grad in'");
}

bool Op::hasInplaceVariant(InIndex) const { return false; }

std::unique_ptr<Op> Op::getInplaceVariant(InIndex) {
  throw error("Op " + op_type() + "cannot get an inplace Op");
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
  ss << '\n' << "Op " << id << " of type " << op_type() << '\n';
  ss << tab << "inputs" << '\n';

  int max_id_length = std::max(input->maxIdLength(), output->maxIdLength());
  input->append(ss, tab + tab, max_id_length);
  ss << '\n' << tab << "outputs" << '\n';
  output->append(ss, tab + tab, max_id_length);
}

const std::string &Op::domain() { return getOpTypes().getDomain(opType); }

const std::string &Op::op_type() const { return getOpTypes().getName(opType); }

const std::string &Op::name() const { return _name; }

OpConstructorBundle::OpConstructorBundle(OpType op_type_,
                                         Ir *pir_,
                                         Attributes atts_)
    : op_type(op_type_), pir(pir_), atts(atts_) {}

Op::Op(const Op &op)
    : Vertex(op), input(new TensorIndexMap), output(new TensorIndexMap),
      priority(op.priority), pir(op.pir), id(pir->getAndIncrOpsCounter()),
      opType(op.opType), nAtts(op.nAtts), _name(op._name) {
  // input, output: empty.
}

Op::Op(const OpConstructorBundle &b)
    : input(new TensorIndexMap), output(new TensorIndexMap), priority(0.0),
      // the Ir
      pir(b.pir),
      // the id
      id(pir->getAndIncrOpsCounter()),
      // opType
      opType(b.op_type),
      // the Attributes
      nAtts(b.atts) {}

Op::Op(const Node &node, Ir *pg)
    : input(new TensorIndexMap), output(new TensorIndexMap), priority(0.0),
      // pointer to the Ir containing this node
      pir(pg), id(pir->getAndIncrOpsCounter()),
      // We set opType, looked up in a map from the string node.op_type()
      opType(getOpTypes().get(node.op_type(), node.domain())),
      // poponnx::Attributes constructed from contained of onnx::Attribute
      nAtts(node.attribute()) {
  // Set the name if it has been specified on the node.
  if (node.has_name()) {
    _name = node.name();
  }
}

std::unique_ptr<Op> Ir::addOp(const Node &node) {
  using pOp = std::unique_ptr<Op>;
  switch (getOpTypes().get(node.op_type(), node.domain())) {
  case OpType::ADD: {
    return pOp(new AddOp(node, this));
  }
  case OpType::AVERAGEPOOL: {
    return pOp(new AveragePoolOp(node, this));
  }
  case OpType::BATCHNORM: {
    return pOp(new BatchNormOp(node, this));
  }
  case OpType::CONSTANT: {
    throw error("ILE. Constant Ops are not to be added");
  }
  case OpType::CONV: {
    return pOp(new ConvOp(node, this));
  }
  case OpType::COS: {
    return pOp(new CosOp(node, this));
  }
  case OpType::COSH: {
    return pOp(new CoshOp(node, this));
  }
  case OpType::DIV: {
    return pOp(new DivOp(node, this));
  }
  case OpType::EXP: {
    return pOp(new ExpOp(node, this));
  }
  case OpType::GEMM: {
    return pOp(new GemmOp(node, this));
  }
  case OpType::IDENTITY: {
    return pOp(new IdentityOp(node, this));
  }
  case OpType::NEGATE: {
    return pOp(new NegateOp(node, this));
  }
  case OpType::RECIPROCAL: {
    return pOp(new ReciprocalOp(node, this));
  }
  case OpType::SQRT: {
    return pOp(new SqrtOp(node, this));
  }
  case OpType::SQUARE: {
    return pOp(new SquareOp(node, this));
  }
  case OpType::SOFTMAX: {
    return pOp(new SoftmaxOp(node, this));
  }
  case OpType::MAXPOOL: {
    return pOp(new MaxPoolOp(node, this));
  }
  case OpType::MUL: {
    return pOp(new MulOp(node, this));
  }
  case OpType::PAD: {
    return pOp(new PadOp(node, this));
  }
  case OpType::REDUCESUM: {
    return pOp(new ReduceSumOp(node, this));
  }
  case OpType::RELU: {
    return pOp(new ReluOp(node, this));
  }
  case OpType::SIGMOID: {
    return pOp(new SigmoidOp(node, this));
  }
  case OpType::SIN: {
    return pOp(new SinOp(node, this));
  }
  case OpType::SUBTRACT: {
    return pOp(new SubtractOp(node, this));
  }
  case OpType::SUBSAMPLE: {
    return pOp(new SubsampleOp(node, this));
  }
  case OpType::SUM: {
    return pOp(new SumOp(node, this));
  }
  case OpType::SQUEEZE: {
    return pOp(new SqueezeOp(node, this));
  }
  case OpType::TAN: {
    return pOp(new TanOp(node, this));
  }
  case OpType::TANH: {
    return pOp(new TanhOp(node, this));
  }
  case OpType::MATMUL: {
    return pOp(new MatMulOp(node, this));
  }
  case OpType::TRANSPOSE: {
    return pOp(new TransposeOp(node, this));
  }
  case OpType::ADDARG0GRAD:
  case OpType::ADDARG1GRAD:
  case OpType::ADDBIASBIASGRAD:
  case OpType::ADDBIASDATAGRAD:
  case OpType::COSGRAD:
  case OpType::DIVARG0GRAD:
  case OpType::DIVARG1GRAD:
  case OpType::EXPGRAD:
  case OpType::SQUEEZEGRAD:
  case OpType::REDUCESUMGRAD:
  case OpType::RELUGRAD:
  case OpType::AVERAGEPOOLGRAD:
  case OpType::CONVDATAGRAD:
  case OpType::CONVWEIGHTSGRAD:
  case OpType::NEGATEGRAD:
  case OpType::IDENTITYGRAD:
  case OpType::NLLGRAD:
  case OpType::L1GRAD:
  case OpType::MAXPOOLGRAD:
  case OpType::MULARG0GRAD:
  case OpType::MULARG1GRAD:
  case OpType::RECIPROCALGRAD:
  case OpType::SIGMOIDGRAD:
  case OpType::SINGRAD:
  case OpType::SCALE:
  case OpType::SCALEGRAD:
  case OpType::SOFTMAXGRAD:
  case OpType::SGDVARUPDATE:
  case OpType::SQRTGRAD:
  case OpType::CONSTSGDVARUPDATE:
  case OpType::SUBTRACTARG0GRAD:
  case OpType::SUBTRACTARG1GRAD:
  case OpType::TANHGRAD:
  case OpType::SUBSAMPLEGRAD:
  case OpType::TRANSPOSEGRAD:
  case OpType::MATMULLHSGRAD:
  case OpType::MATMULRHSGRAD:
    throw error("Gradient Ops not constructable from Node");

  case OpType::NLL:
  case OpType::L1:
    throw error("Loss Ops not constructable from Node");

  case OpType::ADDBIAS:
  case OpType::RELUINPLACE:
  case OpType::SOFTMAXGRADDIRECT:
    throw error("Non-ONNX Ops not constructable from Node");

  default: { throw error("No class for " + node.op_type()); }
  }
}

std::string Op::str() const {
  return std::to_string(id) + " (" + op_type() + ')';
}

} // namespace poponnx
