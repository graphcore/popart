#include <poponnx/error.hpp>
#include <poponnx/matmul.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/util.hpp>

namespace willow {

constexpr unsigned lhsInputIndex() { return 0; }
constexpr unsigned rhsInputIndex() { return 1; }

MatMulOp::MatMulOp(const onnx::NodeProto &node, Ir *pir) : Op(node, pir) {}

std::unique_ptr<Op> MatMulOp::clone() const {
  return make_unique<MatMulOp>(*this);
}

std::vector<std::unique_ptr<Op>> MatMulOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<MatMulLhsGradOp>(*this));
  upops.emplace_back(make_unique<MatMulRhsGradOp>(*this));
  return upops;
}

const Tensor *MatMulOp::lhsIn() const { return input.tensor(lhsInputIndex()); }

const Tensor *MatMulOp::rhsIn() const { return input.tensor(rhsInputIndex()); }

void MatMulOp::setup() {

  const Tensor *lhs = lhsIn();
  const Tensor *rhs = rhsIn();

  // TODO : Assume data type does not change

  if (lhs->info.rank() != 2) {
    std::stringstream ss;
    ss << "MatMulOp only supports input of rank 2. Input 0 " << lhs->id << ":"
       << lhs->info << " has rank " << lhs->info.rank();
    lhs->info.append(ss);
    throw error(ss.str());
  }

  if (rhs->info.rank() != 2) {
    std::stringstream ss;
    ss << "MatMulOp only supports input of rank 2. Input 1 " << rhs->id << ":"
       << rhs->info << " has rank " << rhs->info.rank();
    rhs->info.append(ss);
    throw error(ss.str());
  }

  if (lhs->info.dim(1) != rhs->info.dim(0)) {
    std::stringstream ss;
    ss << "MapMulOp mismatch input sizes " << lhs->id << ":" << lhs->info
       << ",  " << rhs->id << ":" << rhs->info;
    throw error(ss.str());
  }

  // Define the shape of the output tensor
  outputShape            = {lhs->info.dim(0), rhs->info.dim(1)};
  output.tensor(0)->info = {lhs->info.dataType(), outputShape};
}

MatMulLhsGradOp::MatMulLhsGradOp(const MatMulOp &fwdOp)
    : Op({"MatMulLhsGrad", fwdOp.pir, {}, getWillowDomain()}),
      fwdOpOutputGrad(fwdOp.output.tensor(0)->info), rhs(fwdOp.rhsIn()->info) {}

void MatMulLhsGradOp::setup() {
  outputShape            = {fwdOpOutputGrad.dim(0), rhs.dim(0)};
  output.tensor(0)->info = {fwdOpOutputGrad.dataType(), outputShape};
}

const std::vector<GradInOutMapper> &MatMulLhsGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = createMatMulLhsGradInfo();
  return inInfo;
}

const std::map<int, int> &MatMulLhsGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = createMatMulLhsGradOutToIn();
  return outInfo;
}

int MatMulLhsGradOp::getGradInputIndex() const { return 0; }
int MatMulLhsGradOp::getRhsInputIndex() const { return 1; }

std::vector<GradInOutMapper> MatMulLhsGradOp::createMatMulLhsGradInfo() const {
  return {{getGradInputIndex(), 0, GradOpInType::GRADOUT},
          {getRhsInputIndex(), rhsInputIndex(), GradOpInType::IN}};
}

std::map<int, int> MatMulLhsGradOp::createMatMulLhsGradOutToIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 0
  return {{0, lhsInputIndex()}};
}

MatMulRhsGradOp::MatMulRhsGradOp(const MatMulOp &fwdOp)
    : Op({"MatMulRhsGrad", fwdOp.pir, {}, getWillowDomain()}),
      fwdOpOutputGrad(fwdOp.output.tensor(0)->info), lhs(fwdOp.lhsIn()->info) {}

void MatMulRhsGradOp::setup() {
  outputShape            = {lhs.dim(1), fwdOpOutputGrad.dim(1)};
  output.tensor(0)->info = {fwdOpOutputGrad.dataType(), outputShape};
}
const std::vector<GradInOutMapper> &MatMulRhsGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = createMatMulRhsGradInfo();
  return inInfo;
}

const std::map<int, int> &MatMulRhsGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = createMatMulRhsGradOutToIn();
  return outInfo;
}

int MatMulRhsGradOp::getGradInputIndex() const { return 0; }
int MatMulRhsGradOp::getLhsInputIndex() const { return 1; }

std::vector<GradInOutMapper> MatMulRhsGradOp::createMatMulRhsGradInfo() const {
  return {{getGradInputIndex(), 0, GradOpInType::GRADOUT},
          {getLhsInputIndex(), lhsInputIndex(), GradOpInType::IN}};
}

std::map<int, int> MatMulRhsGradOp::createMatMulRhsGradOutToIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 1
  return {{0, rhsInputIndex()}};
}
} // namespace willow
