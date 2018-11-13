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
  // The gradient of the fwd-op is input at index 0.
  // The index at which the rhs tensor is the input to the grad-op
  // is the same as the index at which is the input to the fwd-op
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInputIndex(), 0, GradOpInType::GRADOUT},
      {getRhsInputIndex(), rhsInputIndex(), GradOpInType::IN}};
  return inInfo;
}

const std::map<int, int> &MatMulLhsGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {{0, lhsInputIndex()}};
  return outInfo;
}

int MatMulLhsGradOp::getGradInputIndex() const { return 0; }
int MatMulLhsGradOp::getRhsInputIndex() const { return 1; }

MatMulRhsGradOp::MatMulRhsGradOp(const MatMulOp &fwdOp)
    : Op({"MatMulRhsGrad", fwdOp.pir, {}, getWillowDomain()}),
      fwdOpOutputGrad(fwdOp.output.tensor(0)->info), lhs(fwdOp.lhsIn()->info) {}

void MatMulRhsGradOp::setup() {
  outputShape            = {lhs.dim(1), fwdOpOutputGrad.dim(1)};
  output.tensor(0)->info = {fwdOpOutputGrad.dataType(), outputShape};
}
const std::vector<GradInOutMapper> &MatMulRhsGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInputIndex(), 0, GradOpInType::GRADOUT},
      {getLhsInputIndex(), lhsInputIndex(), GradOpInType::IN}};
  return inInfo;
}

const std::map<int, int> &MatMulRhsGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 1
  static const std::map<int, int> outInfo = {{0, rhsInputIndex()}};
  return outInfo;
}

int MatMulRhsGradOp::getGradInputIndex() const { return 0; }
int MatMulRhsGradOp::getLhsInputIndex() const { return 1; }

} // namespace willow
