#include <poponnx/error.hpp>
#include <poponnx/matmul.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/util.hpp>

namespace willow {

MatMulOp::MatMulOp(const onnx::NodeProto &node, Ir *_pir) : Op(node, _pir) {}

std::unique_ptr<Op> MatMulOp::clone() const {
  return make_unique<MatMulOp>(*this);
}

std::vector<std::unique_ptr<Op>> MatMulOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<MatMulLhsGradOp>(*this));
  upops.emplace_back(make_unique<MatMulRhsGradOp>(*this));
  return upops;
}

const Tensor *MatMulOp::lhsIn() const {
  return input.tensor(getLhsInputIndex());
}

const Tensor *MatMulOp::rhsIn() const {
  return input.tensor(getRhsInputIndex());
}

const Tensor *MatMulOp::out() const { return output.tensor(getOutputIndex()); }

void MatMulOp::setup() {

  const Tensor *lhs = lhsIn();
  const Tensor *rhs = rhsIn();

  // Assumption : The data type of the lhs & rhs are the same, as defined
  // in the ONNX spec.

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
    : Op({"MatMulLhsGrad", fwdOp.pir, {}, getPoponnxDomain()}),
      fwdOpOutputGrad(fwdOp.output.tensor(0)->info), rhs(fwdOp.rhsIn()->info) {}

void MatMulLhsGradOp::setup() {
  outputShape            = {fwdOpOutputGrad.dim(0), rhs.dim(0)};
  output.tensor(0)->info = {fwdOpOutputGrad.dataType(), outputShape};
}

const std::vector<GradInOutMapper> &MatMulLhsGradOp::gradInputInfo() const {
  // The gradient of the fwd-op is input at index 0.
  // The index at which the rhs tensor is the input to the grad-op
  // is the same as the index at which it the input to the fwd-op
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInputIndex(), MatMulOp::getOutputIndex(), GradOpInType::GRADOUT},
      {getRhsInputIndex(), MatMulOp::getRhsInputIndex(), GradOpInType::IN}};
  return inInfo;
}

const std::map<int, int> &MatMulLhsGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {{0, MatMulOp::getLhsInputIndex()}};
  return outInfo;
}

MatMulRhsGradOp::MatMulRhsGradOp(const MatMulOp &fwdOp)
    : Op({"MatMulRhsGrad", fwdOp.pir, {}, getPoponnxDomain()}),
      fwdOpOutputGrad(fwdOp.output.tensor(0)->info), lhs(fwdOp.lhsIn()->info) {}

void MatMulRhsGradOp::setup() {
  outputShape            = {lhs.dim(1), fwdOpOutputGrad.dim(1)};
  output.tensor(0)->info = {fwdOpOutputGrad.dataType(), outputShape};
}
const std::vector<GradInOutMapper> &MatMulRhsGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInputIndex(), MatMulOp::getOutputIndex(), GradOpInType::GRADOUT},
      {getLhsInputIndex(), MatMulOp::getLhsInputIndex(), GradOpInType::IN}};
  return inInfo;
}

const std::map<int, int> &MatMulRhsGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 1
  static const std::map<int, int> outInfo = {{0, MatMulOp::getRhsInputIndex()}};
  return outInfo;
}
} // namespace willow
