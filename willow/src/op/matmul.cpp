#include <poponnx/error.hpp>
#include <poponnx/op/matmul.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/util.hpp>

namespace poponnx {

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

const Tensor *MatMulOp::lhsIn() const { return inTensor(getLhsInIndex()); }

const Tensor *MatMulOp::rhsIn() const { return inTensor(getRhsInIndex()); }

const Tensor *MatMulOp::out() const { return outTensor(getOutIndex()); }

std::vector<int64_t> MatMulOp::lhsBroadcastShape() const {
  const Tensor *lhs = lhsIn();
  const Tensor *rhs = rhsIn();

  return MatMulOp::lhsNpBroadcastShape(lhs->info.shape(), rhs->info.shape());
}

std::vector<int64_t> MatMulOp::rhsBroadcastShape() const {
  const Tensor *lhs = lhsIn();
  const Tensor *rhs = rhsIn();

  return MatMulOp::rhsNpBroadcastShape(lhs->info.shape(), rhs->info.shape());
}

Shape MatMulOp::lhsNpBroadcastShape(Shape lhs, Shape rhs) {
  if (lhs.empty() || rhs.empty()) {
    throw error("MatMul op doesn't support scalars");
  }

  Shape result = MatMulOp::npMatMulOut(lhs, rhs);
  std::copy(lhs.end() - 2, lhs.end(), result.end() - 2);

  return result;
}

Shape MatMulOp::rhsNpBroadcastShape(Shape lhs, Shape rhs) {
  if (lhs.empty() || rhs.empty()) {
    throw error("MatMul op doesn't support scalars");
  }

  Shape result = MatMulOp::npMatMulOut(lhs, rhs);
  std::copy(rhs.end() - 2, rhs.end(), result.end() - 2);

  return result;
}

Shape MatMulOp::npMatMulOut(Shape lhs, Shape rhs) {
  if (lhs.empty() || rhs.empty()) {
    throw error("MatMul op doesn't support scalars");
  }

  const bool lhs_prepend = lhs.size() == 1;
  const bool rhs_append  = rhs.size() == 1;

  // If the first argument is 1-D, it is promoted to a matrix by prepending a 1
  // to its dimensions.
  if (lhs_prepend) {
    lhs.insert(lhs.begin(), 1);
  }

  // If the second argument is 1-D, it is promoted to a matrix by appending a 1
  // to its dimensions
  if (rhs_append) {
    rhs.push_back(1);
  }

  Shape result =
      npOut({lhs.begin(), lhs.end() - 2}, {rhs.begin(), rhs.end() - 2});

  // After matrix multiplication the prepended 1 is removed.
  // We implement this by not adding it.
  if (!lhs_prepend) {
    result.push_back(lhs[lhs.size() - 2]);
  }

  // After matrix multiplication the appended 1 is removed.
  // We implement this by not adding it.
  if (!rhs_append) {
    result.push_back(rhs[rhs.size() - 1]);
  }

  if (lhs[lhs.size() - 1] != rhs[rhs.size() - 2]) {
    throw error("MatMulOp mismatched input sizes");
  }

  return result;
}

void MatMulOp::setup() {
  // Define the shape of the output tensor
  output.tensor(0)->info = {
      lhsIn()->info.dataType(),
      MatMulOp::npMatMulOut(lhsBroadcastShape(), rhsBroadcastShape())};
}

MatMulLhsGradOp::MatMulLhsGradOp(const MatMulOp &fwdOp)
    : Op({"MatMulLhsGrad", fwdOp.pir, {}, getPoponnxDomain()}),
      fwdOpOutputGrad(fwdOp.output.tensor(0)->info),
      fwdOpLhsInfo(fwdOp.lhsIn()->info), fwdOpRhsInfo(fwdOp.rhsIn()->info) {}

void MatMulLhsGradOp::setup() { output.tensor(0)->info = fwdOpLhsInfo; }

const std::vector<GradInOutMapper> &MatMulLhsGradOp::gradInputInfo() const {
  // The gradient of the fwd-op is input at index 0.
  // The index at which the rhs tensor is the input to the grad-op
  // is the same as the index at which it the input to the fwd-op
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), MatMulOp::getOutIndex(), GradOpInType::GRADOUT},
      {getRhsInIndex(), MatMulOp::getRhsInIndex(), GradOpInType::IN}};
  return inInfo;
}

const std::map<int, int> &MatMulLhsGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {
      {getOutIndex(), MatMulOp::getLhsInIndex()}};
  return outInfo;
}

Shape MatMulLhsGradOp::getGradInputShape() const {
  return fwdOpOutputGrad.shape();
}

Shape MatMulLhsGradOp::getRhsInputShape() const { return fwdOpRhsInfo.shape(); }

Shape MatMulLhsGradOp::getOutputShape() const { return fwdOpLhsInfo.shape(); }

MatMulRhsGradOp::MatMulRhsGradOp(const MatMulOp &fwdOp)
    : Op({"MatMulRhsGrad", fwdOp.pir, {}, getPoponnxDomain()}),
      fwdOpOutputGrad(fwdOp.output.tensor(0)->info),
      fwdOpLhsInfo(fwdOp.lhsIn()->info), fwdOpRhsInfo(fwdOp.rhsIn()->info) {}

void MatMulRhsGradOp::setup() { output.tensor(0)->info = fwdOpRhsInfo; }

const std::vector<GradInOutMapper> &MatMulRhsGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), MatMulOp::getOutIndex(), GradOpInType::GRADOUT},
      {getLhsInIndex(), MatMulOp::getLhsInIndex(), GradOpInType::IN}};
  return inInfo;
}

const std::map<int, int> &MatMulRhsGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 1
  static const std::map<int, int> outInfo = {
      {getOutIndex(), MatMulOp::getRhsInIndex()}};
  return outInfo;
}

Shape MatMulRhsGradOp::getGradInputShape() const {
  return fwdOpOutputGrad.shape();
}

Shape MatMulRhsGradOp::getLhsInputShape() const { return fwdOpLhsInfo.shape(); }

Shape MatMulRhsGradOp::getOutputShape() const { return fwdOpRhsInfo.shape(); }

} // namespace poponnx
