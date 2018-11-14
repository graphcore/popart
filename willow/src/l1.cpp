#include <sstream>
#include <poponnx/error.hpp>
#include <poponnx/l1.hpp>
#include <poponnx/tensor.hpp>

namespace willow {

std::unique_ptr<Op> L1Op::clone() const {
  return std::unique_ptr<Op>(new L1Op(*this));
}

std::vector<std::unique_ptr<Op>> L1Op::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::unique_ptr<Op>(new L1GradOp(this)));
  return upops;
}

std::unique_ptr<Op> L1Loss::getOp(Ir *gp) const {
  return std::unique_ptr<Op>(
      new L1Op({op_type(), gp, {}, getWillowDomain()}, this));
}

std::string L1Loss::op_type() const { return "L1"; }

std::vector<TensorId> L1Loss::getStreamTensorNames() const { return {}; }

L1Loss::L1Loss(TensorId in_, TensorId out_, float lmb)
    : Loss({in_}, out_), lambda(lmb) {}

TensorId L1Loss::getInputId() const { return input(0); }

float L1Loss::getLambda() const { return lambda; }

const L1Loss *L1Op::l1l() const { return l1loss_; }
const L1Loss *L1GradOp::l1l() const { return l1loss_; }

L1Op::L1Op(const OpConstructorBundle &b, const L1Loss *n) : Op(b), l1loss_(n) {}

void L1GradOp::setup() {
  // gradient of input has same shape as input to L1
  output.tensor(0)->info = input.tensor(0)->info;
}

void L1Op::setup() {
  // output is a vector of length=batchsize, of the same type as input
  TensorInfo info0 = input.tensor(0)->info;
  if (info0.rank() == 0) {
    throw error("L1Op not valid for rank-0 tensor (scalar)");
  }
  int64_t batchsize = info0.dim(0);
  output.tensor(0)->info.set(input.tensor(0)->info.dataType(), {batchsize});
}

L1GradOp::L1GradOp(L1Op *op_)
    : Op({"L1Grad", op_->pir, {}, getWillowDomain()}), l1loss_(op_->l1l()) {}

const std::vector<GradInOutMapper> &L1GradOp::gradInputInfo() const {
  // input at index 0 of this grad op is the input at index 0 of the L1
  // non-grad op.
  static const std::vector<GradInOutMapper> inInfo = {{0, 0, GradOpInType::IN}};
  return inInfo;
}

const std::map<int, int> &L1GradOp::gradOutToNonGradIn() const {
  // grad-op's (only) output corresponds to op's (only) input.
  static const std::map<int, int> outInfo = {{0, 0}};
  return outInfo;
}

} // namespace willow
