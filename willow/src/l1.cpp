#include <sstream>
#include <willow/error.hpp>
#include <willow/l1.hpp>
#include <willow/tensor.hpp>

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

L1Loss::L1Loss(TensorId input_, TensorId output_, float lmb)
    : Loss({input_}, output_), lambda(lmb) {}

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
  static const std::vector<GradInOutMapper> inInfo = createL1LossGradInfo();
  return inInfo;
}

std::map<int, int> L1GradOp::createL1LossGradOutToIn() const {
  // grad-ops (only) out corresponds to ops (only) in.
  return {{0, 0}};
}

const std::map<int, int> &L1GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = createL1LossGradOutToIn();
  return outInfo;
}

std::vector<GradInOutMapper> L1GradOp::createL1LossGradInfo() const {
  // input at index 0 of grad in input at index 0 of non-grad
  // we will need to use lambda
  return {{0, 0, GradOpInType::IN}};
}

} // namespace willow
