#include <neuralnet/error.hpp>
#include <neuralnet/l1.hpp>
#include <neuralnet/tensor.hpp>
#include <sstream>

namespace neuralnet {

std::vector<std::unique_ptr<Op>> L1Op::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::unique_ptr<Op>(new L1GradOp(this)));
  return upops;
}

std::unique_ptr<Op> L1Loss::getOp(Graph *gp) const {
  return std::unique_ptr<Op>(
      new L1Op({op_type(), gp, {}, getNeuralNetDomain()}, this));
}

std::string L1Loss::op_type() const { return "L1"; }

std::vector<TensorId> L1Loss::getStreamTensorNames() const { return {}; }

L1Loss::L1Loss(const std::string &argstring) : Loss(argstring) {
  // expecting 1 input, 1 arg (lambda)
  confirmSizes(1, 1);
  lambda = std::stof(args()[0]);
}

const L1Loss *L1Op::l1l() const { return l1loss_; }
const L1Loss *L1GradOp::l1l() const { return l1loss_; }

L1Op::L1Op(const OpConstructorBundle &b, const L1Loss *n) : Op(b), l1loss_(n) {}

void L1GradOp::setup() {
  // gradient of input has same shape as input to L1
  output.tensor(0)->info = input.tensor(0)->info;
}

void L1Op::setup() {
  // output is a scalar of the same type as input
  output.tensor(0)->info.set(input.tensor(0)->info.dataType(), {});
}

L1GradOp::L1GradOp(L1Op *op_)
    : GradOp({"L1Grad", op_->pgraph, {}, getNeuralNetDomain()}),
      l1OpId(op_->id), l1loss_(op_->l1l()) {}

Op *L1GradOp::getNonGradOp() const {
  // we have chosen to go via the ID, rather
  // than storing the raw pointer, as it
  // is common for loss ops to be pruned
  // off while loss grad ops remain
  return pgraph->getOp(l1OpId);
}

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

} // namespace neuralnet
