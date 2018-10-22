#include <willow/optimizer.hpp>
#include <willow/varupdate.hpp>

namespace willow {

TensorId getLearningRateId() { return "learnRate"; }

BaseSGD::BaseSGD(float lr) : learnRate_(lr) {}

float BaseSGD::learnRate() const { return learnRate_; }

SGD::SGD(float l) : BaseSGD(l) {}

std::unique_ptr<Optimizer> SGD::clone() const {
  return std::unique_ptr<Optimizer>(new SGD(*this));
}

std::map<TensorId, TensorInfo> SGD::tensorInfos() const {
  return {{getLearningRateId(), {TP::FLOAT, {}}}};
}

std::unique_ptr<Op> SGD::createOp(TensorId varId, Ir *pir) const {
  return std::unique_ptr<Op>(new SGDVarUpdateOp(varId, pir));
}

std::vector<TensorId> SGD::getInputIds(TensorId varId) const {
  std::vector<TensorId> inputs(3, "");
  inputs[VarUpdateOp::getVarIndex()]          = varId;
  inputs[VarUpdateOp::getVarGradIndex()]      = getGradId(varId);
  inputs[SGDVarUpdateOp::getLearnRateIndex()] = getLearningRateId();
  return inputs;
}

std::unique_ptr<Op> ConstSGD::createOp(TensorId varId, Ir *pir) const {
  return std::unique_ptr<Op>(new ConstSGDVarUpdateOp(varId, pir));
}

std::vector<TensorId> ConstSGD::getInputIds(TensorId varId) const {
  std::vector<TensorId> inputs(2, "");
  inputs[VarUpdateOp::getVarIndex()]     = varId;
  inputs[VarUpdateOp::getVarGradIndex()] = getGradId(varId);
  return inputs;
}

ConstSGD::ConstSGD(float l) : BaseSGD(l) {}

std::unique_ptr<Optimizer> ConstSGD::clone() const {
  return std::unique_ptr<Optimizer>(new ConstSGD(*this));
}

std::map<TensorId, TensorInfo> ConstSGD::tensorInfos() const { return {}; }

} // namespace willow
