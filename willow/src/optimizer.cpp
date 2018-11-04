#include <willow/error.hpp>
#include <willow/ir.hpp>
#include <willow/optimizer.hpp>
#include <willow/tensor.hpp>
#include <willow/varupdate.hpp>

namespace willow {

Optimizer::~Optimizer()                 = default;
Optimizer::Optimizer()                  = default;
Optimizer::Optimizer(const Optimizer &) = default;

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

void SGD::setTensorData(Tensor *t) const {
  if (t->id == getLearningRateId()) {
    float lRate = learnRate();
    t->setTensorData(t->info, &lRate);
  } else {
    throw error("SGD can only set the learning rate (" + getLearningRateId() +
                ") and not (" + t->id + ") currently");
  }
}

void SGD::resetTensorDatas(Ir *) const {
  throw error("Request to reset tensor datas for SGD, not implemented");
}

void ConstSGD::setTensorData(Tensor *) const {
  throw error("ILE : ConstSGD does not set tensor data");
}

void ConstSGD::resetTensorDatas(Ir *) const {
  throw error("ConstSGD does not have any tensors to reset");
}

std::unique_ptr<Op> ConstSGD::createOp(TensorId varId, Ir *pir) const {
  return std::unique_ptr<Op>(new ConstSGDVarUpdateOp(varId, pir, learnRate()));
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

bool SGD::validReplacement(const Optimizer *other) const {
  if (other->type() != type()) {
    return false;
  }
  // until we have momentum option, returning true
  return true;
}

OptimizerType SGD::type() const { return OptimizerType::SGD; }

std::string SGD::type_s() const { return "SGD"; }

// ConstSGD can never be replaced
bool ConstSGD::validReplacement(const Optimizer *) const { return false; }

OptimizerType ConstSGD::type() const { return OptimizerType::CONSTSGD; }

std::string ConstSGD::type_s() const { return "ConstSGD"; }

} // namespace willow
