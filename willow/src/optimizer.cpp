#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/op/varupdate.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensornames.hpp>
#include <poponnx/tensors.hpp>

namespace poponnx {

Optimizer::~Optimizer()                 = default;
Optimizer::Optimizer()                  = default;
Optimizer::Optimizer(const Optimizer &) = default;

TensorId getLearningRateId() { return "learnRate"; }
TensorId getWeightDecayId() { return "weightDecay"; }

BaseSGD::BaseSGD(float lr, float wd) : learnRate_(lr), weightDecay_(wd) {}

float BaseSGD::learnRate() const { return learnRate_; }

float BaseSGD::weightDecay() const { return weightDecay_; }

SGD::SGD(float lr, float wd) : BaseSGD(lr, wd) {}

std::unique_ptr<Optimizer> SGD::clone() const {
  return std::unique_ptr<Optimizer>(new SGD(*this));
}

std::map<TensorId, TensorInfo> SGD::tensorInfos() const {
  return {{getLearningRateId(), {DataType::FLOAT, {}}},
          {getWeightDecayId(), {DataType::FLOAT, {}}}};
}

std::unique_ptr<Op> SGD::createOp(TensorId varId, Ir *pir) const {
  return std::unique_ptr<Op>(new SGDVarUpdateOp(varId, {*pir, ""}));
}

std::vector<TensorId> SGD::getInputIds(TensorId varId) const {
  std::vector<TensorId> inputs(4, "");
  inputs[VarUpdateOp::getVarInIndex()]            = varId;
  inputs[VarUpdateOp::getVarGradInIndex()]        = getGradId(varId);
  inputs[SGDVarUpdateOp::getLearnRateInIndex()]   = getLearningRateId();
  inputs[SGDVarUpdateOp::getWeightDecayInIndex()] = getWeightDecayId();
  return inputs;
}

void SGD::setTensorData(Tensor *t) const {
  if (t->id == getLearningRateId()) {
    float lRate = learnRate();
    t->setTensorData(t->info, &lRate);
  } else if (t->id == getWeightDecayId()) {
    // Note: scaling weightDecay scalar by learnRate on host
    // to allow for efficient implementation of weight update
    // on the device
    float wDecay = weightDecay() * learnRate();
    t->setTensorData(t->info, &wDecay);
  } else {
    throw error("SGD cannot set the parameter (" + t->id + ") currently");
  }
}

void SGD::resetTensorDatas(Ir *pir) const {
  Tensor *lrTensor = pir->getTensors().get(getLearningRateId());
  Tensor *wdTensor = pir->getTensors().get(getWeightDecayId());
  float lRate      = learnRate();
  // Note: scaling weightDecay scalar by learnRate on host
  // to allow for efficient implementation of weight update
  // on the device
  float wDecay = weightDecay() * learnRate();
  lrTensor->tensorData()->resetData(lrTensor->info, &lRate);
  wdTensor->tensorData()->resetData(wdTensor->info, &wDecay);
}

void ConstSGD::setTensorData(Tensor *) const {
  throw error("ILE : ConstSGD does not set tensor data");
}

void ConstSGD::resetTensorDatas(Ir *) const {
  throw error("ConstSGD does not have any tensors to reset");
}

std::unique_ptr<Op> ConstSGD::createOp(TensorId varId, Ir *pir) const {
  return std::unique_ptr<Op>(
      new ConstSGDVarUpdateOp(varId, learnRate(), weightDecay(), {*pir, ""}));
}

std::vector<TensorId> ConstSGD::getInputIds(TensorId varId) const {
  std::vector<TensorId> inputs(2, "");
  inputs[VarUpdateOp::getVarInIndex()]     = varId;
  inputs[VarUpdateOp::getVarGradInIndex()] = getGradId(varId);
  return inputs;
}

ConstSGD::ConstSGD(float lr, float wd) : BaseSGD(lr, wd) {}

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

} // namespace poponnx
