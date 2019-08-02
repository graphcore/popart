#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>

namespace popart {

Optimizer::~Optimizer()                 = default;
Optimizer::Optimizer()                  = default;
Optimizer::Optimizer(const Optimizer &) = default;

TensorId getScaledLearningRateId(DataType dtype) {
  return "scaledLearnRate_" + getDataTypeInfoMap().at(dtype).name();
}
TensorId getWeightDecayId(DataType dtype) {
  return "weightDecay_" + getDataTypeInfoMap().at(dtype).name();
}
TensorId getLossScalingId(DataType dtype) {
  return "lossScaling_" + getDataTypeInfoMap().at(dtype).name();
}

// convert a float to type T
template <typename T> std::vector<char> convertFloatTo(float data) {
  std::vector<char> data_out;
  T converted_data{data};
  data_out.resize(sizeof(T));
  *reinterpret_cast<T *>(data_out.data()) = converted_data;
  return data_out;
}

// convert a float to the DataType `dtype`
static std::vector<char> convertFloatToDataType(DataType dtype, float data) {
  if (dtype == DataType::FLOAT) {
    return convertFloatTo<float>(data);
  } else if (dtype == DataType::FLOAT16) {
    return convertFloatTo<Half>(data);
  } else {
    throw error("Can't convert float to DataType {}",
                getDataTypeInfoMap().at(dtype).name());
  }
}

BaseSGD::BaseSGD(float lr, float wd, float ls)
    : learnRate_(lr), weightDecay_(wd), lossScaling_(ls) {

  // Reject loss scaling of 0.
  if (!(lossScaling_ > 0.0f || lossScaling_ < 0.0f)) {
    throw error("Loss scaling can not be 0");
  }
}

float BaseSGD::learnRate() const { return learnRate_; }

float BaseSGD::weightDecay() const { return weightDecay_; }

float BaseSGD::lossScaling() const { return lossScaling_; }

float BaseSGD::weightDecayScaleFactor() const {
  return 1 - (weightDecay() * (scaledLearningRate()));
}
float BaseSGD::scaledLearningRate() const {
  return learnRate() / lossScaling();
}

SGD::SGD(float lr, float wd, float ls) : BaseSGD(lr, wd, ls) {}

std::unique_ptr<Optimizer> SGD::clone() const {
  return std::unique_ptr<Optimizer>(new SGD(*this));
}

std::map<TensorId, TensorInfo> SGD::tensorInfos() const {
  return {{getScaledLearningRateId(DataType::FLOAT), {DataType::FLOAT, {}}},
          {getScaledLearningRateId(DataType::FLOAT16), {DataType::FLOAT16, {}}},
          {getWeightDecayId(DataType::FLOAT), {DataType::FLOAT, {}}},
          {getWeightDecayId(DataType::FLOAT16), {DataType::FLOAT16, {}}},
          {getLossScalingId(DataType::FLOAT), {DataType::FLOAT, {}}},
          {getLossScalingId(DataType::FLOAT16), {DataType::FLOAT16, {}}}};
}

std::unique_ptr<Op> SGD::createOp(TensorId varId, Graph &graph) const {
  return std::unique_ptr<Op>(new SGDVarUpdateOp(varId, {graph, ""}));
}

std::vector<TensorId> SGD::getInputIds(TensorId varId, DataType varType) const {
  std::vector<TensorId> inputs(5, "");
  inputs[VarUpdateOp::getVarToUpdateInIndex()] = varId;
  inputs[VarUpdateOp::getUpdaterInIndex()]     = getGradId(varId);
  inputs[SGDVarUpdateOp::getScaledLearnRateInIndex()] =
      getScaledLearningRateId(varType);
  inputs[SGDVarUpdateOp::getWeightDecayInIndex()] = getWeightDecayId(varType);
  inputs[SGDVarUpdateOp::getLossScalingInIndex()] = getLossScalingId(varType);
  return inputs;
}

TensorId Optimizer::getLossScalingTensorId(DataType varType) const {
  return getLossScalingId(varType);
}

void SGD::setTensorData(Tensor *t) const {
  if (t->id == getScaledLearningRateId(DataType::FLOAT) ||
      t->id == getScaledLearningRateId(DataType::FLOAT16)) {
    // Note:
    // w <- w * (1 - (lr/ls) * wd) - (lr/ls) * delta
    //                                ^^^^^
    // Calculate this term of the weight update formula
    // on host to allow for efficient implementation of weight
    // update on the device
    auto converted_data =
        convertFloatToDataType(t->info.dataType(), scaledLearningRate());
    t->setTensorData(t->info, converted_data.data());
  } else if (t->id == getWeightDecayId(DataType::FLOAT) ||
             t->id == getWeightDecayId(DataType::FLOAT16)) {
    // Note:
    // w <- w * (1 - (lr/ls) * wd) - (lr/ls) * delta
    //          ^^^^^^^^^^^^^^^^^
    // Calculating   this     term of the weight update formula
    // on host to allow for efficient implementation of weight
    // update on the device
    auto converted_data =
        convertFloatToDataType(t->info.dataType(), weightDecayScaleFactor());
    t->setTensorData(t->info, converted_data.data());
  } else if (t->id == getLossScalingId(DataType::FLOAT) ||
             t->id == getLossScalingId(DataType::FLOAT16)) {
    auto converted_data =
        convertFloatToDataType(t->info.dataType(), lossScaling());
    t->setTensorData(t->info, converted_data.data());
  } else {
    throw error("SGD cannot set the parameter (" + t->id + ") currently");
  }
}

void SGD::resetTensorDatas(Graph &graph) const {
  // Check the tensor exists before resetting data
  auto resetTensor = [&](TensorId id, float data) {
    auto &tensors = graph.getTensors();
    if (tensors.contains(id)) {
      auto t              = tensors.get(id);
      auto converted_data = convertFloatToDataType(t->info.dataType(), data);
      t->tensorData()->resetData(t->info, converted_data.data());
    }
  };

  resetTensor(getScaledLearningRateId(DataType::FLOAT), scaledLearningRate());
  resetTensor(getScaledLearningRateId(DataType::FLOAT16), scaledLearningRate());

  resetTensor(getLossScalingId(DataType::FLOAT), lossScaling());
  resetTensor(getLossScalingId(DataType::FLOAT16), lossScaling());

  // Note: scaling weightDecay scalar by learnRate on host
  // to allow for efficient implementation of weight update
  // on the device
  resetTensor(getWeightDecayId(DataType::FLOAT), weightDecayScaleFactor());
  resetTensor(getWeightDecayId(DataType::FLOAT16), weightDecayScaleFactor());
}

void ConstSGD::setTensorData(Tensor *) const {
  throw error("ILE : ConstSGD does not set tensor data");
}

void ConstSGD::resetTensorDatas(Graph &) const {
  throw error("ConstSGD does not have any tensors to reset");
}

std::unique_ptr<Op> ConstSGD::createOp(TensorId varId, Graph &graph) const {
  return std::unique_ptr<Op>(new ConstSGDVarUpdateOp(
      varId, learnRate(), weightDecay(), lossScaling(), {graph, ""}));
}

std::vector<TensorId> ConstSGD::getInputIds(TensorId varId, DataType) const {
  std::vector<TensorId> inputs(2, "");
  inputs[VarUpdateOp::getVarToUpdateInIndex()] = varId;
  inputs[VarUpdateOp::getUpdaterInIndex()]     = getGradId(varId);
  return inputs;
}

ConstSGD::ConstSGD(float lr, float wd, float ls) : BaseSGD(lr, wd, ls) {}

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

} // namespace popart
