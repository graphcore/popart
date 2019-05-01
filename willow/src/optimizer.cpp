#include <poponnx/error.hpp>
#include <poponnx/graph.hpp>
#include <poponnx/op/varupdate.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensornames.hpp>
#include <poponnx/tensors.hpp>

namespace poponnx {

Optimizer::~Optimizer()                 = default;
Optimizer::Optimizer()                  = default;
Optimizer::Optimizer(const Optimizer &) = default;

TensorId getLearningRateId(DataType dtype) {
  return "learnRate_" + getDataTypeInfoMap().at(dtype).name();
}
TensorId getWeightDecayId(DataType dtype) {
  return "weightDecay_" + getDataTypeInfoMap().at(dtype).name();
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

BaseSGD::BaseSGD(float lr, float wd) : learnRate_(lr), weightDecay_(wd) {}

float BaseSGD::learnRate() const { return learnRate_; }

float BaseSGD::weightDecay() const { return weightDecay_; }

SGD::SGD(float lr, float wd) : BaseSGD(lr, wd) {}

std::unique_ptr<Optimizer> SGD::clone() const {
  return std::unique_ptr<Optimizer>(new SGD(*this));
}

std::map<TensorId, TensorInfo> SGD::tensorInfos() const {
  return {{getLearningRateId(DataType::FLOAT), {DataType::FLOAT, {}}},
          {getLearningRateId(DataType::FLOAT16), {DataType::FLOAT16, {}}},
          {getWeightDecayId(DataType::FLOAT), {DataType::FLOAT, {}}},
          {getWeightDecayId(DataType::FLOAT16), {DataType::FLOAT16, {}}}};
}

std::unique_ptr<Op> SGD::createOp(TensorId varId, Graph &graph) const {
  return std::unique_ptr<Op>(new SGDVarUpdateOp(varId, {graph, ""}));
}

std::vector<TensorId> SGD::getInputIds(TensorId varId, DataType varType) const {
  std::vector<TensorId> inputs(4, "");
  inputs[VarUpdateOp::getVarInIndex()]            = varId;
  inputs[VarUpdateOp::getVarGradInIndex()]        = getGradId(varId);
  inputs[SGDVarUpdateOp::getLearnRateInIndex()]   = getLearningRateId(varType);
  inputs[SGDVarUpdateOp::getWeightDecayInIndex()] = getWeightDecayId(varType);
  return inputs;
}

void SGD::setTensorData(Tensor *t) const {
  if (t->id == getLearningRateId(DataType::FLOAT) ||
      t->id == getLearningRateId(DataType::FLOAT16)) {
    auto converted_data =
        convertFloatToDataType(t->info.dataType(), learnRate());
    t->setTensorData(t->info, converted_data.data());
  } else if (t->id == getWeightDecayId(DataType::FLOAT) ||
             t->id == getWeightDecayId(DataType::FLOAT16)) {
    // Note:
    // w <- w * (1 - lr * wd) - lr * delta
    //          ^^^^^^^^^^^^^
    // Calculating   this     term of the weight update formula
    // on host to allow for efficient implementation of weight
    // update on the device
    float weightDecayScaleFactor = 1 - (weightDecay() * learnRate());
    auto converted_data =
        convertFloatToDataType(t->info.dataType(), weightDecayScaleFactor);
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

  resetTensor(getLearningRateId(DataType::FLOAT), learnRate());
  resetTensor(getLearningRateId(DataType::FLOAT16), learnRate());

  // Note: scaling weightDecay scalar by learnRate on host
  // to allow for efficient implementation of weight update
  // on the device
  resetTensor(getWeightDecayId(DataType::FLOAT), weightDecay() * learnRate());
  resetTensor(getWeightDecayId(DataType::FLOAT16), weightDecay() * learnRate());
}

void ConstSGD::setTensorData(Tensor *) const {
  throw error("ILE : ConstSGD does not set tensor data");
}

void ConstSGD::resetTensorDatas(Graph &) const {
  throw error("ConstSGD does not have any tensors to reset");
}

std::unique_ptr<Op> ConstSGD::createOp(TensorId varId, Graph &graph) const {
  return std::unique_ptr<Op>(
      new ConstSGDVarUpdateOp(varId, learnRate(), weightDecay(), {graph, ""}));
}

std::vector<TensorId> ConstSGD::getInputIds(TensorId varId, DataType) const {
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
