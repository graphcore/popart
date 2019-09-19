#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>

namespace popart {

namespace {
float getWeightDecayScaleFactor(float wd, float lr) { return 1.0f - lr * wd; }
} // namespace

bool SGD::weightDecayScaleFactorIsConst(const Tensor &weight) const {
  auto wd = wds.get(weight.id);
  auto lr = lrs.get(weight.id);
  return wd.isConst() && lr.isConst();
}

float SGD::weightDecayScaleFactorVal(const Tensor &weight) const {
  auto wd = wds.get(weight.id).val();
  auto lr = lrs.get(weight.id).val();
  return getWeightDecayScaleFactor(wd, lr);
}

OptimizerValue SGD::weightDecayScaleFactor(const Tensor &weight) const {
  return {weightDecayScaleFactorVal(weight),
          weightDecayScaleFactorIsConst(weight)};
}

namespace {
float getScaledLearningRate(float lr, float ls) { return lr / ls; }
} // namespace

bool SGD::scaledLearningRateIsConst(const Tensor &weight) const {
  auto lr = lrs.get(weight.id);
  return lr.isConst() && lossScaling().isConst();
}

float SGD::scaledLearningRateVal(const Tensor &weight) const {
  auto lr = lrs.get(weight.id).val();
  return getScaledLearningRate(lr, lossScaling().val());
}

OptimizerValue SGD::scaledLearningRate(const Tensor &weight) const {
  return {scaledLearningRateVal(weight), scaledLearningRateIsConst(weight)};
}

void OptimizerValueMap::insertSpecific(const TensorId &id, OptimizerValue ov) {
  auto found = specifics.find(id);
  if (found != specifics.end()) {
    std::ostringstream oss;
    oss << "Attempt to insert specific value for optimization Tensor " << id
        << "failed as there is already a specific value for " << id
        << " present. Bailing, in case this is an error.";
    throw error(oss.str());
  }
  specifics.insert({id, ov});
}

void SGD::insertSpecific(const TensorId &id,
                         OptimizerValue wd,
                         OptimizerValue lr) {
  wds.insertSpecific(id, wd);
  lrs.insertSpecific(id, lr);
}

OptimizerValue OptimizerValueMap::get(const TensorId &id) const {
  auto found = specifics.find(id);
  if (found != specifics.end()) {
    return found->second;
  }
  return global;
};

TensorId SGD::getScaledLearningRateId(const Tensor &t) const {
  if (lrs.hasSpecific(t.id)) {
    return reservedSpecificScaledLearningRatePrefix() + t.id;
  }
  return reservedGlobalScaledLearningRatePrefix() + t.info.data_type();
}

TensorId SGD::getWeightDecayScaleFactorId(const Tensor &t) const {
  if (lrs.hasSpecific(t.id)) {
    return reservedSpecificWeightDecayScaleFactorPrefix() + t.id;
  }
  return reservedGlobalWeightDecayScaleFactorPrefix() + t.info.data_type();
}

TensorId Optimizer::getLossScalingTensorId(DataType t) const {
  return reservedLossScalingPrefix() + getDataTypeInfoMap().at(t).name();
}

std::string
SGD::stripWeightIdFromSpecificLearningRate(const TensorId &id) const {
  return std::string(
      id.begin() +
          std::string(reservedSpecificScaledLearningRatePrefix()).size(),
      id.end());
}

std::string
SGD::stripWeightIdFromSpecificWeightDecay(const TensorId &id) const {
  return std::string(
      id.begin() +
          std::string(reservedSpecificWeightDecayScaleFactorPrefix()).size(),
      id.end());
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
  }

  else if (dtype == DataType::FLOAT16) {
    return convertFloatTo<Half>(data);
  }

  throw error("Can't convert float to DataType {}",
              getDataTypeInfoMap().at(dtype).name());
}
Optimizer::Optimizer(OptimizerValue ls_) : ls(ls_) {
  // Reject loss scaling of 0.
  if (!(ls.val() > 0.0f || ls.val() < 0.0f)) {
    throw error("Loss scaling cannot be 0");
  }
}

SGD::SGD(OptimizerValue lr, OptimizerValue wd, OptimizerValue ls)
    : Optimizer(ls), lrs(lr), wds(wd) {}

std::unique_ptr<Op> SGD::createOp(const Tensor &w, Graph &graph) const {
  return std::make_unique<SGDVarUpdateOp>(w.id,
                                          scaledLearningRate(w),
                                          weightDecayScaleFactor(w),
                                          Op::Settings(graph, ""));
}

std::vector<TensorId> SGD::getInputIds(const Tensor &w) const {

  const TensorId &varId = w.id;
  std::vector<TensorId> inputs(4, "");

  // variable
  inputs[VarUpdateOp::getVarToUpdateInIndex()] = varId;

  // gradient
  inputs[VarUpdateOp::getUpdaterInIndex()] = getGradId(varId);

  // scaled learning rate (optional)
  inputs[SGDVarUpdateOp::getScaledLearningRateInIndex()] =
      scaledLearningRateIsConst(w) ? "" : getScaledLearningRateId(w);

  // weight decay scale factor (optional)
  inputs[SGDVarUpdateOp::getWeightDecayScaleFactorInIndex()] =
      weightDecayScaleFactorIsConst(w) ? "" : getWeightDecayScaleFactorId(w);
  return inputs;
}

std::vector<std::tuple<TensorId, TensorInfo>>
SGD::getOptimizerInputs(const Tensor &weight) const {
  std::vector<TensorId> optimizerTensorInputIds;
  if (!scaledLearningRateIsConst(weight)) {
    optimizerTensorInputIds.push_back(getScaledLearningRateId(weight));
  }
  if (!weightDecayScaleFactorIsConst(weight)) {
    optimizerTensorInputIds.push_back(getWeightDecayScaleFactorId(weight));
  }

  std::vector<std::tuple<TensorId, TensorInfo>> optInputs;
  for (auto id : optimizerTensorInputIds) {
    optInputs.push_back(
        std::make_tuple(id, TensorInfo(weight.info.dataType(), {})));
  }
  return optInputs;
}

void SGD::setTensorData(Tensor &optTensor) const {
  const auto &info   = optTensor.info;
  float storedValue  = getStoredValue(optTensor);
  auto convertedData = convertFloatToDataType(info.dataType(), storedValue);

  logging::ir::trace(
      "Setting TensorData for {} to {}", optTensor.str(), storedValue);
  optTensor.setTensorData(info, convertedData.data());
}

void SGD::resetTensorData(Tensor &optTensor) const {
  const auto &info   = optTensor.info;
  float storedValue  = getStoredValue(optTensor);
  auto convertedData = convertFloatToDataType(info.dataType(), storedValue);
  logging::ir::trace(
      "Resetting TensorData for {} to {}", optTensor.str(), storedValue);
  optTensor.tensorData()->resetData(info, convertedData.data());
}

float SGD::getStoredValue(const Tensor &opt) const {

  // loss scaling
  if (opt.id.find(reservedLossScalingPrefix()) != std::string::npos) {
    return lossScaling().val();
  }

  // global learning rate
  if (opt.id.find(reservedGlobalScaledLearningRatePrefix()) !=
      std::string::npos) {
    return getScaledLearningRate(lrs.getGlobal().val(), lossScaling().val());
  }

  // specific learning rate
  if (opt.id.find(reservedSpecificScaledLearningRatePrefix()) !=
      std::string::npos) {
    auto weightId = stripWeightIdFromSpecificLearningRate(opt.id);
    return getScaledLearningRate(lrs.get(weightId).val(), lossScaling().val());
  }

  // weight decay
  if (opt.id.find(reservedGlobalWeightDecayScaleFactorPrefix()) !=
      std::string::npos) {
    return getWeightDecayScaleFactor(wds.getGlobal().val(),
                                     lrs.getGlobal().val());
  }

  if (opt.id.find(reservedSpecificWeightDecayScaleFactorPrefix()) !=
      std::string::npos) {
    auto weightId = stripWeightIdFromSpecificWeightDecay(opt.id);
    return getWeightDecayScaleFactor(wds.get(weightId).val(),
                                     lrs.get(weightId).val());
  }

  throw error("In getStoredValue for {}, it doesn't match any existing "
              "optimizer prefix",
              opt.id);
}

bool OptimizerValueMap::validReplacement(const OptimizerValueMap &ovm) const {

  if (!global.validReplacement(ovm.global)) {
    return false;
  }

  if (specifics.size() != ovm.specifics.size()) {
    return false;
  }

  for (const auto &id_ov : specifics) {
    const TensorId &id = id_ov.first;
    const auto &ov     = id_ov.second;
    auto ovm_found     = ovm.specifics.find(id);
    if (ovm_found == ovm.specifics.end()) {
      return false;
    }
    if (!ov.validReplacement(ovm_found->second)) {
      return false;
    }
  }
  return true;
}

bool SGD::validReplacement(const Optimizer &other) const {
  if (other.type() != type()) {
    return false;
  }

  auto asSgd = dynamic_cast<const SGD *>(&other);
  if (!asSgd) {
    throw error("ILE: other has same `type' as this SGD, but cannot be "
                "dynamically cast to SGD. Has there been a redesign of the "
                "optimizer classes, if so this needs a rethink");
  }

  if (!lossScaling().validReplacement(other.lossScaling())) {
    return false;
  }

  if (!lrs.validReplacement(asSgd->lrs)) {
    return false;
  }

  if (!wds.validReplacement(asSgd->wds)) {
    return false;
  }

  return true;
}

std::unique_ptr<Optimizer> SGD::clone() const {
  return std::make_unique<SGD>(*this);
}

} // namespace popart
