// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/adam.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/adamcombo.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/util.hpp>

#include <boost/functional/hash.hpp>

namespace popart {

Adam Adam::fromDefaultMap(const std::map<std::string, OptimizerValue> &m,
                          AdamMode adamMode_,
                          WeightDecayMode decayMode_,
                          DataType accumType_,
                          DataType accl1Type_,
                          DataType accl2Type_) {
  return Adam(getComplete(m),
              adamMode_,
              decayMode_,
              accumType_,
              accl1Type_,
              accl2Type_,
              1011);
}

namespace {
const std::vector<std::string> &getSpecificNames() {
  const static std::vector<std::string> names{
      "learningRate", "weightDecay", "beta1", "beta2", "eps"};
  return names;
}
} // namespace

Adam::Adam(const std::map<std::string, std::pair<float, bool>> &m,
           AdamMode adamMode_,
           WeightDecayMode decayMode_,
           DataType accumType_,
           DataType accl1Type_,
           DataType accl2Type_)
    : Adam(getComplete(getOptMap(m)),
           adamMode_,
           decayMode_,
           accumType_,
           accl1Type_,
           accl2Type_,
           31415) {}

void Adam::insertSpecific(
    const TensorId &id,
    const std::map<std::string, std::pair<float, bool>> &m0) {

  const auto &names = getSpecificNames();

  std::map<std::string, OptimizerValue> complete;

  // Note that these take the default values already set in the maps, not the
  // "unset" values. So if a user has set default momentum to 0.7, and there is
  // no momentum key in m0, then this tensor "id" will have momentum 0.7.
  complete.insert({"learningRate", lrs.getDefault()});
  complete.insert({"weightDecay", wds.getDefault()});
  complete.insert({"beta1", b1s.getDefault()});
  complete.insert({"beta2", b2s.getDefault()});
  complete.insert({"eps", epsvs.getDefault()});
  for (auto key_val : m0) {
    if (std::find(names.cbegin(), names.cend(), key_val.first) ==
        names.cend()) {
      std::ostringstream oss;
      oss << "Invalid key " << key_val.first
          << " in Adam::insertSpecific. Permitted keys are ( ";
      for (auto x : names) {
        oss << x << ' ';
      }
      oss << ')';
      throw error(oss.str());
    }
    complete[key_val.first] = key_val.second;
  }

  insertSpecific(id,
                 complete.at("learningRate"),
                 complete.at("weightDecay"),
                 complete.at("beta1"),
                 complete.at("beta2"),
                 complete.at("eps"));
}

bool Adam::hasSpecific(const Tensor &w) const {

  // confirm that all the atomic scalars have a specific value for "w"
  const auto &id = w.id;
  int counter    = 0;
  counter += lrs.hasSpecific(id);
  counter += wds.hasSpecific(id);
  counter += b1s.hasSpecific(id);
  counter += b2s.hasSpecific(id);
  counter += epsvs.hasSpecific(id);

  if (counter != 0 && counter != getSpecificNames().size()) {
    throw error("Inconsistency in Adam::hasSpecific : there should either be a "
                "specific value for ALL (5) or NO (0) atomic scalar values, "
                "not {} of them. ",
                counter);
  }

  return counter > 0;
}

void Adam::insertSpecific(const TensorId &id,
                          OptimizerValue lr,
                          OptimizerValue wd,
                          OptimizerValue b1,
                          OptimizerValue b2,
                          OptimizerValue eps) {

  lrs.insertSpecific(id, lr);
  wds.insertSpecific(id, wd);
  b1s.insertSpecific(id, b1);
  b2s.insertSpecific(id, b2);
  epsvs.insertSpecific(id, eps);

  runValueChecks(lr, wd, b1, b2, eps);
}

void Adam::runValueChecks(OptimizerValue lr,
                          OptimizerValue wd,
                          OptimizerValue b1,
                          OptimizerValue b2,
                          OptimizerValue eps) const {
  if (lr.val() < 0) {
    throw error("Negative learning rate ({}) in Adam, bailing as this might be "
                "a user error.",
                lr.val());
  } else if (lr.val() == 0.0f && lr.isConst()) {
    throw error(
        "Constant, zero learning rate in Adam, bailing as this might be "
        "a user error.");
  }

  if (wd.val() < 0) {
    throw error(
        "Negative weight decay ({}) in Adam, bailing as this might be a "
        "user error",
        wd.val());
  }

  if (b1.val() < 0) {
    throw error(
        "Negative beta1 ({}) in Adam, bailing as this might be a user error",
        b1.val());
  }

  if (b2.val() < 0) {
    throw error("Negative beta2 ({}) in Adam, bailing as this might be a "
                "user error",
                b2.val());
  }

  if (eps.val() <= 0) {
    throw error("Non-positive eps ({}) in Adam is not supported", eps.val());
  }
}

Adam::Adam(const std::map<std::string, OptimizerValue> &cmap,
           AdamMode mode_,
           WeightDecayMode decayMode_,
           DataType accumType_,
           DataType accl1Type_,
           DataType accl2Type_,
           int)
    : Adam(cmap.at("defaultLearningRate"),
           cmap.at("defaultWeightDecay"),
           cmap.at("defaultBeta1"),
           cmap.at("defaultBeta2"),
           cmap.at("defaultEps"),
           cmap.at("lossScaling"),
           cmap.at("maxWeightNorm"),
           mode_,
           decayMode_,
           accumType_,
           accl1Type_,
           accl2Type_) {}

Adam::Adam(OptimizerValue lr,
           OptimizerValue wd,
           OptimizerValue b1,
           OptimizerValue b2,
           OptimizerValue eps,
           OptimizerValue lossScaling,
           AdamMode mode_,
           WeightDecayMode decayMode_,
           DataType accumType_,
           DataType accl1Type_,
           DataType accl2Type_)
    : Adam(lr,
           wd,
           b1,
           b2,
           eps,
           lossScaling,
           getUnsetMaxWeightNorm(),
           mode_,
           decayMode_,
           accumType_,
           accl1Type_,
           accl2Type_) {}

Adam::Adam(OptimizerValue lr,
           OptimizerValue wd,
           OptimizerValue b1,
           OptimizerValue b2,
           OptimizerValue eps,
           OptimizerValue lossScaling,
           OptimizerValue mwn_,
           AdamMode mode_,
           DataType accumType_,
           DataType accl1Type_,
           DataType accl2Type_)
    : Adam(lr,
           wd,
           b1,
           b2,
           eps,
           lossScaling,
           mwn_,
           mode_,
           WeightDecayMode::Decay,
           accumType_,
           accl1Type_,
           accl2Type_) {}

Adam::Adam(OptimizerValue lr,
           OptimizerValue wd,
           OptimizerValue b1,
           OptimizerValue b2,
           OptimizerValue eps,
           OptimizerValue lossScaling,
           AdamMode mode_,
           DataType accumType_,
           DataType accl1Type_,
           DataType accl2Type_)
    : Adam(lr,
           wd,
           b1,
           b2,
           eps,
           lossScaling,
           getUnsetMaxWeightNorm(),
           mode_,
           WeightDecayMode::Decay,
           accumType_,
           accl1Type_,
           accl2Type_) {}

Adam::Adam(OptimizerValue lr,
           OptimizerValue wd,
           OptimizerValue b1,
           OptimizerValue b2,
           OptimizerValue eps,
           OptimizerValue lossScaling,
           OptimizerValue mwn_,
           AdamMode mode_,
           WeightDecayMode decayMode_,
           DataType accumType_,
           DataType accl1Type_,
           DataType accl2Type_)
    : Optimizer(lossScaling, {}), lrs(lr), wds(wd), b1s(b1), b2s(b2),
      epsvs(eps), mwns(mwn_), mode(mode_), decayMode(decayMode_),
      accumType(accumType_),

      accl1Type(accl1Type_), accl2Type(accl2Type_) {
  runValueChecks(lr, wd, b1, b2, eps);
}

std::map<std::string, OptimizerValue>
Adam::getComplete(const std::map<std::string, OptimizerValue> &m) {

  std::vector<std::string> argNames{"defaultLearningRate",
                                    "defaultWeightDecay",
                                    "defaultBeta1",
                                    "defaultBeta2",
                                    "defaultEps",
                                    "lossScaling",
                                    "maxWeightNorm"};

  std::map<std::string, OptimizerValue> complete{};

  complete.insert({"defaultLearningRate", getUnsetLearningRate()});
  complete.insert({"defaultWeightDecay", getUnsetWeightDecay()});
  complete.insert({"defaultBeta1", getUnsetBeta1()});
  complete.insert({"defaultBeta2", getUnsetBeta2()});
  complete.insert({"defaultEps", getUnsetEps()});
  complete.insert({"lossScaling", getUnsetLossScaling()});
  complete.insert({"maxWeightNorm", getUnsetMaxWeightNorm()});

  for (auto key_val : m) {
    auto key = key_val.first;
    auto val = key_val.second;
    if (std::find(argNames.cbegin(), argNames.cend(), key) == argNames.cend()) {
      std::ostringstream oss;
      oss << "Invalid Adam key, " << key << ", the allowed keys are ( ";
      for (auto x : argNames) {
        oss << x << ' ';
      }
      oss << ')';
      throw error(oss.str());
    }
    complete[key] = val;
  }

  return complete;
}

std::unique_ptr<Op> Adam::createOp(const Tensor &w, Graph &graph) const {

  auto opSettings = Op::Settings(graph, "");

  for (Op *op : w.consumers.getOps()) {
    for (auto &outlineAttribute : op->settings.extraOutlineAttributes) {
      opSettings.extraOutlineAttributes.insert(outlineAttribute);
    }
  }

  OptimizerReductionType reductionType{OptimizerReductionType::None};

  if (getReplicatedGraphCount() > 1) {
    if (gradientAccumulationEnabled()) {
      reductionType = OptimizerReductionType::AccumReduce;
    } else {
      // Disable acclReduce in favor of gradReduce when not using gradient
      // accumulation
      reductionType = OptimizerReductionType::GradReduce;
    }
  }

  // velocity required
  return std::make_unique<AdamComboOp>(
      lrhelper.getFromWeightId(w.id, *this),
      wdhelper.getFromWeightId(w.id, *this),
      b1helper.getFromWeightId(w.id, *this),
      b2helper.getFromWeightId(w.id, *this),
      epshelper.getFromWeightId(w.id, *this),
      lshelper.getFromWeightId(w.id, *this),
      mwnhelper.getFromWeightId(w.id, *this),
      gshelper.getFromWeightId(w.id, *this),
      mode,
      decayMode,
      gradientAccumulationEnabled(),
      reductionType,
      accumType == DataType::UNDEFINED ? w.info.getDataTypeInfo()->type()
                                       : accumType,
      accl1Type == DataType::UNDEFINED ? w.info.getDataTypeInfo()->type()
                                       : accl1Type,
      accl2Type == DataType::UNDEFINED ? w.info.getDataTypeInfo()->type()
                                       : accl2Type,
      opSettings);
}

std::vector<TensorId> Adam::getInputIds(const Tensor &w) const {
  const TensorId &varId = w.id;
  std::vector<TensorId> inputs(10, "");

  // variable
  inputs[VarUpdateOp::getVarToUpdateInIndex()] = varId;

  // gradient
  inputs[VarUpdateWithUpdaterOp::getUpdaterInIndex()] = getGradId(varId);

  // lrhelper
  inputs[AdamComboOp::getLrInIndex()] =
      lrhelper.getScalarIdIfNonConst(w, *this);

  // wdhelper
  inputs[AdamComboOp::getWdInIndex()] =
      wdhelper.getScalarIdIfNonConst(w, *this);

  // beta1
  inputs[AdamComboOp::getBeta1InIndex()] =
      b1helper.getScalarIdIfNonConst(w, *this);

  // beta2
  inputs[AdamComboOp::getBeta2InIndex()] =
      b2helper.getScalarIdIfNonConst(w, *this);

  // eps
  inputs[AdamComboOp::getEpsInIndex()] =
      epshelper.getScalarIdIfNonConst(w, *this);

  // loss scaling
  inputs[AdamComboOp::getLsInIndex()] =
      lshelper.getScalarIdIfNonConst(w, *this);

  // max weight norm
  inputs[AdamComboOp::getMwnInIndex()] =
      mwnhelper.getScalarIdIfNonConst(w, *this);

  // gradient scaling
  inputs[AdamComboOp::getGsInIndex()] =
      gshelper.getScalarIdIfNonConst(w, *this);

  return inputs;
}

std::vector<std::tuple<TensorId, TensorInfo>>
Adam::getOptimizerInputs(const Tensor &weight) const {

  std::vector<TensorId> ids;

  ids.push_back(lrhelper.getScalarIdIfNonConst(weight, *this));
  ids.push_back(wdhelper.getScalarIdIfNonConst(weight, *this));
  ids.push_back(b1helper.getScalarIdIfNonConst(weight, *this));
  ids.push_back(b2helper.getScalarIdIfNonConst(weight, *this));
  ids.push_back(epshelper.getScalarIdIfNonConst(weight, *this));
  ids.push_back(lshelper.getScalarIdIfNonConst(weight, *this));
  ids.push_back(mwnhelper.getScalarIdIfNonConst(weight, *this));
  ids.push_back(gshelper.getScalarIdIfNonConst(weight, *this));

  std::vector<std::tuple<TensorId, TensorInfo>> optInputs;
  for (const auto &id : ids) {
    if (!id.empty()) {
      auto tuppy = std::make_tuple(id, TensorInfo(DataType::FLOAT, {}));
      optInputs.push_back(tuppy);
    }
  }

  return optInputs;
}

void Adam::setTensorData(Tensor &optTensor) const {
  const auto &info   = optTensor.info;
  float storedValue  = getStoredValue(optTensor.id);
  auto convertedData = convertFloatToDataType(info.dataType(), storedValue);

  logging::ir::trace(
      "Setting TensorData for {} to {}", optTensor.str(), storedValue);
  optTensor.setTensorData(info, convertedData.data());
}

void Adam::resetTensorData(Tensor &optTensor) const {
  const auto &info   = optTensor.info;
  float storedValue  = getStoredValue(optTensor.id);
  auto convertedData = convertFloatToDataType(info.dataType(), storedValue);
  logging::ir::trace(
      "Resetting TensorData for {} to {}", optTensor.str(), storedValue);
  optTensor.tensorData()->resetData(info, convertedData.data());
}

float Adam::getStoredValue(const TensorId &optId) const {

  if (optId.find(reservedLossScalingPrefix()) != std::string::npos) {
    return lossScaling().val();
  }

  if (b1helper.idMatch(optId)) {
    return b1helper.getFromScalarId(optId, *this).val();
  }

  if (b2helper.idMatch(optId)) {
    return b2helper.getFromScalarId(optId, *this).val();
  }

  if (wdhelper.idMatch(optId)) {
    return wdhelper.getFromScalarId(optId, *this).val();
  }

  if (epshelper.idMatch(optId)) {
    return epshelper.getFromScalarId(optId, *this).val();
  }

  if (lrhelper.idMatch(optId)) {
    return lrhelper.getFromScalarId(optId, *this).val();
  }

  if (lshelper.idMatch(optId)) {
    return lshelper.getFromScalarId(optId, *this).val();
  }

  if (mwnhelper.idMatch(optId)) {
    return mwnhelper.getFromScalarId(optId, *this).val();
  }

  if (gshelper.idMatch(optId)) {
    return gshelper.getFromScalarId(optId, *this).val();
  }

  throw error("In getStoredValue for {}, it doesn't match any existing "
              "optimizer prefix",
              optId);
}

bool Adam::validReplacement(const Optimizer &other) const {
  if (!Optimizer::validReplacement(other)) {
    return false;
  }

  if (other.type() != type()) {
    return false;
  }

  auto asAdam = dynamic_cast<const Adam *>(&other);
  if (!asAdam) {
    throw internal_error(
        "other has same `type' as this Adam, but cannot be "
        "dynamically cast to Adam. Has there been a redesign of the "
        "optimizer classes? if so this needs a rethink");
  }

  if (asAdam->mode != mode) {
    return false;
  }

  if (asAdam->decayMode != decayMode) {
    return false;
  }

  logging::ir::debug("Checking loss scaling for compatibility");
  if (!lossScaling().validReplacement(other.lossScaling())) {
    return false;
  }

  logging::ir::debug("Checking learning rates for compatibility");
  if (!lrs.validReplacement(asAdam->lrs)) {
    return false;
  }

  logging::ir::debug("Checking weight decays for compatibility");
  if (!wds.validReplacement(asAdam->wds)) {
    return false;
  }

  logging::ir::debug("Checking beta1s for compatibility");
  if (!b1s.validReplacement(asAdam->b1s)) {
    return false;
  }

  logging::ir::debug("Checking beta2s for compatibility");
  if (!b2s.validReplacement(asAdam->b2s)) {
    return false;
  }

  logging::ir::debug("Checking eps for compatibility");
  if (!epsvs.validReplacement(asAdam->epsvs)) {
    return false;
  }

  logging::ir::debug("Checking mwn for compatibility");
  if (!mwns.validReplacement(asAdam->mwns)) {
    return false;
  }

  return true;
}

size_t Adam::hash() const {
  std::size_t seed = 0;
  boost::hash_combine(seed, Optimizer::hash());
  boost::hash_combine(seed, lrs);
  boost::hash_combine(seed, wds);
  boost::hash_combine(seed, b1s);
  boost::hash_combine(seed, b2s);
  boost::hash_combine(seed, epsvs);
  boost::hash_combine(seed, mwns);

  boost::hash_combine(seed, static_cast<int>(type()));
  boost::hash_combine(seed, static_cast<int>(mode));
  boost::hash_combine(seed, static_cast<int>(decayMode));
  boost::hash_combine(seed, static_cast<int>(accumType));
  boost::hash_combine(seed, static_cast<int>(accl1Type));
  boost::hash_combine(seed, static_cast<int>(accl2Type));

  return seed;
}

std::unique_ptr<Optimizer> Adam::clone() const {
  return std::make_unique<Adam>(*this);
}

} // namespace popart
