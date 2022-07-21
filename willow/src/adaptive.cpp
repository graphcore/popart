// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <boost/container_hash/hash.hpp>
#include <cstddef>
#include <initializer_list>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <popart/adaptive.hpp>
#include <popart/error.hpp>
#include <popart/op/adaptivecombo.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensornames.hpp>
#include <popart/util.hpp>

#include "popart/clipnormsettings.hpp"
#include "popart/compoundscalarhelper.hpp"
#include "popart/datatype.hpp"
#include "popart/debugcontext.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/op/varupdate.hpp"
#include "popart/optimizer.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/optimizervaluemap.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
class Graph;

Adaptive
Adaptive::fromDefaultMap(const std::map<std::string, OptimizerValue> &m,
                         AdaptiveMode adaptiveMode_,
                         WeightDecayMode decayMode_,
                         DataType accumType_,
                         DataType accl1Type_,
                         DataType accl2Type_,
                         DataType accl3Type_,
                         const DebugContext &debugContext) {
  return Adaptive(getComplete(m),
                  adaptiveMode_,
                  decayMode_,
                  accumType_,
                  accl1Type_,
                  accl2Type_,
                  accl3Type_,
                  1011,
                  false,
                  debugContext);
}

namespace {
const std::vector<std::string> &getSpecificNames() {
  const static std::vector<std::string> names{
      "learningRate", "weightDecay", "alpha", "momentum", "eps"};
  return names;
}
} // namespace

Adaptive::Adaptive(const std::map<std::string, std::pair<float, bool>> &m,
                   AdaptiveMode adaptiveMode_,
                   WeightDecayMode decayMode_,
                   DataType accumType_,
                   DataType accl1Type_,
                   DataType accl2Type_,
                   DataType accl3Type_,
                   bool rmspropTFVariant,
                   const DebugContext &debugContext)
    : Adaptive(getComplete(getOptMap(m)),
               adaptiveMode_,
               decayMode_,
               accumType_,
               accl1Type_,
               accl2Type_,
               accl3Type_,
               31415,
               rmspropTFVariant,
               debugContext) {}

void Adaptive::insertSpecific(
    const TensorId &id,
    const std::map<std::string, std::pair<float, bool>> &m0) {

  const auto &names = getSpecificNames();

  std::map<std::string, OptimizerValue> complete;

  // Note that these take the default values already set in the maps, not the
  // "unset" values. So if a user has set default momentum to 0.7, and there is
  // no momentum key in m0, then this tensor "id" will have momentum 0.7.
  complete.insert({"learningRate", lrs.getDefault()});
  complete.insert({"weightDecay", wds.getDefault()});
  complete.insert({"alpha", as.getDefault()});
  complete.insert({"momentum", ms.getDefault()});
  complete.insert({"eps", epsvs.getDefault()});
  for (auto key_val : m0) {
    if (std::find(names.cbegin(), names.cend(), key_val.first) ==
        names.cend()) {
      std::ostringstream oss;
      oss << "Invalid key " << key_val.first
          << " in Adaptive::insertSpecific. Permitted keys are ( ";
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
                 complete.at("alpha"),
                 complete.at("momentum"),
                 complete.at("eps"));
}

bool Adaptive::hasSpecific(const Tensor &w) const {

  // confirm that all the atomic scalars have a specific value for "w"
  const auto &id = w.id;
  int counter    = 0;
  counter += lrs.hasSpecific(id);
  counter += wds.hasSpecific(id);
  counter += as.hasSpecific(id);
  counter += ms.hasSpecific(id);
  counter += epsvs.hasSpecific(id);

  if (counter != 0 && counter != getSpecificNames().size()) {
    throw error(
        "Inconsistency in Adaptive::hasSpecific : there should either be a "
        "specific value for ALL (5) or NO (0) atomic scalar values, "
        "not {} of them. ",
        counter);
  }

  return counter > 0;
}

bool Adaptive::hasSpecific() const {
  auto specifics = {lrs.hasSpecific(),
                    wds.hasSpecific(),
                    as.hasSpecific(),
                    ms.hasSpecific(),
                    epsvs.hasSpecific()};
  return std::any_of(
      specifics.begin(), specifics.end(), [](bool s) { return s; });
}

void Adaptive::insertSpecific(const TensorId &id,
                              OptimizerValue lr,
                              OptimizerValue wd,
                              OptimizerValue a,
                              OptimizerValue m,
                              OptimizerValue eps) {

  lrs.insertSpecific(id, lr);
  wds.insertSpecific(id, wd);
  as.insertSpecific(id, a);
  ms.insertSpecific(id, m);
  epsvs.insertSpecific(id, eps);

  runValueChecks(lr, wd, a, m, eps);
}

TensorId Adaptive::getInverseLossScalingTensorId(const Tensor &weight) const {
  return getInputIds(weight).at(AdaptiveComboOp::getGsInIndex());
}

void Adaptive::runValueChecks(OptimizerValue lr,
                              OptimizerValue wd,
                              OptimizerValue a,
                              OptimizerValue m,
                              OptimizerValue eps) const {
  if (lr.val() < 0) {
    throw error(
        "Negative learning rate ({}) in Adaptive, bailing as this might be "
        "a user error.",
        lr.val());
  } else if (lr.val() == 0.0f && lr.isConst()) {
    throw error(
        "Constant, zero learning rate in Adaptive, bailing as this might be "
        "a user error.");
  }

  if (wd.val() < 0) {
    throw error(
        "Negative weight decay ({}) in Adaptive, bailing as this might be a "
        "user error",
        wd.val());
  }

  if (a.val() < 0) {
    throw error("Negative alpha ({}) in Adaptive, bailing as this might be a "
                "user error",
                a.val());
  }

  if (m.val() < 0) {
    throw error(
        "Negative momentum ({}) in Adaptive, bailing as this might be a "
        "user error",
        a.val());
  }

  if (eps.val() <= 0) {
    throw error("Non-positive eps ({}) in Adaptive is not supported",
                eps.val());
  }
}

Adaptive::Adaptive(const std::map<std::string, OptimizerValue> &cmap,
                   AdaptiveMode mode_,
                   WeightDecayMode decayMode_,
                   DataType accumType_,
                   DataType accl1Type_,
                   DataType accl2Type_,
                   DataType accl3Type_,
                   int,
                   bool rmspropTFVariant,
                   const DebugContext &debugContext)
    : Adaptive(cmap.at("defaultLearningRate"),
               cmap.at("defaultWeightDecay"),
               cmap.at("defaultAlpha"),
               cmap.at("defaultMomentum"),
               cmap.at("defaultEps"),
               cmap.at("lossScaling"),
               mode_,
               decayMode_,
               accumType_,
               accl1Type_,
               accl2Type_,
               accl3Type_,
               rmspropTFVariant,
               debugContext) {}

Adaptive::Adaptive(OptimizerValue lr,
                   OptimizerValue wd,
                   OptimizerValue a,
                   OptimizerValue m,
                   OptimizerValue eps,
                   OptimizerValue lossScaling,
                   AdaptiveMode mode_,
                   WeightDecayMode decayMode_,
                   DataType accumType_,
                   DataType accl1Type_,
                   DataType accl2Type_,
                   DataType accl3Type_,
                   bool rmspropTFVariant_,
                   const DebugContext &debugContext_)
    : Optimizer(lossScaling, {}, debugContext_), lrs(lr), wds(wd), as(a), ms(m),
      epsvs(eps), mode(mode_), decayMode(decayMode_), accumType(accumType_),
      accl1Type(accl1Type_), accl2Type(accl1Type_), accl3Type(accl3Type_),
      rmspropTFVariant(rmspropTFVariant_) {
  if (rmspropTFVariant_ && mode != AdaptiveMode::RMSProp &&
      mode != AdaptiveMode::CenteredRMSProp) {
    throw error("The rmspropTFVariant parameter is only valid with the RMSProp "
                "and CenteredRMSProp optimizers.");
  }
  runValueChecks(lr, wd, a, m, eps);
}

std::map<std::string, OptimizerValue>
Adaptive::getComplete(const std::map<std::string, OptimizerValue> &m) {

  std::vector<std::string> argNames{"defaultLearningRate",
                                    "defaultWeightDecay",
                                    "defaultAlpha",
                                    "defaultMomentum",
                                    "defaultEps",
                                    "lossScaling"};

  std::map<std::string, OptimizerValue> complete{};

  complete.insert({"defaultLearningRate", getUnsetLearningRate()});
  complete.insert({"defaultWeightDecay", getUnsetWeightDecay()});
  complete.insert({"defaultAlpha", getUnsetAlpha()});
  complete.insert({"defaultMomentum", getUnsetMomentum()});
  complete.insert({"defaultEps", getUnsetEps()});
  complete.insert({"lossScaling", getUnsetLossScaling()});

  for (auto key_val : m) {
    auto key = key_val.first;
    auto val = key_val.second;
    if (std::find(argNames.cbegin(), argNames.cend(), key) == argNames.cend()) {
      std::ostringstream oss;
      oss << "Invalid Adaptive key, " << key << ", the allowed keys are ( ";
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

std::unique_ptr<Op> Adaptive::createOp(const Tensor &w, Graph &graph) const {

  DebugInfo debugInfo(debugContext, "popartbuilder");
  auto opSettings = Op::Settings(graph, "", debugInfo.getId());

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

  return std::make_unique<AdaptiveComboOp>(
      lrhelper.getFromWeightId(w.id, *this),
      wdhelper.getFromWeightId(w.id, *this),
      ahelper.getFromWeightId(w.id, *this),
      mhelper.getFromWeightId(w.id, *this),
      epshelper.getFromWeightId(w.id, *this),
      lshelper.getFromWeightId(w.id, *this),
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
      accl3Type == DataType::UNDEFINED ? w.info.getDataTypeInfo()->type()
                                       : accl3Type,
      rmspropTFVariant,
      opSettings);
}

std::vector<TensorId> Adaptive::getInputIds(const Tensor &w) const {
  const TensorId &varId = w.id;
  std::vector<TensorId> inputs(9, "");

  // variable
  inputs[VarUpdateOp::getVarToUpdateInIndex()] = varId;

  // gradient
  inputs[VarUpdateWithUpdaterOp::getUpdaterInIndex()] = getGradId(varId);

  // lrhelper
  inputs[AdaptiveComboOp::getLrInIndex()] =
      lrhelper.getScalarIdIfNonConst(w, *this);

  // wdhelper
  inputs[AdaptiveComboOp::getWdInIndex()] =
      wdhelper.getScalarIdIfNonConst(w, *this);

  // alpha
  inputs[AdaptiveComboOp::getAlphaInIndex()] =
      ahelper.getScalarIdIfNonConst(w, *this);

  // momentum
  inputs[AdaptiveComboOp::getMomentumInIndex()] =
      mhelper.getScalarIdIfNonConst(w, *this);

  // eps
  inputs[AdaptiveComboOp::getEpsInIndex()] =
      epshelper.getScalarIdIfNonConst(w, *this);

  // loss scaling
  inputs[AdaptiveComboOp::getLsInIndex()] =
      lshelper.getScalarIdIfNonConst(w, *this);

  // gradient scaling
  inputs[AdaptiveComboOp::getGsInIndex()] =
      gshelper.getScalarIdIfNonConst(w, *this);

  return inputs;
}

std::vector<std::tuple<TensorId, TensorInfo>>
Adaptive::getOptimizerInputs(const Tensor &weight) const {

  std::vector<TensorId> ids;

  ids.push_back(lrhelper.getScalarIdIfNonConst(weight, *this));
  ids.push_back(wdhelper.getScalarIdIfNonConst(weight, *this));
  ids.push_back(ahelper.getScalarIdIfNonConst(weight, *this));
  ids.push_back(mhelper.getScalarIdIfNonConst(weight, *this));
  ids.push_back(epshelper.getScalarIdIfNonConst(weight, *this));
  ids.push_back(lshelper.getScalarIdIfNonConst(weight, *this));
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

void Adaptive::setTensorData(Tensor &optTensor) const {
  const auto &info   = optTensor.info;
  float storedValue  = getStoredValue(optTensor.id);
  auto convertedData = convertFloatToDataType(info.dataType(), storedValue);

  logging::ir::trace(
      "Setting TensorData for {} to {}", optTensor.str(), storedValue);
  optTensor.setTensorData(info, convertedData.data());
}

void Adaptive::resetTensorData(Tensor &optTensor) const {
  const auto &info   = optTensor.info;
  float storedValue  = getStoredValue(optTensor.id);
  auto convertedData = convertFloatToDataType(info.dataType(), storedValue);
  logging::ir::trace(
      "Resetting TensorData for {} to {}", optTensor.str(), storedValue);
  optTensor.tensorData()->resetData(info, convertedData.data());
}

float Adaptive::getStoredValue(const TensorId &optId) const {

  if (optId.find(reservedLossScalingPrefix()) != std::string::npos) {
    return lossScaling().val();
  }

  if (ahelper.idMatch(optId)) {
    return ahelper.getFromScalarId(optId, *this).val();
  }

  if (mhelper.idMatch(optId)) {
    return mhelper.getFromScalarId(optId, *this).val();
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

  if (gshelper.idMatch(optId)) {
    return gshelper.getFromScalarId(optId, *this).val();
  }

  throw runtime_error("In getStoredValue for {}, it doesn't match any existing "
                      "optimizer prefix",
                      optId);
}

void Adaptive::validReplacement(const Optimizer &other) const {
  Optimizer::validReplacement(other);

  auto asAdaptive = dynamic_cast<const Adaptive *>(&other);
  if (!asAdaptive) {
    throw internal_error(
        "other has same `type' as this Adaptive, but cannot be "
        "dynamically cast to Adaptive. Has there been a redesign of the "
        "optimizer classes? if so this needs a rethink");
  }

  if (asAdaptive->mode != mode) {
    throw optimizer_replacement_error("Adaptive modes do not match");
  }

  if (asAdaptive->decayMode != decayMode) {
    throw optimizer_replacement_error("Weight decay modes do not match");
  }

  checkReplacementValue(lossScaling(), other.lossScaling(), "loss scaling");
  checkReplacementValue(lrs, asAdaptive->lrs, "learning rates");
  checkReplacementValue(wds, asAdaptive->wds, "weight decays");
  checkReplacementValue(as, asAdaptive->as, "alphas");
  checkReplacementValue(ms, asAdaptive->ms, "momentums");
  checkReplacementValue(epsvs, asAdaptive->epsvs, "eps");
}

std::unique_ptr<Optimizer> Adaptive::clone() const {
  return std::make_unique<Adaptive>(*this);
}

size_t Adaptive::hash() const {
  std::size_t seed = 0;
  boost::hash_combine(seed, Optimizer::hash());
  boost::hash_combine(seed, lrs);
  boost::hash_combine(seed, wds);
  boost::hash_combine(seed, as);
  boost::hash_combine(seed, ms);
  boost::hash_combine(seed, epsvs);

  boost::hash_combine(seed, static_cast<int>(type()));
  boost::hash_combine(seed, static_cast<int>(mode));
  boost::hash_combine(seed, static_cast<int>(accumType));
  boost::hash_combine(seed, static_cast<int>(accl1Type));
  boost::hash_combine(seed, static_cast<int>(accl2Type));
  boost::hash_combine(seed, static_cast<int>(accl3Type));
  boost::hash_combine(seed, static_cast<int>(decayMode));

  bool hasVelocityTensor =
      !ms.getDefault().isConst() || ms.getDefault().val() != 0.0f;
  boost::hash_combine(seed, hasVelocityTensor);
  return seed;
}

} // namespace popart
