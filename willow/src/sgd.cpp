// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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
#include <popart/error.hpp>
#include <popart/op/sgd0combo.hpp>
#include <popart/op/sgd1combo.hpp>
#include <popart/op/sgd2combo.hpp>
#include <popart/sgd.hpp>
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
#include "popart/op/sgdcombobase.hpp"
#include "popart/op/varupdate.hpp"
#include "popart/optimizer.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/optimizervaluemap.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
class Graph;

namespace {
const std::vector<std::string> &getSpecificNames() {
  const static std::vector<std::string> names{"learningRate",
                                              "weightDecay",
                                              "momentum",
                                              "dampening",
                                              "velocityScaling",
                                              "nesterov"};
  return names;
}

} // namespace

SGD SGD::fromDefaultMap(const std::map<std::string, OptimizerValue> &m,
                        const DebugContext &debugContext) {
  return SGD(getComplete(m),
             {},
             SGDAccumulatorAndMomentum::Combined,
             DataType::UNDEFINED,
             DataType::UNDEFINED,
             1011,
             debugContext);
}

SGD::SGD(const std::map<std::string, std::pair<float, bool>> &m,
         const std::vector<ClipNormSettings> &clipNormSettings,
         SGDAccumulatorAndMomentum sgdAccMm,
         DataType accumType,
         DataType accl1Type,
         const DebugContext &debugContext)
    : SGD(getComplete(getOptMap(m)),
          clipNormSettings,
          sgdAccMm,
          accumType,
          accl1Type,
          31415,
          debugContext) {}

void SGD::insertSpecific(
    const TensorId &id,
    const std::map<std::string, std::pair<float, bool>> &m0) {

  const auto &names = getSpecificNames();

  std::map<std::string, OptimizerValue> complete;

  // Note that these take the default values already set in the maps, not the
  // "unset" values. So if a user has set default momentum to 0.7, and there is
  // no momentum key in m0, then this tensor "id" will have momentum 0.7.
  complete.insert({"learningRate", lrs.getDefault()});
  complete.insert({"weightDecay", wds.getDefault()});
  complete.insert({"momentum", mms.getDefault()});
  complete.insert({"dampening", dps.getDefault()});
  complete.insert({"velocityScaling", vss.getDefault()});
  complete.insert({"nesterov", nts.getDefault()});
  for (auto key_val : m0) {
    if (std::find(names.cbegin(), names.cend(), key_val.first) ==
        names.cend()) {
      std::ostringstream oss;
      oss << "Invalid key " << key_val.first
          << " in SGD::insertSpecific. Permitted keys are ( ";
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
                 complete.at("momentum"),
                 complete.at("dampening"),
                 complete.at("velocityScaling"),
                 complete.at("nesterov"));
}

bool SGD::hasSpecific(const Tensor &w) const {

  // confirm that all the atomic scalars have a specigic value for "w"
  const auto &id = w.id;
  int counter    = 0;
  counter += lrs.hasSpecific(id);
  counter += wds.hasSpecific(id);
  counter += mms.hasSpecific(id);
  counter += dps.hasSpecific(id);
  counter += vss.hasSpecific(id);
  counter += nts.hasSpecific(id);

  if (counter != 0 && counter != getSpecificNames().size()) {
    throw error("Inconsistency in SGD::hasSpecific : there should either be a "
                "specific value for ALL (5) or NO (0) atomic scalar values, "
                "not {} of them. ",
                counter);
  }

  return counter > 0;
}

bool SGD::hasSpecific() const {
  auto specifics = {lrs.hasSpecific(),
                    wds.hasSpecific(),
                    mms.hasSpecific(),
                    dps.hasSpecific(),
                    vss.hasSpecific(),
                    nts.hasSpecific()};
  return std::any_of(
      specifics.begin(), specifics.end(), [](bool s) { return s; });
}

bool SGD::hasMomentum(const Tensor &weight) const {
  OptimizerValue mm = mms.get(weight.id);
  return !mm.isConst() || mm.val() != 0.0f;
}

bool SGD::enableNesterov(const Tensor &weight) const {
  float nt = nts.get(weight.id).val();
  float mm = mms.get(weight.id).val();
  return nt != 0 && mm > 0;
}

void SGD::insertSpecific(const TensorId &id,
                         OptimizerValue lr,
                         OptimizerValue wd,
                         OptimizerValue mm,
                         OptimizerValue dp,
                         OptimizerValue vs,
                         OptimizerValue nt) {

  lrs.insertSpecific(id, lr);
  wds.insertSpecific(id, wd);
  mms.insertSpecific(id, mm);
  dps.insertSpecific(id, dp);
  vss.insertSpecific(id, vs);
  nts.insertSpecific(id, nt);

  runValueChecks(lr, wd, mm, dp, vs, nt);
}

void SGD::runValueChecks(OptimizerValue lr,
                         OptimizerValue wd,
                         OptimizerValue mm,
                         OptimizerValue dp,
                         OptimizerValue vs,
                         OptimizerValue nt) const {
  if (lr.val() < 0) {
    throw error("Negative learning rate ({}) in SGD, bailing as this might be "
                "a user error.",
                lr.val());
  } else if (lr.val() == 0.0f && lr.isConst()) {
    throw error("Constant, zero learning rate in SGD, bailing as this might be "
                "a user error.");
  }

  if (wd.val() < 0) {
    throw error("Negative weight decay ({}) in SGD, bailing as this might be a "
                "user error",
                wd.val());
  }

  if (mm.val() < 0) {
    throw error(
        "Negative momentum ({}) in SGD, bailing as this might be a user error",
        mm.val());
  }

  if (dp.val() < 0) {
    throw error(
        "Negative dampening ({}) in SGD, bailing as this might be a user error",
        dp.val());
  }

  if (vs.val() <= 0) {
    throw error("Non-positive velocity scaling ({}) in SGD is not supported",
                vs.val());
  }

  if (nt.val() != 0 && dp.val() > 0) {
    throw error(
        "When enable nesterov momentum, dampening ({}) in SGD shouled be zero",
        dp.val());
  }
}

SGD::SGD(const std::map<std::string, OptimizerValue> &cmap,
         const std::vector<ClipNormSettings> &clipNormSettings,
         SGDAccumulatorAndMomentum sgdAccMm,
         DataType accumType,
         DataType accl1Type,
         int,
         const DebugContext &debugContext)
    : SGD(cmap.at("defaultLearningRate"),
          cmap.at("defaultWeightDecay"),
          cmap.at("defaultMomentum"),
          cmap.at("defaultDampening"),
          cmap.at("defaultVelocityScaling"),
          cmap.at("lossScaling"),
          cmap.at("nesterov"),
          clipNormSettings,
          sgdAccMm,
          accumType,
          accl1Type,
          debugContext) {}

SGD::SGD(OptimizerValue lr,
         OptimizerValue wd,
         OptimizerValue mm,
         OptimizerValue dp,
         OptimizerValue vs,
         OptimizerValue lossScaling,
         OptimizerValue nesterov,
         const std::vector<ClipNormSettings> &clipNormSettings,
         SGDAccumulatorAndMomentum sgdAccMm_,
         DataType accumType_,
         DataType accl1Type_,
         const DebugContext &debugContext_)
    : Optimizer(lossScaling, clipNormSettings, debugContext_), lrs(lr), wds(wd),
      mms(mm), dps(dp), vss(vs), nts(nesterov), sgdAccMm(sgdAccMm_),
      sgdAccumType(accumType_), sgd2Accl1Type(accl1Type_) {
  runValueChecks(lr, wd, mm, dp, vs, nesterov);
}

SGD::SGD(OptimizerValue lr,
         OptimizerValue wd,
         OptimizerValue mm,
         OptimizerValue dp,
         OptimizerValue vs,
         OptimizerValue lossScaling,
         const std::vector<ClipNormSettings> &clipNormSettings,
         SGDAccumulatorAndMomentum sgdAccMm_,
         DataType accumType_,
         DataType accl1Type_,
         const DebugContext &debugContext_)
    : Optimizer(lossScaling, clipNormSettings, debugContext_), lrs(lr), wds(wd),
      mms(mm), dps(dp), vss(vs), nts(OptimizerValue(0.0f, true)),
      sgdAccMm(sgdAccMm_), sgdAccumType(accumType_), sgd2Accl1Type(accl1Type_) {
  runValueChecks(lr, wd, mm, dp, vs, OptimizerValue(0.0f, true));
}

std::map<std::string, OptimizerValue>
SGD::getComplete(const std::map<std::string, OptimizerValue> &m) {

  std::vector<std::string> sevenParamArgs{"defaultLearningRate",
                                          "defaultWeightDecay",
                                          "defaultMomentum",
                                          "defaultDampening",
                                          "defaultVelocityScaling",
                                          "lossScaling",
                                          "nesterov"};

  std::map<std::string, OptimizerValue> complete{};

  complete.insert({"defaultLearningRate", getUnsetLearningRate()});
  complete.insert({"defaultWeightDecay", getUnsetWeightDecay()});
  complete.insert({"defaultMomentum", getUnsetMomentum()});
  complete.insert({"defaultDampening", getUnsetDampening()});
  complete.insert({"defaultVelocityScaling", getUnsetVelocityScaling()});
  complete.insert({"lossScaling", getUnsetLossScaling()});
  complete.insert({"nesterov", getUnsetNesterov()});

  for (auto key_val : m) {
    auto key = key_val.first;
    auto val = key_val.second;
    if (std::find(sevenParamArgs.cbegin(), sevenParamArgs.cend(), key) ==
        sevenParamArgs.cend()) {
      std::ostringstream oss;
      oss << "Invalid SGD key, " << key << ", the allowed keys are ( ";
      for (auto x : sevenParamArgs) {
        oss << x << ' ';
      }
      oss << ')';
      throw error(oss.str());
    }
    complete[key] = val;
  }

  return complete;
}

std::unique_ptr<Op> SGD::createOp(const Tensor &w, Graph &graph) const {

  OptimizerReductionType reductionType{OptimizerReductionType::None};

  bool withMomentum = hasMomentum(w);

  bool withNesterov = enableNesterov(w);

  DebugInfo debugInfo(debugContext, "popartbuilder");
  auto opSettings = Op::Settings(graph, "", debugInfo.getId());

  for (Op *op : w.consumers.getOps()) {
    for (auto &outlineAttribute : op->settings.extraOutlineAttributes) {
      opSettings.extraOutlineAttributes.insert(outlineAttribute);
    }
  }

  if (getReplicatedGraphCount() > 1) {
    if (gradientAccumulationEnabled()) {
      if (!withMomentum || sgdAccMm == SGDAccumulatorAndMomentum::Separate) {
        reductionType = OptimizerReductionType::AccumReduce;
      } else if (sgdAccMm == SGDAccumulatorAndMomentum::Combined) {
        reductionType = OptimizerReductionType::AcclReduce;
      } else {
        throw internal_error("SGD::createOp: Unknown SGDAccumulatorAndMomentum "
                             "with int value {}",
                             static_cast<int>(sgdAccMm));
      }
    } else {
      // Disable [accl|accum]Reduce in favor of gradReduce when not using
      // gradient accumulation.
      reductionType = OptimizerReductionType::GradReduce;
    }
  }

  if (!withMomentum) {
    return std::make_unique<SGD0ComboOp>(
        slr0helper.getFromWeightId(w.id, *this),
        wdsf0helper.getFromWeightId(w.id, *this),
        gradientAccumulationEnabled(),
        reductionType,
        sgdAccumType == DataType::UNDEFINED ? w.info.getDataTypeInfo()->type()
                                            : sgdAccumType,
        opSettings);
  }

  // velocity required
  const auto swd = swd1helper.getFromWeightId(w.id, *this);

  switch (this->sgdAccMm) {
  case SGDAccumulatorAndMomentum::Combined: {
    const auto smm1  = smm1helper.getFromWeightId(w.id, *this);
    const auto dpsf1 = dpsf1helper.getFromWeightId(w.id, *this);
    const auto slr1  = slr1helper.getFromWeightId(w.id, *this);

    if (withNesterov) {
      const auto mm    = mmHelper.getFromWeightId(w.id, *this);
      const auto wd    = wdHelper.getFromWeightId(w.id, *this);
      const auto ngsf1 = ngsf1helper.getFromWeightId(w.id, *this);
      const auto ndsf1 = ndsf1helper.getFromWeightId(w.id, *this);

      return std::make_unique<SGD1ComboOp>(smm1,
                                           dpsf1,
                                           swd,
                                           slr1,
                                           mm,
                                           wd,
                                           ngsf1,
                                           ndsf1,
                                           reductionType,
                                           opSettings);
    } else {
      return std::make_unique<SGD1ComboOp>(
          smm1, dpsf1, swd, slr1, reductionType, opSettings);
    }
  }
  case SGDAccumulatorAndMomentum::Separate: {
    const auto smm2  = smm2helper.getFromWeightId(w.id, *this);
    const auto dpsf2 = dpsf2helper.getFromWeightId(w.id, *this);
    const auto slr2  = slr2helper.getFromWeightId(w.id, *this);

    if (withNesterov) {
      const auto mm    = mmHelper.getFromWeightId(w.id, *this);
      const auto wd    = wdHelper.getFromWeightId(w.id, *this);
      const auto ngsf2 = ngsf2helper.getFromWeightId(w.id, *this);
      const auto ndsf2 = ndsf2helper.getFromWeightId(w.id, *this);
      return std::make_unique<SGD2ComboOp>(
          smm2,
          dpsf2,
          swd,
          slr2,
          mm,
          wd,
          ngsf2,
          ndsf2,
          gradientAccumulationEnabled(),
          reductionType,
          sgdAccumType == DataType::UNDEFINED ? w.info.getDataTypeInfo()->type()
                                              : sgdAccumType,
          sgd2Accl1Type == DataType::UNDEFINED
              ? w.info.getDataTypeInfo()->type()
              : sgd2Accl1Type,
          opSettings);
    } else {
      return std::make_unique<SGD2ComboOp>(
          smm2,
          dpsf2,
          swd,
          slr2,
          gradientAccumulationEnabled(),
          reductionType,
          sgdAccumType == DataType::UNDEFINED ? w.info.getDataTypeInfo()->type()
                                              : sgdAccumType,
          sgd2Accl1Type == DataType::UNDEFINED
              ? w.info.getDataTypeInfo()->type()
              : sgd2Accl1Type,
          opSettings);
    }
  }
  default:
    throw internal_error(
        "SGD::createOp: Unknown SGDAccumulatorAndMomentum with int value {}",
        static_cast<int>(this->sgdAccMm));
  }
}

std::vector<TensorId> SGD::getInputIds(const Tensor &w) const {

  bool withMomentum = hasMomentum(w);

  bool withNesterov = enableNesterov(w);

  const TensorId &varId = w.id;
  std::vector<TensorId> inputs;
  if (!withMomentum) {
    inputs.resize(4, "");
  } else {
    if (withNesterov) {
      inputs.resize(10, "");
    } else {
      inputs.resize(6, "");
    }
  }

  // variable
  inputs[VarUpdateOp::getVarToUpdateInIndex()] = varId;

  // gradient
  inputs[VarUpdateWithUpdaterOp::getUpdaterInIndex()] = getGradId(varId);

  if (!withMomentum) {
    // scaled learning rate (optional)
    inputs[SGD0ComboOp::getSlr0InIndex()] =
        slr0helper.getScalarIdIfNonConst(w, *this);

    // weight decay scale factor (optional)
    inputs[SGD0ComboOp::getWdsf0InIndex()] =
        wdsf0helper.getScalarIdIfNonConst(w, *this);
  }

  // with momentum:
  else {
    // weight decay scale factor (optional)
    inputs[SGDMComboBaseOp::getSwd1InIndex()] =
        swd1helper.getScalarIdIfNonConst(w, *this);

    if (sgdAccMm == SGDAccumulatorAndMomentum::Combined) {
      // scaled learning rate (optional)
      inputs[SGDMComboBaseOp::getSlr1InIndex()] =
          slr1helper.getScalarIdIfNonConst(w, *this);

      // scaled momentum (optional)
      inputs[SGDMComboBaseOp::getSmm1InIndex()] =
          smm1helper.getScalarIdIfNonConst(w, *this);

      // dampening scale factor (optional)
      inputs[SGDMComboBaseOp::getDpsf1InIndex()] =
          dpsf1helper.getScalarIdIfNonConst(w, *this);

      if (withNesterov) {
        // nesterov dampening scale factor (optional)
        inputs[SGDMComboBaseOp::getNdsfInIndex()] =
            ndsf1helper.getScalarIdIfNonConst(w, *this);

        // nesterov gradient scale factor (optional)
        inputs[SGDMComboBaseOp::getNgsfInIndex()] =
            ngsf1helper.getScalarIdIfNonConst(w, *this);
      }
    } else {
      // scaled learning rate (optional)
      inputs[SGDMComboBaseOp::getSlr1InIndex()] =
          slr2helper.getScalarIdIfNonConst(w, *this);

      // scaled momentum (optional)
      inputs[SGDMComboBaseOp::getSmm1InIndex()] =
          smm2helper.getScalarIdIfNonConst(w, *this);

      // dampening scale factor (optional)
      inputs[SGDMComboBaseOp::getDpsf1InIndex()] =
          dpsf2helper.getScalarIdIfNonConst(w, *this);

      if (withNesterov) {
        // nesterov dampening scale factor (optional)
        inputs[SGDMComboBaseOp::getNdsfInIndex()] =
            ndsf2helper.getScalarIdIfNonConst(w, *this);

        // nesterov gradient scale factor (optional)
        inputs[SGDMComboBaseOp::getNgsfInIndex()] =
            ngsf2helper.getScalarIdIfNonConst(w, *this);
      }
    }
    if (withNesterov) {
      // momentum (optional)
      inputs[SGDMComboBaseOp::getMmInIndex()] =
          mmHelper.getScalarIdIfNonConst(w, *this);

      // weight decay (optional)
      inputs[SGDMComboBaseOp::getWdInIndex()] =
          wdHelper.getScalarIdIfNonConst(w, *this);
    }
  }

  return inputs;
}

std::vector<std::tuple<TensorId, TensorInfo>>
SGD::getOptimizerInputs(const Tensor &weight) const {

  bool withMomentum = hasMomentum(weight);

  bool withNesterov = enableNesterov(weight);

  std::vector<TensorId> ids;
  if (!withMomentum) {
    ids.push_back(slr0helper.getScalarIdIfNonConst(weight, *this));
    ids.push_back(wdsf0helper.getScalarIdIfNonConst(weight, *this));
  } else {
    ids.push_back(swd1helper.getScalarIdIfNonConst(weight, *this));
    if (sgdAccMm == SGDAccumulatorAndMomentum::Combined) {
      ids.push_back(slr1helper.getScalarIdIfNonConst(weight, *this));
      ids.push_back(smm1helper.getScalarIdIfNonConst(weight, *this));
      ids.push_back(dpsf1helper.getScalarIdIfNonConst(weight, *this));
      if (withNesterov) {
        ids.push_back(ndsf1helper.getScalarIdIfNonConst(weight, *this));
        ids.push_back(ngsf1helper.getScalarIdIfNonConst(weight, *this));
      }
    } else {
      ids.push_back(slr2helper.getScalarIdIfNonConst(weight, *this));
      ids.push_back(smm2helper.getScalarIdIfNonConst(weight, *this));
      ids.push_back(dpsf2helper.getScalarIdIfNonConst(weight, *this));
      if (withNesterov) {
        ids.push_back(ndsf2helper.getScalarIdIfNonConst(weight, *this));
        ids.push_back(ngsf2helper.getScalarIdIfNonConst(weight, *this));
      }
    }
    if (withNesterov) {
      ids.push_back(mmHelper.getScalarIdIfNonConst(weight, *this));
      ids.push_back(wdHelper.getScalarIdIfNonConst(weight, *this));
    }
  }

  std::vector<std::tuple<TensorId, TensorInfo>> optInputs;
  for (const auto &id : ids) {
    // empty denotes const, not an input
    if ((withMomentum && (smm1helper.idMatch(id) || smm2helper.idMatch(id) ||
                          mmHelper.idMatch(id))) ||
        wdsf0helper.idMatch(id)) {
      // Use weight dtype for momentum and weight decay, Float32 for everything
      // else.
      auto tuppy = std::make_tuple(id, TensorInfo(weight.info.dataType(), {}));
      optInputs.push_back(tuppy);
    } else if (!id.empty()) {
      auto tuppy = std::make_tuple(id, TensorInfo(DataType::FLOAT, {}));
      optInputs.push_back(tuppy);
    }
  }

  return optInputs;
}

void SGD::setTensorData(Tensor &optTensor) const {
  const auto &info   = optTensor.info;
  float storedValue  = getStoredValue(optTensor.id);
  auto convertedData = convertFloatToDataType(info.dataType(), storedValue);

  logging::ir::trace(
      "Setting TensorData for {} to {}", optTensor.str(), storedValue);
  optTensor.setTensorDataByEmplaceOf(std::move(convertedData));
}

void SGD::resetTensorData(Tensor &optTensor) const {
  const auto &info   = optTensor.info;
  float storedValue  = getStoredValue(optTensor.id);
  auto convertedData = convertFloatToDataType(info.dataType(), storedValue);
  logging::ir::trace(
      "Resetting TensorData for {} to {}", optTensor.str(), storedValue);
  optTensor.tensorData()->resetData(info, convertedData.data());
}

float SGD::getStoredValue(const TensorId &optId) const {

  if (optId.find(reservedLossScalingPrefix()) != std::string::npos) {
    return getFinalLossScalingVal();
  }

  if (slr0helper.idMatch(optId)) {
    return slr0helper.getFromScalarId(optId, *this).val();
  }

  if (wdsf0helper.idMatch(optId)) {
    return wdsf0helper.getFromScalarId(optId, *this).val();
  }

  if (slr1helper.idMatch(optId)) {
    return slr1helper.getFromScalarId(optId, *this).val();
  }

  if (swd1helper.idMatch(optId)) {
    return swd1helper.getFromScalarId(optId, *this).val();
  }

  if (dpsf1helper.idMatch(optId)) {
    return dpsf1helper.getFromScalarId(optId, *this).val();
  }

  if (smm1helper.idMatch(optId)) {
    return smm1helper.getFromScalarId(optId, *this).val();
  }

  if (slr2helper.idMatch(optId)) {
    return slr2helper.getFromScalarId(optId, *this).val();
  }

  if (dpsf2helper.idMatch(optId)) {
    return dpsf2helper.getFromScalarId(optId, *this).val();
  }

  if (smm2helper.idMatch(optId)) {
    return smm2helper.getFromScalarId(optId, *this).val();
  }

  if (mmHelper.idMatch(optId)) {
    return mmHelper.getFromScalarId(optId, *this).val();
  }

  if (wdHelper.idMatch(optId)) {
    return wdHelper.getFromScalarId(optId, *this).val();
  }

  if (ngsf1helper.idMatch(optId)) {
    return ngsf1helper.getFromScalarId(optId, *this).val();
  }

  if (ngsf2helper.idMatch(optId)) {
    return ngsf2helper.getFromScalarId(optId, *this).val();
  }

  if (ndsf1helper.idMatch(optId)) {
    return ndsf1helper.getFromScalarId(optId, *this).val();
  }

  if (ndsf2helper.idMatch(optId)) {
    return ndsf2helper.getFromScalarId(optId, *this).val();
  }

  throw runtime_error("In getStoredValue for {}, it doesn't match any existing "
                      "optimizer prefix",
                      optId);
}

void SGD::validReplacement(const Optimizer &other) const {
  Optimizer::validReplacement(other);

  auto asSgd = dynamic_cast<const SGD *>(&other);
  if (!asSgd) {
    throw internal_error(
        "other has same `type' as this SGD, but cannot be "
        "dynamically cast to SGD. Has there been a redesign of the "
        "optimizer classes? if so this needs a rethink");
  }

  checkReplacementValue(lossScaling(), other.lossScaling(), "loss scaling");
  checkReplacementValue(lrs, asSgd->lrs, "learning rates");
  checkReplacementValue(wds, asSgd->wds, "weight decays");
  checkReplacementValue(mms, asSgd->mms, "momentums");
  checkReplacementValue(vss, asSgd->vss, "velocity scalings");
  checkReplacementValue(nts, asSgd->nts, "nesterov");

  auto hasVelocityTensor = [](const SGD *sgd) {
    return !sgd->mms.getDefault().isConst() ||
           sgd->mms.getDefault().val() != 0.0f;
  };

  auto thisHasVelocityTensor = hasVelocityTensor(this);

  if (thisHasVelocityTensor != hasVelocityTensor(asSgd)) {
    throw optimizer_replacement_error(
        "this does not require a velocity tensor, but other does.");
  }

  if (thisHasVelocityTensor) {
    if (sgdAccMm != asSgd->sgdAccMm) {
      throw optimizer_replacement_error(
          "this has SGDAccumulatorAndMomentum {}, but other has {}",
          sgdAccMm,
          asSgd->sgdAccMm);
    }

    if (sgdAccMm == SGDAccumulatorAndMomentum::Separate) {
      if (sgdAccumType != asSgd->sgdAccumType) {
        throw optimizer_replacement_error(
            "this has SGDAccumulatorAndMomentum::Separate and sgdAccumType "
            "{}, but other has {}",
            sgdAccumType,
            asSgd->sgdAccumType);
      }

      if (sgd2Accl1Type != asSgd->sgd2Accl1Type) {
        throw optimizer_replacement_error(
            "this has SGDAccumulatorAndMomentum::Separate and sgd2Accl1Type "
            "{}, but other has {}",
            sgd2Accl1Type,
            asSgd->sgd2Accl1Type);
      }
    }
  }
}

std::unique_ptr<Optimizer> SGD::clone() const {
  return std::make_unique<SGD>(*this);
}

TensorId SGD::getInverseLossScalingTensorId(const Tensor &weight) const {
  if (hasMomentum(weight)) {
    return getInputIds(weight).at(SGDMComboBaseOp::getDpsf1InIndex());
  } else {
    return getInputIds(weight).at(SGD0ComboOp::getSlr0InIndex());
  }
}

std::ostream &operator<<(std::ostream &os,
                         const SGDAccumulatorAndMomentum &sgdAccMm) {
  switch (sgdAccMm) {
  case SGDAccumulatorAndMomentum::Combined:
    return os << "SGDAccumulatorAndMomentum::Combined";
  case SGDAccumulatorAndMomentum::Separate:
    return os << "SGDAccumulatorAndMomentum::Separate";
  default:
    throw internal_error(
        "Missing case for SGDAccumulatorAndMomentum enum with int value {}",
        static_cast<int>(sgdAccMm));
  }
}

size_t SGD::hash() const {
  std::size_t seed = 0;
  boost::hash_combine(seed, Optimizer::hash());
  boost::hash_combine(seed, static_cast<int>(type()));
  boost::hash_combine(seed, lrs);
  boost::hash_combine(seed, wds);
  boost::hash_combine(seed, mms);
  boost::hash_combine(seed, dps);
  boost::hash_combine(seed, vss);
  boost::hash_combine(seed, nts);

  bool hasVelocityTensor =
      !mms.getDefault().isConst() || mms.getDefault().val() != 0.0f;
  boost::hash_combine(seed, hasVelocityTensor);

  if (hasVelocityTensor) {
    boost::hash_combine(seed, sgdAccMm);

    if (sgdAccMm == SGDAccumulatorAndMomentum::Separate) {
      boost::hash_combine(seed, sgdAccumType);
      boost::hash_combine(seed, sgd2Accl1Type);
    }
  }

  return seed;
}

} // namespace popart
