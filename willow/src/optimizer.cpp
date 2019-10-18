#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/op/sgd1varupdatecombo.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>

namespace popart {

namespace {
std::map<std::string, OptimizerValue>
getOptMap(const std::map<std::string, std::pair<float, bool>> &m) {
  std::map<std::string, OptimizerValue> mOptVals;
  for (auto x : m) {
    mOptVals.insert({x.first, OptimizerValue(x.second)});
  }
  return mOptVals;
}

} // namespace

SGD SGD::fromDefaultMap(const std::map<std::string, OptimizerValue> &m) {
  return SGD(getComplete(m), 1011);
}

namespace {
const std::vector<std::string> &getSpecificNames() {
  const static std::vector<std::string> names{"learningRate",
                                              "weightDecay",
                                              "momentum",
                                              "dampening",
                                              "velocityScaling"};
  return names;
}

} // namespace

SGD::SGD(const std::map<std::string, std::pair<float, bool>> &m)
    : SGD(getComplete(getOptMap(m)), 31415) {}

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
                 complete.at("velocityScaling"));
}

TensorId Optimizer::getLossScalingTensorId(DataType t) {
  return reservedLossScalingPrefix() + getDataTypeInfoMap().at(t).name();
}

Optimizer::Optimizer(OptimizerValue ls_) : ls(ls_) {
  // Reject loss scaling of 0.
  if (!(ls.val() > 0.0f || ls.val() < 0.0f)) {
    throw error("Loss scaling cannot be 0");
  }
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

  if (counter != 0 && counter != getSpecificNames().size()) {
    throw error("Inconsistency in SGD::hasSpecific : there should either be a "
                "specific value for ALL (5) or NO (0) atomic scalar values, "
                "not {} of them. ",
                counter);
  }

  return counter > 0;
}

bool SGD::requiresAccl(const Tensor &weight,
                       bool gradAcclEnabled,
                       int64_t gradAcclFactor) const {

  // might be used in the future
  (void)gradAcclFactor;

  OptimizerValue mm = mms.get(weight.id);

  bool doesRequire = gradAcclEnabled || !mm.isConst() || mm.val() != 0.0f;

  logging::ir::trace(
      "{} does require grad accl ? {} mm.isConst ? {} mm.val ? {} ",
      weight.id,
      doesRequire,
      mm.isConst(),
      mm.val());

  return doesRequire;
}

void SGD::insertSpecific(const TensorId &id,
                         OptimizerValue lr,
                         OptimizerValue wd,
                         OptimizerValue mm,
                         OptimizerValue dp,
                         OptimizerValue vs) {

  lrs.insertSpecific(id, lr);
  wds.insertSpecific(id, wd);
  mms.insertSpecific(id, mm);
  dps.insertSpecific(id, dp);
  vss.insertSpecific(id, vs);

  runValueChecks(lr.val(), wd.val(), mm.val(), dp.val(), vs.val());
}

void SGD::runValueChecks(float lr,
                         float wd,
                         float mm,
                         float dp,
                         float vs) const {
  if (lr <= 0) {
    throw error("Non-positive learning rate ({}) in SGD, bailing as this might "
                "be a user error.",
                lr);
  }

  if (wd < 0) {
    throw error("Negative weight decay ({}) in SGD, bailing as this might be a "
                "user error",
                wd);
  }

  if (mm < 0) {
    throw error(
        "Negative momentum ({}) in SGD, bailing as this might be a user error",
        mm);
  }

  if (dp < 0) {
    throw error(
        "Negative dampening ({}) in SGD, bailing as this might be a user error",
        dp);
  }

  if (vs <= 0) {
    throw error("Non-positive velocity scaling ({}) in SGD is not supported",
                vs);
  }
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

SGD::SGD(const std::map<std::string, OptimizerValue> &cmap, int)
    : SGD(cmap.at("defaultLearningRate"),
          cmap.at("defaultWeightDecay"),
          cmap.at("defaultMomentum"),
          cmap.at("defaultDampening"),
          cmap.at("defaultVelocityScaling"),
          cmap.at("lossScaling")) {}

SGD::SGD(OptimizerValue lr,
         OptimizerValue wd,
         OptimizerValue mm,
         OptimizerValue dp,
         OptimizerValue vs,
         OptimizerValue ls)
    : Optimizer(ls), lrs(lr), wds(wd), mms(mm), dps(dp), vss(vs) {
  runValueChecks(lr.val(), wd.val(), mm.val(), dp.val(), vs.val());
}

OptimizerValue SGD::getLossScalingOrDefault(
    const std::map<std::string, OptimizerValue> &m) const {
  auto found = m.find("lossScaling");
  if (found != m.end()) {
    return found->second;
  }
  return {1, true};
}

std::map<std::string, OptimizerValue>
SGD::getComplete(const std::map<std::string, OptimizerValue> &m) {

  std::vector<std::string> sixParamArgs{"defaultLearningRate",
                                        "defaultWeightDecay",
                                        "defaultMomentum",
                                        "defaultDampening",
                                        "defaultVelocityScaling",
                                        "lossScaling"};

  std::map<std::string, OptimizerValue> complete{};

  complete.insert({"defaultLearningRate", getUnsetLearningRate()});
  complete.insert({"defaultWeightDecay", getUnsetWeightDecay()});
  complete.insert({"defaultMomentum", getUnsetMomentum()});
  complete.insert({"defaultDampening", getUnsetDampening()});
  complete.insert({"defaultVelocityScaling", getUnsetVelocityScaling()});
  complete.insert({"lossScaling", getUnsetLossScaling()});

  for (auto key_val : m) {
    auto key = key_val.first;
    auto val = key_val.second;
    if (std::find(sixParamArgs.cbegin(), sixParamArgs.cend(), key) ==
        sixParamArgs.cend()) {
      std::ostringstream oss;
      oss << "Invalid SGD key, " << key << ", the allowed keys are ( ";
      for (auto x : sixParamArgs) {
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

  const auto &sessionOptions = graph.getIr().getSessionOptions();

  bool withAccl = requiresAccl(w,
                               sessionOptions.enableGradientAccumulation,
                               sessionOptions.accumulationFactor);

  auto opSettings = Op::Settings(graph, "");

  if (!withAccl) {
    return std::make_unique<SGD0VarUpdateOp>(
        w.id,
        slr0helper.getFromWeightId(w.id, *this),
        wdsf0helper.getFromWeightId(w.id, *this),
        opSettings);
  }

  // velocity required
  return std::make_unique<SGD1VarUpdateComboOp>(
      w.id,
      mm1helper.getFromWeightId(w.id, *this),
      dpsf1helper.getFromWeightId(w.id, *this),
      wdsf1helper.getFromWeightId(w.id, *this),
      slr1helper.getFromWeightId(w.id, *this),
      opSettings);
}

std::vector<TensorId> SGD::getInputIds(const Tensor &w,
                                       bool enableGradientAccumulation,
                                       int64_t accumulationFactor) const {

  bool withAccl =
      requiresAccl(w, enableGradientAccumulation, accumulationFactor);

  const TensorId &varId = w.id;
  std::vector<TensorId> inputs;
  if (!withAccl) {
    inputs.resize(4, "");
  } else {
    inputs.resize(6, "");
  }

  // variable
  inputs[VarUpdateOp::getVarToUpdateInIndex()] = varId;

  // gradient
  inputs[VarUpdateOp::getUpdaterInIndex()] = getGradId(varId);

  if (!withAccl) {
    // scaled learning rate (optional)
    inputs[SGD0VarUpdateOp::getSlr0InIndex()] =
        slr0helper.getScalarIdIfNonConst(w, *this);

    // weight decay scale factor (optional)
    inputs[SGD0VarUpdateOp::getWdsf0InIndex()] =
        wdsf0helper.getScalarIdIfNonConst(w, *this);
  }

  // with accumulation:
  else {

    // momentum (optional)
    inputs[SGD1VarUpdateComboOp::getMm1InIndex()] =
        mm1helper.getScalarIdIfNonConst(w, *this);

    // dampening scale factor (optional)
    inputs[SGD1VarUpdateComboOp::getDpsf1InIndex()] =
        dpsf1helper.getScalarIdIfNonConst(w, *this);

    // weight decay scale factor (optional)
    inputs[SGD1VarUpdateComboOp::getWdsf1InIndex()] =
        wdsf1helper.getScalarIdIfNonConst(w, *this);

    // scaled learning rate (optional)
    inputs[SGD1VarUpdateComboOp::getSlr1InIndex()] =
        slr1helper.getScalarIdIfNonConst(w, *this);
  }

  return inputs;
}

std::vector<std::tuple<TensorId, TensorInfo>>
SGD::getOptimizerInputs(const Tensor &weight,
                        bool enableGradientAccumulation,
                        int64_t accumulationFactor) const {

  bool withAccl =
      requiresAccl(weight, enableGradientAccumulation, accumulationFactor);

  std::vector<TensorId> ids;
  if (!withAccl) {
    ids.push_back(slr0helper.getScalarIdIfNonConst(weight, *this));
    ids.push_back(wdsf0helper.getScalarIdIfNonConst(weight, *this));
  } else {
    ids.push_back(slr1helper.getScalarIdIfNonConst(weight, *this));
    ids.push_back(wdsf1helper.getScalarIdIfNonConst(weight, *this));
    ids.push_back(mm1helper.getScalarIdIfNonConst(weight, *this));
    ids.push_back(dpsf1helper.getScalarIdIfNonConst(weight, *this));
  }

  std::vector<std::tuple<TensorId, TensorInfo>> optInputs;
  for (const auto &id : ids) {
    // empty denotes const, not an input
    if (!id.empty()) {

      auto tuppy = std::make_tuple(id, TensorInfo(weight.info.dataType(), {}));
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
  optTensor.setTensorData(info, convertedData.data());
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
    return lossScaling().val();
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

  if (wdsf1helper.idMatch(optId)) {
    return wdsf1helper.getFromScalarId(optId, *this).val();
  }

  if (dpsf1helper.idMatch(optId)) {
    return dpsf1helper.getFromScalarId(optId, *this).val();
  }

  if (mm1helper.idMatch(optId)) {
    return mm1helper.getFromScalarId(optId, *this).val();
  }

  throw error("In getStoredValue for {}, it doesn't match any existing "
              "optimizer prefix",
              optId);
}

bool SGD::validReplacement(const Optimizer &other) const {
  if (other.type() != type()) {
    return false;
  }

  auto asSgd = dynamic_cast<const SGD *>(&other);
  if (!asSgd) {
    throw error("ILE: other has same `type' as this SGD, but cannot be "
                "dynamically cast to SGD. Has there been a redesign of the "
                "optimizer classes? if so this needs a rethink");
  }

  logging::ir::debug("Checking loss scaling for compatibility");
  if (!lossScaling().validReplacement(other.lossScaling())) {
    return false;
  }

  logging::ir::debug("Checking learning rates for compatibility");
  if (!lrs.validReplacement(asSgd->lrs)) {
    return false;
  }

  logging::ir::debug("Checking weight decays for compatibility");
  if (!wds.validReplacement(asSgd->wds)) {
    return false;
  }

  logging::ir::debug("Checking momentums for compatibility");
  if (!mms.validReplacement(asSgd->mms)) {
    return false;
  }

  logging::ir::debug("Checking velocity scalings for compatibility");
  if (!vss.validReplacement(asSgd->vss)) {
    return false;
  }

  return true;
}

std::unique_ptr<Optimizer> SGD::clone() const {
  return std::make_unique<SGD>(*this);
}

} // namespace popart
