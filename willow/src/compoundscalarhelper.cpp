// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <string>
#include <popart/adam.hpp>
#include <popart/compoundscalarhelper.hpp>
#include <popart/op.hpp>
#include <popart/optimizer.hpp>

namespace popart {

template class CompoundScalarHelper<SGD>;
template class CompoundScalarHelper<Adam>;

template <class T>
bool CompoundScalarHelper<T>::idMatch(const TensorId &optId) const {
  return (optId.find(specificPrefix()) != std::string::npos) ||
         (optId.find(defaultPrefix()) != std::string::npos);
}

template bool CompoundScalarHelper<SGD>::idMatch(const TensorId &optId) const;
template bool CompoundScalarHelper<Adam>::idMatch(const TensorId &optId) const;

template <class T>
OptimizerValue
CompoundScalarHelper<T>::getFromWeightId(const TensorId &weightId,
                                         const T &sgd) const {
  return {val(weightId, sgd), isConst(weightId, sgd)};
}

template OptimizerValue
CompoundScalarHelper<SGD>::getFromWeightId(const TensorId &weightId,
                                           const SGD &sgd) const;
template OptimizerValue
CompoundScalarHelper<Adam>::getFromWeightId(const TensorId &weightId,
                                            const Adam &sgd) const;

template <class T>
OptimizerValue CompoundScalarHelper<T>::getFromScalarId(const TensorId &optId,
                                                        const T &sgd) const {
  // the Optimizer Tensor is specific to a weight
  if (optId.find(specificPrefix()) != std::string::npos) {
    return getFromWeightId(getWeightId(optId), sgd);
  }

  else if (optId.find(defaultPrefix()) != std::string::npos) {
    return getFromWeightId("fudgeCakeSpaghettiBonanza", sgd);
  }

  throw internal_error("failed to determine optimizer type from id {}", optId);
}

template OptimizerValue
CompoundScalarHelper<SGD>::getFromScalarId(const TensorId &optId,
                                           const SGD &sgd) const;
template OptimizerValue
CompoundScalarHelper<Adam>::getFromScalarId(const TensorId &optId,
                                            const Adam &adam) const;
template <class T>
TensorId CompoundScalarHelper<T>::getScalarId(const Tensor &w,
                                              const T &sgd) const {
  if (sgd.hasSpecific(w)) {
    return specificPrefix() + w.id;
  }
  return defaultPrefix() + w.info.data_type();
}

template TensorId CompoundScalarHelper<SGD>::getScalarId(const Tensor &w,
                                                         const SGD &sgd) const;
template TensorId
CompoundScalarHelper<Adam>::getScalarId(const Tensor &w,
                                        const Adam &adam) const;

template <class T>
TensorId CompoundScalarHelper<T>::getScalarIdIfNonConst(const Tensor &w,
                                                        const T &sgd) const {
  return isConst(w.id, sgd) ? "" : getScalarId(w, sgd);
}

template TensorId
CompoundScalarHelper<SGD>::getScalarIdIfNonConst(const Tensor &w,
                                                 const SGD &sgd) const;
template TensorId
CompoundScalarHelper<Adam>::getScalarIdIfNonConst(const Tensor &w,
                                                  const Adam &adam) const;

// remove specific prefix to obtain the TensorId of the weight
template <class T>
TensorId CompoundScalarHelper<T>::getWeightId(const TensorId &scalarId) const {
  if (scalarId.find(specificPrefix()) != std::string::npos) {
    return std::string(scalarId.begin() + specificPrefix().size(),
                       scalarId.end());
  }

  throw internal_error("CompoundScalarHelper::getWeightId(.) : failed to find "
                       "substring {} in {}",
                       specificPrefix(),
                       scalarId);
}

template TensorId
CompoundScalarHelper<SGD>::getWeightId(const TensorId &scalarId) const;
template TensorId
CompoundScalarHelper<Adam>::getWeightId(const TensorId &scalarId) const;

float WeightDecayScaleFactor0Helper::val(const TensorId &weightId,
                                         const SGD &sgd) const {
  auto wd = sgd.weightDecays().get(weightId).val();
  auto lr = sgd.learningRates().get(weightId).val();
  auto dp = sgd.dampenings().get(weightId).val();
  return val(wd, lr, dp);
}

bool WeightDecayScaleFactor0Helper::isConst(const TensorId &weightId,
                                            const SGD &sgd) const {
  auto wd = sgd.weightDecays().get(weightId);
  auto lr = sgd.learningRates().get(weightId);
  auto dp = sgd.dampenings().get(weightId);
  return wd.isConst() && lr.isConst() && dp.isConst();
}

float ScaledLearningRate0Helper::val(const TensorId &weightId,
                                     const SGD &sgd) const {
  auto lr = sgd.learningRates().get(weightId).val();
  auto ls = sgd.lossScaling().val();
  auto dp = sgd.dampenings().get(weightId).val();
  return val(lr, ls, dp);
}

bool ScaledLearningRate0Helper::isConst(const TensorId &weightId,
                                        const SGD &sgd) const {
  auto lr = sgd.learningRates().get(weightId);
  auto ls = sgd.lossScaling();
  auto dp = sgd.dampenings().get(weightId);
  return lr.isConst() && ls.isConst() && dp.isConst();
}

float ScaledMomentum1Helper::val(const TensorId &weightId,
                                 const SGD &sgd) const {
  auto mm = sgd.momentums().get(weightId).val();
  return val(mm,
             sgd.gradientAccumulationEnabled() ? sgd.getReplicatedGraphCount()
                                               : 1);
}

bool ScaledMomentum1Helper::isConst(const TensorId &weightId,
                                    const SGD &sgd) const {
  auto mm = sgd.momentums().get(weightId);
  return mm.isConst();
}

float ScaledLearningRate1Helper::val(const TensorId &weightId,
                                     const SGD &sgd) const {
  auto lr = sgd.learningRates().get(weightId).val();
  auto vs = sgd.velocityScalings().get(weightId).val();
  return val(lr,
             vs,
             sgd.gradientAccumulationEnabled() ? sgd.getReplicatedGraphCount()
                                               : 1);
}

bool ScaledLearningRate1Helper::isConst(const TensorId &weightId,
                                        const SGD &sgd) const {
  auto lr = sgd.learningRates().get(weightId);
  auto vs = sgd.velocityScalings().get(weightId);
  return lr.isConst() && vs.isConst();
}

float DampeningScaleFactor1Helper::val(const TensorId &weightId,
                                       const SGD &sgd) const {
  auto dm = sgd.dampenings().get(weightId).val();
  auto vs = sgd.velocityScalings().get(weightId).val();
  auto ls = sgd.lossScaling().val();
  return val(dm,
             vs,
             ls,
             sgd.gradientAccumulationEnabled() ? sgd.getReplicatedGraphCount()
                                               : 1);
}

bool DampeningScaleFactor1Helper::isConst(const TensorId &weightId,
                                          const SGD &sgd) const {
  auto dm = sgd.dampenings().get(weightId);
  auto vs = sgd.velocityScalings().get(weightId);
  auto ls = sgd.lossScaling();
  return dm.isConst() && vs.isConst() && ls.isConst();
}

float ScaledWeightDecay1Helper::val(const TensorId &weightId,
                                    const SGD &sgd) const {
  auto dm = sgd.dampenings().get(weightId).val();
  auto wd = sgd.weightDecays().get(weightId).val();
  auto vs = sgd.velocityScalings().get(weightId).val();
  return val(dm, wd, vs);
}

bool ScaledWeightDecay1Helper::isConst(const TensorId &weightId,
                                       const SGD &sgd) const {
  auto dm = sgd.dampenings().get(weightId);
  auto wd = sgd.weightDecays().get(weightId);
  auto vs = sgd.velocityScalings().get(weightId);
  return dm.isConst() && wd.isConst() && vs.isConst();
}

float AdamBeta1Helper::val(const TensorId &weightId, const Adam &adam) const {
  auto b1 = adam.beta1s().get(weightId).val();
  return val(b1);
}

bool AdamBeta1Helper::isConst(const TensorId &weightId,
                              const Adam &adam) const {
  return adam.beta1s().get(weightId).isConst();
}

float AdamBeta2Helper::val(const TensorId &weightId, const Adam &adam) const {
  auto b2 = adam.beta2s().get(weightId).val();
  return val(b2);
}

bool AdamBeta2Helper::isConst(const TensorId &weightId,
                              const Adam &adam) const {
  return adam.beta2s().get(weightId).isConst();
}

float AdamLearningRateHelper::val(const TensorId &weightId,
                                  const Adam &adam) const {
  auto wd = adam.learningRates().get(weightId).val();
  return val(wd);
}

bool AdamLearningRateHelper::isConst(const TensorId &weightId,
                                     const Adam &adam) const {
  return adam.learningRates().get(weightId).isConst();
}

float AdamWeightDecayHelper::val(const TensorId &weightId,
                                 const Adam &adam) const {
  auto wd = adam.weightDecays().get(weightId).val();
  return val(wd);
}

bool AdamWeightDecayHelper::isConst(const TensorId &weightId,
                                    const Adam &adam) const {
  return adam.weightDecays().get(weightId).isConst();
}

float AdamEpsHelper::val(const TensorId &weightId, const Adam &adam) const {
  auto wd = adam.epss().get(weightId).val();
  return val(wd);
}

bool AdamEpsHelper::isConst(const TensorId &weightId, const Adam &adam) const {
  return adam.epss().get(weightId).isConst();
}

float AdamLossScalingHelper::val(const TensorId &weightId,
                                 const Adam &adam) const {
  auto wd = adam.lossScaling().val();
  return val(wd);
}

bool AdamLossScalingHelper::isConst(const TensorId &weightId,
                                    const Adam &adam) const {
  return adam.lossScaling().isConst();
}

} // namespace popart
