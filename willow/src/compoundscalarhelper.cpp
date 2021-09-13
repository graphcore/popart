// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <string>
#include <popart/adam.hpp>
#include <popart/adaptive.hpp>
#include <popart/compoundscalarhelper.hpp>
#include <popart/op.hpp>
#include <popart/sgd.hpp>

namespace popart {

template class CompoundScalarHelper<SGD>;
template class CompoundScalarHelper<Adam>;
template class CompoundScalarHelper<Adaptive>;

template <class T>
bool CompoundScalarHelper<T>::idMatch(const TensorId &optId) const {
  return (optId.find(specificPrefix()) != std::string::npos) ||
         (optId.find(defaultPrefix()) != std::string::npos);
}

template bool CompoundScalarHelper<SGD>::idMatch(const TensorId &optId) const;
template bool CompoundScalarHelper<Adam>::idMatch(const TensorId &optId) const;
template bool
CompoundScalarHelper<Adaptive>::idMatch(const TensorId &optId) const;

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
                                            const Adam &adam) const;
template OptimizerValue
CompoundScalarHelper<Adaptive>::getFromWeightId(const TensorId &weightId,
                                                const Adaptive &adaptive) const;

template <class T>
OptimizerValue CompoundScalarHelper<T>::getFromScalarId(const TensorId &optId,
                                                        const T &sgd) const {
  // the Optimizer Tensor is specific to a weight
  if (optId.find(specificPrefix()) != std::string::npos) {
    return getFromWeightId(getWeightId(optId), sgd);
  }

  else if (optId.find(defaultPrefix()) != std::string::npos) {
    return getFromWeightId("defaultCompoundScalarPrefix", sgd);
  }

  throw internal_error("failed to determine optimizer type from id {}", optId);
}

template OptimizerValue
CompoundScalarHelper<SGD>::getFromScalarId(const TensorId &optId,
                                           const SGD &sgd) const;
template OptimizerValue
CompoundScalarHelper<Adam>::getFromScalarId(const TensorId &optId,
                                            const Adam &adam) const;
template OptimizerValue
CompoundScalarHelper<Adaptive>::getFromScalarId(const TensorId &optId,
                                                const Adaptive &adaptive) const;

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
template TensorId
CompoundScalarHelper<Adaptive>::getScalarId(const Tensor &w,
                                            const Adaptive &adaptive) const;

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
template TensorId CompoundScalarHelper<Adaptive>::getScalarIdIfNonConst(
    const Tensor &w,
    const Adaptive &adaptive) const;

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
template TensorId
CompoundScalarHelper<Adaptive>::getWeightId(const TensorId &scalarId) const;

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
  float lr = sgd.learningRates().get(weightId).val();
  float dp = sgd.dampenings().get(weightId).val();
  float ls = sgd.lossScaling().val();
  float af =
      sgd.postMeanAccumulationEnabled() ? sgd.getAccumulationFactor() : 1;
  float rf =
      sgd.postMeanReplicationEnabled() ? sgd.getReplicatedGraphCount() : 1;
  return val(lr, dp, ls, af, rf);
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
  float mm = sgd.momentums().get(weightId).val();
  float rf =
      sgd.gradientAccumulationEnabled() ? sgd.getReplicatedGraphCount() : 1.0f;
  return val(mm, rf);
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
  float rf =
      sgd.gradientAccumulationEnabled() ? sgd.getReplicatedGraphCount() : 1.0f;
  return val(lr, vs, rf);
}

bool ScaledLearningRate1Helper::isConst(const TensorId &weightId,
                                        const SGD &sgd) const {
  auto lr = sgd.learningRates().get(weightId);
  auto vs = sgd.velocityScalings().get(weightId);
  return lr.isConst() && vs.isConst();
}

float DampeningScaleFactor1Helper::val(const TensorId &weightId,
                                       const SGD &sgd) const {
  float dm = sgd.dampenings().get(weightId).val();
  float vs = sgd.velocityScalings().get(weightId).val();
  float ls = sgd.lossScaling().val();
  float af = sgd.meanReductionEnabled() ? sgd.getAccumulationFactor() : 1.0f;
  float rf = 1.0f;
  if (sgd.gradientAccumulationEnabled() &&
      (!sgd.meanReductionEnabled() || sgd.lossMeanReplicationEnabled())) {
    // Gradient Accumulation with ReductionType::Sum or
    // MeanReductionStrategy::PostAndLoss
    rf = sgd.getReplicatedGraphCount();
  } else if (!sgd.gradientAccumulationEnabled() &&
             (sgd.meanReductionEnabled() &&
              !sgd.lossMeanReplicationEnabled())) {
    // No Gradient Accumulation without MeanReductionStrategy::PostAndLoss
    rf = 1.0f / static_cast<float>(sgd.getReplicatedGraphCount());
  }
  return val(dm, vs, ls, af, rf);
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

float ScaledLearningRate2Helper::val(const TensorId &weightId,
                                     const SGD &sgd) const {
  auto lr = sgd.learningRates().get(weightId).val();
  auto vs = sgd.velocityScalings().get(weightId).val();
  return val(lr, vs);
}

bool ScaledLearningRate2Helper::isConst(const TensorId &weightId,
                                        const SGD &sgd) const {
  auto lr = sgd.learningRates().get(weightId);
  auto vs = sgd.velocityScalings().get(weightId);
  return lr.isConst() && vs.isConst();
}

float ScaledMomentum2Helper::val(const TensorId &weightId,
                                 const SGD &sgd) const {
  float mm = sgd.momentums().get(weightId).val();
  return val(mm);
}

bool ScaledMomentum2Helper::isConst(const TensorId &weightId,
                                    const SGD &sgd) const {
  auto mm = sgd.momentums().get(weightId);
  return mm.isConst();
}

float DampeningScaleFactor2Helper::val(const TensorId &weightId,
                                       const SGD &sgd) const {
  float dm = sgd.dampenings().get(weightId).val();
  float vs = sgd.velocityScalings().get(weightId).val();
  float ls = sgd.lossScaling().val();
  float af =
      sgd.postMeanAccumulationEnabled() ? sgd.getAccumulationFactor() : 1;
  float rf =
      sgd.postMeanReplicationEnabled() ? sgd.getReplicatedGraphCount() : 1;
  return val(dm, vs, ls, af, rf);
}

bool DampeningScaleFactor2Helper::isConst(const TensorId &weightId,
                                          const SGD &sgd) const {
  auto dm = sgd.dampenings().get(weightId);
  auto vs = sgd.velocityScalings().get(weightId);
  auto ls = sgd.lossScaling();
  return dm.isConst() && vs.isConst() && ls.isConst();
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
  auto lr = adam.learningRates().get(weightId).val();
  return val(lr);
}

bool AdamLearningRateHelper::isConst(const TensorId &weightId,
                                     const Adam &adam) const {
  return adam.learningRates().get(weightId).isConst();
}

float AdamWeightDecayHelper::val(const TensorId &weightId,
                                 const Adam &adam) const {
  auto wd = adam.weightDecays().get(weightId).val();
  auto ls =
      adam.useScaledOptimizerState() &&
              adam.getWeightDecayMode() == WeightDecayMode::L2Regularization
          ? adam.lossScaling().val()
          : 1.0f;
  return val(wd, ls);
}

bool AdamWeightDecayHelper::isConst(const TensorId &weightId,
                                    const Adam &adam) const {
  auto constLs =
      adam.useScaledOptimizerState() &&
              adam.getWeightDecayMode() == WeightDecayMode::L2Regularization
          ? adam.lossScaling().isConst()
          : true;
  return adam.weightDecays().get(weightId).isConst() && constLs;
}

float AdamEpsHelper::val(const TensorId &weightId, const Adam &adam) const {
  auto eps = adam.epss().get(weightId).val();
  auto ls  = adam.useScaledOptimizerState() ? adam.lossScaling().val() : 1.0f;
  return val(eps, ls);
}

bool AdamEpsHelper::isConst(const TensorId &weightId, const Adam &adam) const {
  auto constLs =
      adam.useScaledOptimizerState() ? adam.lossScaling().isConst() : true;
  return adam.epss().get(weightId).isConst() && constLs;
}

float AdamLossScalingHelper::val(const TensorId &weightId,
                                 const Adam &adam) const {
  auto ls = adam.lossScaling().val();
  return val(ls);
}

bool AdamLossScalingHelper::isConst(const TensorId &weightId,
                                    const Adam &adam) const {
  return adam.lossScaling().isConst();
}

float AdamMaxWeightNormHelper::val(const TensorId &weightId,
                                   const Adam &adam) const {
  auto wd = adam.maxWeightNorms().get(weightId).val();
  return val(wd);
}

bool AdamMaxWeightNormHelper::isConst(const TensorId &weightId,
                                      const Adam &adam) const {
  return adam.maxWeightNorms().get(weightId).isConst();
}

float AdamGradientScalingHelper::val(const TensorId &weightId,
                                     const Adam &adam) const {
  float ls = adam.useScaledOptimizerState() ? 1.0f : adam.lossScaling().val();
  float af =
      adam.postMeanAccumulationEnabled() ? adam.getAccumulationFactor() : 1;
  float rf =
      adam.postMeanReplicationEnabled() ? adam.getReplicatedGraphCount() : 1;
  return val(ls, af, rf);
}

bool AdamGradientScalingHelper::isConst(const TensorId &weightId,
                                        const Adam &adam) const {
  auto constLs =
      adam.useScaledOptimizerState() ? true : adam.lossScaling().isConst();
  return constLs;
}

float AdaptiveAlphaHelper::val(const TensorId &weightId,
                               const Adaptive &adaptive) const {
  auto a = adaptive.alphas().get(weightId).val();
  return val(a);
}

bool AdaptiveAlphaHelper::isConst(const TensorId &weightId,
                                  const Adaptive &adaptive) const {
  return adaptive.alphas().get(weightId).isConst();
}

float AdaptiveMomentumHelper::val(const TensorId &weightId,
                                  const Adaptive &adaptive) const {
  auto a = adaptive.momentums().get(weightId).val();
  return val(a);
}

bool AdaptiveMomentumHelper::isConst(const TensorId &weightId,
                                     const Adaptive &adaptive) const {
  return adaptive.momentums().get(weightId).isConst();
}

float AdaptiveLearningRateHelper::val(const TensorId &weightId,
                                      const Adaptive &adaptive) const {
  auto lr = adaptive.learningRates().get(weightId).val();
  return val(lr);
}

bool AdaptiveLearningRateHelper::isConst(const TensorId &weightId,
                                         const Adaptive &adaptive) const {
  return adaptive.learningRates().get(weightId).isConst();
}

float AdaptiveWeightDecayHelper::val(const TensorId &weightId,
                                     const Adaptive &adaptive) const {
  auto wd = adaptive.weightDecays().get(weightId).val();
  return val(wd);
}

bool AdaptiveWeightDecayHelper::isConst(const TensorId &weightId,
                                        const Adaptive &adaptive) const {
  return adaptive.weightDecays().get(weightId).isConst();
}

float AdaptiveEpsHelper::val(const TensorId &weightId,
                             const Adaptive &adaptive) const {
  auto eps = adaptive.epss().get(weightId).val();
  return val(eps);
}

bool AdaptiveEpsHelper::isConst(const TensorId &weightId,
                                const Adaptive &adaptive) const {
  return adaptive.epss().get(weightId).isConst();
}

float AdaptiveLossScalingHelper::val(const TensorId &weightId,
                                     const Adaptive &adaptive) const {
  auto ls = adaptive.lossScaling().val();
  return val(ls);
}

bool AdaptiveLossScalingHelper::isConst(const TensorId &weightId,
                                        const Adaptive &adaptive) const {
  return adaptive.lossScaling().isConst();
}

float AdaptiveGradientScalingHelper::val(const TensorId &weightId,
                                         const Adaptive &adaptive) const {
  auto ls  = adaptive.lossScaling().val();
  float af = adaptive.postMeanAccumulationEnabled()
                 ? adaptive.getAccumulationFactor()
                 : 1;
  float rf = adaptive.postMeanReplicationEnabled()
                 ? adaptive.getReplicatedGraphCount()
                 : 1;
  return val(ls, af, rf);
}

bool AdaptiveGradientScalingHelper::isConst(const TensorId &weightId,
                                            const Adaptive &adaptive) const {
  return adaptive.lossScaling().isConst();
}

} // namespace popart
