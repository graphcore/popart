#include <string>
#include <popart/compoundscalarhelper.hpp>
#include <popart/op.hpp>
#include <popart/optimizer.hpp>

namespace popart {

bool CompoundScalarHelper::idMatch(const TensorId &optId) const {
  return (optId.find(specificPrefix()) != std::string::npos) ||
         (optId.find(defaultPrefix()) != std::string::npos);
}

OptimizerValue CompoundScalarHelper::getFromWeightId(const TensorId &weightId,
                                                     const SGD &sgd) const {
  return {val(weightId, sgd), isConst(weightId, sgd)};
}

OptimizerValue CompoundScalarHelper::getFromScalarId(const TensorId &optId,
                                                     const SGD &sgd) const {

  // the Optimizer Tensor is specific to a weight
  if (optId.find(specificPrefix()) != std::string::npos) {
    return getFromWeightId(getWeightId(optId), sgd);
  }

  else if (optId.find(defaultPrefix()) != std::string::npos) {
    return getFromWeightId("fudgeCakeSpaghettiBonanza", sgd);
  }

  throw error("ILE: failed to determine optimizer type from id {}", optId);
}

TensorId CompoundScalarHelper::getScalarId(const Tensor &w,
                                           const SGD &sgd) const {
  if (sgd.hasSpecific(w)) {
    return specificPrefix() + w.id;
  }
  return defaultPrefix() + w.info.data_type();
}

TensorId CompoundScalarHelper::getScalarIdIfNonConst(const Tensor &w,
                                                     const SGD &sgd) const {
  return isConst(w.id, sgd) ? "" : getScalarId(w, sgd);
}

// remove specific prefix to obtain the TensorId of the weight
TensorId CompoundScalarHelper::getWeightId(const TensorId &scalarId) const {
  if (scalarId.find(specificPrefix()) != std::string::npos) {
    return std::string(scalarId.begin() + specificPrefix().size(),
                       scalarId.end());
  }

  throw error("ILE in CompoundScalarHelper::getWeightId(.) : failed to find "
              "substring {} in {}",
              specificPrefix(),
              scalarId);
}

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
  return val(mm, sgd.getReplicatedGraphCount());
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
  return val(lr, vs, sgd.getReplicatedGraphCount());
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
  return val(dm, vs, ls, sgd.getReplicatedGraphCount());
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

} // namespace popart
