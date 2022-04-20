// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_COMPOUND_SCALAR_HELPERS_HPP
#define GUARD_NEURALNET_COMPOUND_SCALAR_HELPERS_HPP

#include <string>
#include <popart/optimizervalue.hpp>
#include <popart/tensornames.hpp>

#include "popart/tensordebuginfo.hpp"

namespace popart {

class SGD;
class Adam;
class Adaptive;
class Tensor;

// Base helper class for scalars composed of other scalars
template <class T> class CompoundScalarHelper {

public:
  CompoundScalarHelper()                             = default;
  virtual ~CompoundScalarHelper()                    = default;
  CompoundScalarHelper(const CompoundScalarHelper &) = default;

  OptimizerValue getFromWeightId(const TensorId &weightId, const T &) const;

  OptimizerValue getFromScalarId(const TensorId &compoundScalarId,
                                 const T &) const;

  // remove specific prefix to obtain the TensorId of the weight
  TensorId getWeightId(const TensorId &compoundScalarId) const;

  // Does the name of optId match the default or specific prefix
  bool idMatch(const TensorId &optId) const;

  virtual float val(const TensorId &weightId, const T &) const    = 0;
  virtual bool isConst(const TensorId &weightId, const T &) const = 0;

  // prepend appropriate prefix, which depends on if it is default or specific
  TensorId getScalarId(const Tensor &weight, const T &) const;

  // As above, but returns "" if the OptimizerValue is const
  TensorId getScalarIdIfNonConst(const Tensor &weight, const T &) const;

private:
  virtual std::string defaultPrefix() const  = 0;
  virtual std::string specificPrefix() const = 0;
};

extern template class CompoundScalarHelper<SGD>;
extern template class CompoundScalarHelper<Adam>;

class WeightDecayScaleFactor0Helper : public CompoundScalarHelper<SGD> {
public:
  float val(const TensorId &weightId, const SGD &) const final;
  bool isConst(const TensorId &weightId, const SGD &) const final;
  float val(float wd, float lr, float dp) const {
    return 1.0f - lr * (1.0f - dp) * wd;
  }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultWeightDecayScaleFactor0Prefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificWeightDecayScaleFactor0Prefix();
  }
};

class ScaledLearningRate0Helper : public CompoundScalarHelper<SGD> {
public:
  float val(const TensorId &weightId, const SGD &) const final;
  bool isConst(const TensorId &weightId, const SGD &) const final;
  float val(float lr, float dp, float ls, float af, float rf) const {
    return (lr * (1.0f - dp)) / (ls * af * rf);
  }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultScaledLearningRate0Prefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificScaledLearningRate0Prefix();
  }
};

class ScaledWeightDecay1Helper : public CompoundScalarHelper<SGD> {
public:
  float val(const TensorId &weightId, const SGD &) const final;
  bool isConst(const TensorId &weightId, const SGD &) const final;
  float val(float dm, float wd, float vs) const {
    return (1.0f - dm) * wd * vs;
  }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultScaledWeightDecay1Prefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificScaledWeightDecay1Prefix();
  }
};

class ScaledLearningRate1Helper : public CompoundScalarHelper<SGD> {
public:
  float val(const TensorId &weightId, const SGD &) const final;
  bool isConst(const TensorId &weightId, const SGD &) const final;
  float val(float lr, float vs, float rf) const { return lr / (vs * rf); }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultScaledLearningRate1Prefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificScaledLearningRate1Prefix();
  }
};

class DampeningScaleFactor1Helper : public CompoundScalarHelper<SGD> {
public:
  float val(const TensorId &weightId, const SGD &) const final;
  bool isConst(const TensorId &weightId, const SGD &) const final;
  float val(float dm, float vs, float ls, float af, float rf) const {
    return (1.0f - dm) * vs * rf / (ls * af);
  }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultDampeningScaleFactor1Prefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificDampeningScaleFactor1Prefix();
  }
};

class ScaledMomentum1Helper : public CompoundScalarHelper<SGD> {
public:
  float val(const TensorId &weightId, const SGD &) const final;
  bool isConst(const TensorId &weightId, const SGD &) const final;
  float val(float mm, float rf) const { return mm / rf; }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultScaledMomentum1Prefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificScaledMomentum1Prefix();
  }
};

class ScaledLearningRate2Helper : public CompoundScalarHelper<SGD> {
public:
  float val(const TensorId &weightId, const SGD &) const final;
  bool isConst(const TensorId &weightId, const SGD &) const final;
  float val(float lr, float vs) const { return lr / vs; }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultScaledLearningRate2Prefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificScaledLearningRate2Prefix();
  }
};

class DampeningScaleFactor2Helper : public CompoundScalarHelper<SGD> {
public:
  float val(const TensorId &weightId, const SGD &) const final;
  bool isConst(const TensorId &weightId, const SGD &) const final;
  float val(float dm, float vs, float ls, float af, float rf) const {
    return (1.0f - dm) * vs / (ls * af * rf);
  }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultDampeningScaleFactor2Prefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificDampeningScaleFactor2Prefix();
  }
};

class ScaledMomentum2Helper : public CompoundScalarHelper<SGD> {
public:
  float val(const TensorId &weightId, const SGD &) const final;
  bool isConst(const TensorId &weightId, const SGD &) const final;
  float val(float mm) const { return mm; }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultScaledMomentum2Prefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificScaledMomentum2Prefix();
  }
};

class AdamBeta1Helper : public CompoundScalarHelper<Adam> {
public:
  float val(const TensorId &weightId, const Adam &) const final;
  bool isConst(const TensorId &weightId, const Adam &) const final;
  float val(float b1) const { return b1; }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultAdamBeta1Prefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificAdamBeta1Prefix();
  }
};

class AdamBeta2Helper : public CompoundScalarHelper<Adam> {
public:
  float val(const TensorId &weightId, const Adam &) const final;
  bool isConst(const TensorId &weightId, const Adam &) const final;
  float val(float b2) const { return b2; }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultAdamBeta2Prefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificAdamBeta2Prefix();
  }
};

class AdamLearningRateHelper : public CompoundScalarHelper<Adam> {
public:
  float val(const TensorId &weightId, const Adam &) const final;
  bool isConst(const TensorId &weightId, const Adam &) const final;
  float val(float lr) const { return lr; }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultLearningRatePrefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificLearningRatePrefix();
  }
};

class AdamWeightDecayHelper : public CompoundScalarHelper<Adam> {
public:
  float val(const TensorId &weightId, const Adam &) const final;
  bool isConst(const TensorId &weightId, const Adam &) const final;
  float val(float wd, float ls) const { return wd * ls; }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultWeightDecayPrefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificWeightDecayPrefix();
  }
};

class AdamEpsHelper : public CompoundScalarHelper<Adam> {
public:
  float val(const TensorId &weightId, const Adam &) const final;
  bool isConst(const TensorId &weightId, const Adam &) const final;
  float val(float eps, float ls) const { return eps * ls; }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultAdamEpsPrefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificAdamEpsPrefix();
  }
};

class AdamLossScalingHelper : public CompoundScalarHelper<Adam> {
public:
  float val(const TensorId &weightId, const Adam &) const final;
  bool isConst(const TensorId &weightId, const Adam &) const final;
  float val(float ls) const { return ls; }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultLossScalingPrefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificLossScalingPrefix();
  }
};

class AdamMaxWeightNormHelper : public CompoundScalarHelper<Adam> {
public:
  float val(const TensorId &weightId, const Adam &) const final;
  bool isConst(const TensorId &weightId, const Adam &) const final;
  float val(float mwn) const { return mwn; }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultMaxWeightNormPrefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificMaxWeightNormPrefix();
  }
};

class AdamGradientScalingHelper : public CompoundScalarHelper<Adam> {
public:
  float val(const TensorId &weightId, const Adam &) const final;
  bool isConst(const TensorId &weightId, const Adam &) const final;
  float val(float ls, float af, float rf) const { return 1 / (ls * af * rf); }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultAdamGradientScalingPrefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificAdamGradientScalingPrefix();
  }
};

class AdaptiveAlphaHelper : public CompoundScalarHelper<Adaptive> {
public:
  float val(const TensorId &weightId, const Adaptive &) const final;
  bool isConst(const TensorId &weightId, const Adaptive &) const final;
  float val(float a) const { return a; }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultAdaptiveAlphaPrefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificAdaptiveAlphaPrefix();
  }
};

class AdaptiveMomentumHelper : public CompoundScalarHelper<Adaptive> {
public:
  float val(const TensorId &weightId, const Adaptive &) const final;
  bool isConst(const TensorId &weightId, const Adaptive &) const final;
  float val(float a) const { return a; }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultAdaptiveMomentumPrefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificAdaptiveMomentumPrefix();
  }
};

class AdaptiveLearningRateHelper : public CompoundScalarHelper<Adaptive> {
public:
  float val(const TensorId &weightId, const Adaptive &) const final;
  bool isConst(const TensorId &weightId, const Adaptive &) const final;
  float val(float b2) const { return b2; }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultLearningRatePrefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificLearningRatePrefix();
  }
};

class AdaptiveWeightDecayHelper : public CompoundScalarHelper<Adaptive> {
public:
  float val(const TensorId &weightId, const Adaptive &) const final;
  bool isConst(const TensorId &weightId, const Adaptive &) const final;
  float val(float wd) const { return wd; }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultWeightDecayPrefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificWeightDecayPrefix();
  }
};

class AdaptiveEpsHelper : public CompoundScalarHelper<Adaptive> {
public:
  float val(const TensorId &weightId, const Adaptive &) const final;
  bool isConst(const TensorId &weightId, const Adaptive &) const final;
  float val(float eps) const { return eps; }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultAdaptiveEpsPrefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificAdaptiveEpsPrefix();
  }
};

class AdaptiveLossScalingHelper : public CompoundScalarHelper<Adaptive> {
public:
  float val(const TensorId &weightId, const Adaptive &) const final;
  bool isConst(const TensorId &weightId, const Adaptive &) const final;
  float val(float ls) const { return ls; }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultLossScalingPrefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificLossScalingPrefix();
  }
};

class AdaptiveMaxWeightNormHelper : public CompoundScalarHelper<Adaptive> {
public:
  float val(const TensorId &weightId, const Adaptive &) const final;
  bool isConst(const TensorId &weightId, const Adaptive &) const final;
  float val(float mwn) const { return mwn; }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultMaxWeightNormPrefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificMaxWeightNormPrefix();
  }
};

class AdaptiveGradientScalingHelper : public CompoundScalarHelper<Adaptive> {
public:
  float val(const TensorId &weightId, const Adaptive &) const final;
  bool isConst(const TensorId &weightId, const Adaptive &) const final;
  float val(float ls, float af, float rf) const { return 1 / (ls * af * rf); }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultAdaptiveGradientScalingPrefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificAdaptiveGradientScalingPrefix();
  }
};

} // namespace popart

#endif
