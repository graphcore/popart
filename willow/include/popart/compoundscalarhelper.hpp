#ifndef GUARD_NEURALNET_COMPOUND_SCALAR_HELPERS_HPP
#define GUARD_NEURALNET_COMPOUND_SCALAR_HELPERS_HPP

#include <memory>
#include <popart/names.hpp>
#include <popart/optimizervalue.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

namespace popart {

class SGD;

// Base helper class for scalars composed of other scalars
class CompoundScalarHelper {

public:
  CompoundScalarHelper()                             = default;
  virtual ~CompoundScalarHelper()                    = default;
  CompoundScalarHelper(const CompoundScalarHelper &) = default;

  OptimizerValue getFromWeightId(const TensorId &weightId, const SGD &) const;

  OptimizerValue getFromScalarId(const TensorId &compoundScalarId,
                                 const SGD &) const;

  // remove specific prefix to obtain the TensorId of the weight
  TensorId getWeightId(const TensorId &compoundScalarId) const;

  // Does the name of optId match the default or specific prefix
  bool idMatch(const TensorId &optId) const;

  virtual float val(const TensorId &weightId, const SGD &) const    = 0;
  virtual bool isConst(const TensorId &weightId, const SGD &) const = 0;

  // prepend appropriate prefix, which depends on if it is default or specific
  TensorId getScalarId(const Tensor &weight, const SGD &) const;

  // As above, but returns "" if the OptimizerValue is const
  TensorId getScalarIdIfNonConst(const Tensor &weight, const SGD &) const;

private:
  virtual std::string defaultPrefix() const  = 0;
  virtual std::string specificPrefix() const = 0;
};

class WeightDecayScaleFactor0Helper : public CompoundScalarHelper {
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

class ScaledLearningRate0Helper : public CompoundScalarHelper {
public:
  float val(const TensorId &weightId, const SGD &) const final;
  bool isConst(const TensorId &weightId, const SGD &) const final;
  float val(float lr, float ls, float dp) const {
    return lr * (1.0f - dp) / ls;
  }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultScaledLearningRate0Prefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificScaledLearningRate0Prefix();
  }
};

class WeightDecayScaleFactor1Helper : public CompoundScalarHelper {
public:
  float val(const TensorId &weightId, const SGD &) const final;
  bool isConst(const TensorId &weightId, const SGD &) const final;
  float val(float dm, float wd, float vs) const {
    return (1.0f - dm) * wd * vs;
  }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultWeightDecayScaleFactor1Prefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificWeightDecayScaleFactor1Prefix();
  }
};

class ScaledLearningRate1Helper : public CompoundScalarHelper {
public:
  float val(const TensorId &weightId, const SGD &) const final;
  bool isConst(const TensorId &weightId, const SGD &) const final;
  float val(float lr, float vs) const { return lr / vs; }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultScaledLearningRate1Prefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificScaledLearningRate1Prefix();
  }
};

class DampeningScaleFactor1Helper : public CompoundScalarHelper {
public:
  float val(const TensorId &weightId, const SGD &) const final;
  bool isConst(const TensorId &weightId, const SGD &) const final;
  float val(float dm, float vs, float ls) const {
    return (1.0f - dm) * vs / ls;
  }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultDampeningScaleFactor1Prefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificDampeningScaleFactor1Prefix();
  }
};

class Momentum1Helper : public CompoundScalarHelper {
public:
  float val(const TensorId &weightId, const SGD &) const final;
  bool isConst(const TensorId &weightId, const SGD &) const final;
  float val(float mm) const { return mm; }

private:
  std::string defaultPrefix() const final {
    return reservedDefaultMomentum1Prefix();
  }
  std::string specificPrefix() const final {
    return reservedSpecificMomentum1Prefix();
  }
};

} // namespace popart

#endif
