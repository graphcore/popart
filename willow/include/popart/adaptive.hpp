// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ADAPTIVE_HPP
#define GUARD_NEURALNET_ADAPTIVE_HPP

#include <memory>
#include <popart/compoundscalarhelper.hpp>
#include <popart/names.hpp>
#include <popart/optimizer.hpp>
#include <popart/optimizervalue.hpp>
#include <popart/optimizervaluemap.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

namespace popart {

enum class AdaptiveMode { AdaGrad = 0, RMSProp, CenteredRMSProp, AdaDelta };

// Implements AdaGrad, RMSProp, CenteredAdaGrad, CenteredRMSProp, AdaDelta
//
// g = gradient computed in backward pass
// s = accumulated gradient
// x = updater term
// a = alpha parameter
// m = momentum parameter
// ls = loss scaling
// wd = weight decay
// eps = stability additive
// lr = learning rate
// w = weight
// v1 = first accl tensor
// v2 = second accl tensor
// v3 = third accl tensor
// rf = replication factor
// af = gradient accumulation factor
//
// If accumulationReductionType is set to ReductionType::Sum 'af' is set to 1
//
// accumulator (only used if gradient accumulation is enabled):
// s = s + g                             (otherwise: s = g)
//
// remove loss scaling factor:
// s = cast(s, FP16/FP32)
// s = s / (ls * af)
//
// L2 regularization (if wd > 0.0 and weight decay mode: L2 regularization)
// s = s + wd * w
//
// Accl1 (FP16/FP32, RMSProp/AdaDelta):
// v1 = a * m + (1 - a) * s^2
//
// Accl1 (FP16/FP32, AdaGrad):
// v1 = v1 + s^2
//
// Accl2 (FP16/FP32, CenteredRMSProp):
// v2 = a * v2 + (1 - a) * s
//
// Updater term (AdaGrad/RMSProp):
// x = g / (sqrt(v1 - v2^2) + eps)
//
// Updater term (AdaDelta):
// x = (g * sqrt(v2 + eps)) / (sqrt(v1 + eps))
//
// Accl2 (FP16/FP32, AdaDelta):
// v2 = a * v2 + (1 - a) * x^2
//
// Accl3 (momentum):
// v3 = m * v3 + x                       (if m = 0.0: v3 = x)
//
// Var update (with weight decay mode: decay and wd > 0.0)
// w -= lr * (wd * w + v3)
//
// Var update (without weight decay mode: decay)
// w -= lr * v3
//
// accumulator update:
//   (if gradient accumulation is enabled)
// s = 0
//
// Reference compute graph including all variants:
//   - replicated weight sharding:
//       every replica only holds a 1/rf optimizer shard
//   - gradient accumulation
//       update is applied only every N-th iteration, gradients accumulated in s
//   - gradient reduction
//       gradient is reduced every iteration (on g)
//   - accumulation reduction
//       accumulator is reduced every N-th iteration (on s)
//
// RMSProp/AdaGrad:
//                  [s]     [g]
//                   |       |
//                   |    (AllReduce)  (gradient reduction only)
//                   |       |
//                  (Accumulate)       (gradient accumulation only)
//                     |
//                    [s']
//                     |
//                  (AllReduceInplace) (accumulation reduction only)
//                     |
//                   [s''] [a]------------.
//                     |    |             |
//    .----------------+-------------.    |
//    |          [v1]  |    |  [v2]  |    |
//    |            |   |    |    |   |    |
//    |       (Accumulate(^1)) (Accumulate(^2)) (centered RMSProp only)
//    |               |             |
//    |             [v1']  [eps]  [v2']
//    |               \      |     /
//    +--------------(RMSPropUpdater)
//    |                      |
// (AccumulatorUpd)    [v3] [x]  [m]
//    |                  |   |    |
//   [s''']            (Accumulate(^3))         (momentum m > 0.0 only)
//                           |
//                    [w]  [v3'] [wd] [lr]
//                     |     |    |    |
//                     (ScaledVarUpdate)
//                           |
//                          [w']
//
// AdaDelta:
//                  [s]     [g]
//                   |       |
//                   |    (AllReduce)  (gradient reduction only)
//                   |       |
//                  (Accumulate)       (gradient accumulation only)
//                     |
//                    [s']
//                     |
//                  (AllReduceInplace) (accumulation reduction only)
//                     |
//                   [s''] [a]--------------.
//                     |    |               |
//    .----------------+    |               |
//    |          [v1]  |    |               |
//    |            |   |    |               |
//    |       (Accumulate(^1))     [v2]     |
//    |               |             |       |
//    |             [v1']  [eps]    +-----. |
//    |               \      |      |     | |
//    +---------------(AdaDeltaUpdater)   | |
//    |                      |            | |
// (AccumulatorUpd)         [x]           | |
//    |                      +----------. | |
//   [s''']                  |          | | |
//                           |         (Accumulate(^2))
//                     [v3]  |   [m]      |
//                       |   |    |      [v2']
//                     (Accumulate(^3))         (momentum m > 0.0 only)
//                           |
//                    [w]  [v3'] [wd] [lr]
//                     |     |    |    |
//                     (ScaledVarUpdate)
//                           |
//                          [w']

class Adaptive : public Optimizer {

public:
  static OptimizerValue getUnsetAlpha() {
    return {0.99f, true}; // fixed alpha of 0.99
  }

  static OptimizerValue getUnsetMomentum() {
    return {0.0f, true}; // fixed momentum of 0.0
  }

  static OptimizerValue getUnsetEps() {
    return {1e-6f, true}; // a denominator stability term of 1e-6 forever
  }

  static OptimizerValue getUnsetWeightDecay() {
    return {0.0f, true}; // no weight decay, ever
  }

  static OptimizerValue getUnsetLossScaling() {
    return {1.0f, true}; // no loss scaling, ever
  }

  static OptimizerValue getUnsetLearningRate() {
    return {0.01f, true}; // a learning rate of 0.1 forever
  }

public:
  // Does "w" have specific OptimizerValues, or will it use default?
  bool hasSpecific(const Tensor &w) const;

  // Adaptive constructor with all parameters
  // ----------------
  Adaptive(OptimizerValue default_lr,
           OptimizerValue default_wd,
           OptimizerValue default_alpha,
           OptimizerValue default_momentum,
           OptimizerValue default_eps,
           OptimizerValue ls,
           AdaptiveMode adaptiveMode_,
           WeightDecayMode decayMode_,
           DataType accumType_,
           DataType accl1Type_,
           DataType accl2Type_,
           DataType accl3Type_);

  // Example:
  //
  // Adaptive({{"defaultLearningRate", {0.02, False}},
  //       {"defaultAlpha", {0.99, True}}});
  //
  // will create an Adaptive Optimizer which has a constant alpha of
  // 0.99 and a changeable learning rate initially of 0.02. All
  // OptimizerValues not present in the map will take values from the getUnset*
  // functions.
  //
  // Construct from pair instead of OptimizerValue for pybind11 support
  //
  Adaptive(const std::map<std::string, std::pair<float, bool>> &,
           AdaptiveMode adaptiveMode_,
           WeightDecayMode decayMode_,
           DataType accumType_,
           DataType accl1Type_,
           DataType accl2Type_,
           DataType accl3Type_);
  static Adaptive fromDefaultMap(const std::map<std::string, OptimizerValue> &,
                                 AdaptiveMode adaptiveMode_,
                                 WeightDecayMode decayMode_,
                                 DataType accumType_,
                                 DataType accl1Type_,
                                 DataType accl2Type_,
                                 DataType accl3Type_);

  Adaptive(const Adaptive &) = default;
  ~Adaptive()                = default;

  OptimizerType type() const final { return OptimizerType::Adaptive; }
  std::string type_s() const final { return "Adaptive"; }

  std::unique_ptr<Optimizer> clone() const final;

  std::unique_ptr<Op> createOp(const Tensor &weight, Graph &) const final;

  // The names of the inputs for the VarUpdateOp for the Variable Tensor
  // "weight". In the returned vector,  a "" is used as a placeholder for
  // constant inputs
  std::vector<TensorId> getInputIds(const Tensor &weight) const final;

  // The names and infos of the optimizer Tensors
  std::vector<std::tuple<TensorId, TensorInfo>>
  getOptimizerInputs(const Tensor &weight) const final;

  bool validReplacement(const Optimizer &other) const final;

  void resetTensorData(Tensor &) const final;
  void setTensorData(Tensor &) const final;

  // Tensor "opt" has an id, based on which it matches a compound scalar which
  // this object can compute from the atomic scalars
  float getStoredValue(const TensorId &optId) const;

  void insertSpecific(const TensorId &,
                      OptimizerValue lr,
                      OptimizerValue wd,
                      OptimizerValue a,
                      OptimizerValue m,
                      OptimizerValue eps);

  void setStep(int64_t step);
  void setStep(const TensorId &, int64_t step);
  void setStep(std::map<TensorId, int64_t> steps);

  // insert OptimizerValues specific to one Tensor. The keys of the map should
  // be the names of atomic optimizer scalars, such as "momentum",
  // "learningRate". The map does not need to be complete. If it is not
  // complete, the default values already set for the SGD will be used.
  void insertSpecific(const TensorId &,
                      const std::map<std::string, std::pair<float, bool>> &);

  const OptimizerValueMap &learningRates() const { return lrs; }
  const OptimizerValueMap &weightDecays() const { return wds; }
  const OptimizerValueMap &alphas() const { return as; }
  const OptimizerValueMap &momentums() const { return ms; }
  const OptimizerValueMap &epss() const { return epsvs; }

private:
  void runValueChecks(OptimizerValue lr,
                      OptimizerValue wd,
                      OptimizerValue a,
                      OptimizerValue m,
                      OptimizerValue eps) const;

  // The atomic scalars
  // ------------------
  // learning rates
  OptimizerValueMap lrs;

  // weight decays
  OptimizerValueMap wds;

  // alpha
  OptimizerValueMap as;

  // momentum
  OptimizerValueMap ms;

  // eps values
  OptimizerValueMap epsvs;

  // Adaptive settings
  AdaptiveMode mode;
  WeightDecayMode decayMode;
  DataType accumType;
  DataType accl1Type;
  DataType accl2Type;
  DataType accl3Type;

  // The compound scalars
  // --------------------
  AdaptiveLearningRateHelper lrhelper;
  AdaptiveWeightDecayHelper wdhelper;
  AdaptiveAlphaHelper ahelper;
  AdaptiveMomentumHelper mhelper;
  AdaptiveEpsHelper epshelper;
  AdaptiveLossScalingHelper lshelper;
  AdaptiveGradientScalingHelper gshelper;

  // int argument only to disambiguate from the other SGD constructor
  Adaptive(const std::map<std::string, OptimizerValue> &,
           AdaptiveMode mode_,
           WeightDecayMode decayMode_,
           DataType accumType_,
           DataType accl1Type_,
           DataType accl2Type_,
           DataType accl3Type_,
           int);

  static std::map<std::string, OptimizerValue>
  getComplete(const std::map<std::string, OptimizerValue> &);
};

} // namespace popart

#endif
