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

/**
 * Enum class representing a type of adaptive optimizer.
 */
enum class AdaptiveMode {
  /// AdaGrad optimizer.
  AdaGrad = 0,
  /// RMSProp optimizer.
  RMSProp,
  /// CenteredRMSProp optimizer.
  CenteredRMSProp,
  /// AdaDelta optimizer.
  AdaDelta
};

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
// Accl1 (FP16/FP32, RMSProp/AdaDelta/CenteredRMSProp):
// v1 = a * m + (1 - a) * s^2
//
// Accl1 (FP16/FP32, AdaGrad):
// v1 = v1 + s^2
//
// Accl2 (FP16/FP32, CenteredRMSProp):
// v2 = a * v2 + (1 - a) * s
//
// Updater term (AdaGrad/RMSProp):
// x = s / (sqrt(v1) + eps)
//
// Updater term (CenteredRMSProp):
// x = s / (sqrt(v1 - v2^2) + eps)
//
// Updater term (AdaDelta):
// x = (s * sqrt(v2 + eps)) / (sqrt(v1 + eps))
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

/**
 * AdaDelta, RMSProp and AdaGrad optimizer implementation.
 *
 * Akin to any optimizer implementation, this class is responsible for updating
 * each weight tensor (\f$w\f$) in the model using the gradient (\f$g\f$) of
 * the loss function with respect to the weight as calculated during the
 * backwards pass.
 *
 * The optimizer has the following **state** for each weight:
 *
 *  * *first-order momentum* (\f$v_1\f$)
 *  * *second-order momentum* (\f$v_2\f$) (only for AdaGrad/RMSProp)
 *  * *third-order momentum* (\f$v_3\f$)
 *
 * The optimizer has the following **hyper parameters**:
 *
 *  * *learning rate* (\f$\text{lr}\f$)
 *  * *weight decay* (\f$\text{wd}\f$)
 *  * *alpha* (\f$\alpha\f$)
 *  * *momentum* (\f$\text{m}\f$))
 *  * *epsilon* (\f$\epsilon\f$)
 *  * *loss scaling* (\f$\text{ls}\f$)
 *
 * The values of these parameters can be shared between all weights but some
 * can be overridden with weight-specific values (see Adaptive::insertSpecific).
 * Hyper parameters are captured using OptimizerValue objects and therefore
 * can be either a constant value or a non-constant value that can be adjusted
 * by the user.
 *
 * The values of #AdaptiveMode and #WeightDecayMode passed to the constructor
 * determines how weights are updated (see below).
 *
 * In the following we will describe how this optimizer updates a weight
 * using a gradient. In the context of this description the gradient is
 * is the value of the gradient *after* any gradient accumulation has been
 * performed and *after* the application of a loss scaling factor to the
 * gradient has been corrected for.
 *
 * When the optimizer needs to update a weight, \f$w\f$, using a gradient,
 * \f$g\f$,
 * it first computes a term \f$g_\text{tmp}\f$, which is effectively
 * is \f$g\f$ with L2 regularization applied if the #WeightDecayMode is
 * set to WeightDecayMode::L2Regularization this, as follows:
 *
 * \f[
 *    g_\text{tmp} := \left\{\begin{aligned}
 *        g                   & \text{ \; (Decay) } \\
 *        (g + \text{wd} * w) & \text{ \; (L2Regularization) \; . } \\
 *    \end{aligned}\right.\\
 * \f]
 *
 * Secondly, the optimizer updates \f$v_1\f$ the optimizer state as follows:
 *
 * \f[
 *    v_1' &:= \left\{\begin{aligned}
 *        \alpha * m + (1 - \alpha) * g_\text{tmp}^2 & \text{ \;
 * (RMSProp/AdaDelta) } \\
 *        \alpha * m + (1 - \alpha) * g_\text{tmp}^2 & \text{ \;
 * (CenteredRMSProp) } \\
 *        v_1 + g_\text{tmp}^2 & \text{ \; (AdaGrad) } \\
 *    \end{aligned}\right.\\
 * \f]
 *
 * Next, \f$v_2\f$ is updated, but only for CenteredRMSProp:
 *
 * \f[
 *    v_2' &:= \alpha * v_2 + (1 - \alpha) * g_\text{tmp} \text{ \;
 * (CenteredRMSProp) } \\ \f]
 *
 * Next, it computes the update term \f$u_\text{tmp}\f$:
 *
 * \f[
 *    u_\text{tmp} &:= \left\{\begin{aligned}
 *        \frac{g_\text{tmp}}{\sqrt{v_1'} + \epsilon}
 *             & \text{ \; (AdaGrad/RMSProp) } \\
 *        \frac{g_\text{tmp}}{\sqrt{v_1' - v_2'^2} + \epsilon}
 *            & \text{ \; (CenteredRMSProp) } \\
 *        \frac{g_\text{tmp} * \sqrt{v_2 + \epsilon}}{\sqrt{v_1' + \epsilon}}
 *            & \text{ \; (AdaDelta) } \\
 *    \end{aligned}\right.
 * \f]
 *
 * Next, \f$v_2\f$ is updated, but only for AdaDelta:
 *
 * \f[
 *    v_2' := \alpha * v_2 + (1 - \alpha) * u_\text{tmp}^2  \text{ \; (AdaDelta)
 * } \\ \f]
 *
 * Next the third momentum is updated for all modes:
 *
 * \f[
 *    v_3' := m * v_3 + u_\text{tmp}
 * \f]
 *
 * Finally, the optimizer updates the weight as follows:
 *
 * \f[
 *    w' := \left\{\begin{aligned}
 *        w - \text{lr} * (v_3' + \text{wd} * w) &\text{ \; (Decay) } \\
 *        w - \text{lr} * v_3'                   &\text{ \; (L2Regularization) }
 * \\ \end{aligned}\right. \f]
 *
 * In addition to the above, the *loss scaling* hyper parameter is similar in
 * nature to the velocity scaling parameter. It is a scaling value that is
 * applied to the loss gradient at the start of the the backwards pass and, at
 * the end of the backwards pass, this scaling is reversed by multiplying the
 * gradients for each weight with the inverse of the loss scaling value prior to
 * updating the optimizer state. Using loss scaling can also improve numerical
 * stability in some cases.
 */
class Adaptive : public Optimizer {

public:
  /// Default learning rate value.
  static OptimizerValue getUnsetLearningRate() {
    return {0.01f, true}; // a learning rate of 0.1 forever
  }

  /// Default weight decay value.
  static OptimizerValue getUnsetWeightDecay() {
    return {0.0f, true}; // no weight decay, ever
  }

  /// Default alpha value.
  static OptimizerValue getUnsetAlpha() {
    return {0.99f, true}; // fixed alpha of 0.99
  }

  /// Default momentum value.
  static OptimizerValue getUnsetMomentum() {
    return {0.0f, true}; // fixed momentum of 0.0
  }

  /// Default epsilon value.
  static OptimizerValue getUnsetEps() {
    return {1e-6f, true}; // a denominator stability term of 1e-6 forever
  }

  /// Default loss scaling value.
  static OptimizerValue getUnsetLossScaling() {
    return {1.0f, true}; // no loss scaling, ever
  }

public:
  // Does "w" have specific OptimizerValues, or will it use default?
  bool hasSpecific(const Tensor &w) const;

  /// Constructor.
  /// \param defaultLearningRate The learning rate value to use for weights
  ///     for which no weight-specific hyper parameter have been inserted.
  /// \param defaultWeightDecay The weight decay value to use for weights
  ///     for which no weight-specific hyper parameter have been inserted.
  /// \param defaultAlpha The alpha value to use for weights
  ///     for which no weight-specific hyper parameter have been inserted.
  /// \param defaultMomentum The momentum value to use for weights
  ///     for which no weight-specific hyper parameter have been inserted.
  /// \param defaultEps The epsilon value to use for weights
  ///     for which no weight-specific hyper parameter have been inserted.
  /// \param lossScaling The loss scaling value to use.
  /// \param adaptiveMode The AdaptiveMode value to use.
  /// \param weightDecayMode The WeightDecayMode value to use.
  /// \param accumType Data type to use for gradient accumulation.
  /// \param accl1Type Data type to use for tensor that stores first-order
  ///     momentum optimizer state.
  /// \param accl2Type Data type to use for tensor that stores second-order
  ///     momentum optimizer state.
  /// \param accl2Type Data type to use for tensor that stores third-order
  ///     momentum optimizer state.
  Adaptive(OptimizerValue defaultLearningRate,
           OptimizerValue defaultWeightDecay,
           OptimizerValue defaultAlpha,
           OptimizerValue defaultMomentum,
           OptimizerValue defaultEps,
           OptimizerValue lossScaling,
           AdaptiveMode adaptiveMode,
           WeightDecayMode weightDecayMode,
           DataType accumType,
           DataType accl1Type,
           DataType accl2Type,
           DataType accl3Type);

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

  /// Constructor.
  /// \param params A parameter map where keys are one of
  ///     `"defaultLearningRate"`, `"defaultWeightDecay"`, `"defaultAlpha"`,
  ///     `"defaultMomentum"`, `"defaultEps"` or `"lossScaling"`, and the
  ///     map's values pairs of floats and booleans
  ///     representing OptimizerValue constructor arguments. The map does not
  ///     have to specify each hyper parameter as default values will be used
  ///     where parameters are missing.
  /// \param adaptiveMode The AdaptiveMode value to use.
  /// \param weightDecayMode The WeightDecayMode value to use.
  /// \param accumType Data type to use for gradient accumulation.
  /// \param accl1Type Data type to use for tensor that stores first-order
  ///     momentum optimizer state.
  /// \param accl2Type Data type to use for tensor that stores second-order
  ///     momentum optimizer state.
  /// \param accl2Type Data type to use for tensor that stores third-order
  ///     momentum optimizer state.
  ///
  /// **EXAMPLE**:
  /// ```
  /// Adaptive({{"defaultLearningRate", {0.02, False}},
  //           {"defaultAlpha", {0.99, True}}},
  ///          AdaptiveMode::RMSProp,
  ///          WeightDecayMode::Decay,
  ///          DataType::FLOAT,
  ///          DataType::FLOAT,
  ///          DataType::FLOAT,
  ///          DataType::FLOAT);
  /// ```
  Adaptive(const std::map<std::string, std::pair<float, bool>> &params,
           AdaptiveMode adaptiveMode,
           WeightDecayMode weightDecayMode,
           DataType accumType,
           DataType accl1Type,
           DataType accl2Type,
           DataType accl3Type);
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

  /// The names of the inputs for the VarUpdateOp for the Variable Tensor
  /// "weight". In the returned vector,  an empty string ("") is used as a
  /// placeholder for constant inputs.
  std::vector<TensorId> getInputIds(const Tensor &weight) const final;

  /// The names and infos of the optimizer tensors.
  std::vector<std::tuple<TensorId, TensorInfo>>
  getOptimizerInputs(const Tensor &weight) const final;

  void validReplacement(const Optimizer &other) const final;

  void resetTensorData(Tensor &) const final;
  void setTensorData(Tensor &) const final;

  /// Tensor "opt" has an id, based on which it matches a compound scalar which
  /// this object can compute from the atomic scalars.
  float getStoredValue(const TensorId &optId) const;

  /// Insert a weight-specific set of hyper parameters.
  /// \param weight The TensorId of the weight.
  /// \param learningRate The learning rate value to use for this specific
  ///     weight.
  /// \param weightDecay The weight decay value to use for this specific
  ///     weight.
  /// \param alpha The alpha value to use for this specific
  ///     weight.
  /// \param momentum The momentum value to use for this specific
  ///     weight.
  /// \param eps The epsilon value to use for this specific
  ///     weight.
  void insertSpecific(const TensorId &weight,
                      OptimizerValue learningRate,
                      OptimizerValue weightDecay,
                      OptimizerValue alpha,
                      OptimizerValue momentum,
                      OptimizerValue eps);

  void setStep(int64_t step);
  void setStep(const TensorId &, int64_t step);
  void setStep(std::map<TensorId, int64_t> steps);

  /// Insert a weight-specific set of hyper parameters.
  /// \param weight The TensorId of the weight.
  /// \param params A parameter map where keys are one of
  ///     `"defaultLearningRate"`, `"defaultWeightDecay"`, `"defaultAlpha"`,
  ///     `"defaultMomentum"`, `"defaultEps"` or `"lossScaling"`
  ///     and the map's values pairs of floats and booleans representing
  ///     OptimizerValue constructor arguments. The map does not have to
  ///     specify each hyper parameter as default values will be used where
  ///     parameters are missing.
  void
  insertSpecific(const TensorId &weight,
                 const std::map<std::string, std::pair<float, bool>> &params);

  const OptimizerValueMap &learningRates() const { return lrs; }
  const OptimizerValueMap &weightDecays() const { return wds; }
  const OptimizerValueMap &alphas() const { return as; }
  const OptimizerValueMap &momentums() const { return ms; }
  const OptimizerValueMap &epss() const { return epsvs; }

  virtual size_t hash() const;

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
