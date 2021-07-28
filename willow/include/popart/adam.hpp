// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ADAM_HPP
#define GUARD_NEURALNET_ADAM_HPP

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
 * Enum type describing the mode of an Adam optimizer instance.
 */
enum class AdamMode {
  /// Adam or AdamW mode, depending on weight decay setting (see
  /// [Kingma & Ba, 2015](https://arxiv.org/abs/1412.6980)
  /// and [Loshchilov & Hutter, 2018](https://arxiv.org/pdf/1711.05101.pdf)).
  Adam = 0,
  /// Like Adam but without bias correction.
  AdamNoBias,
  /// Adamax mode.
  AdaMax,
  /// Lamb mode (see [You et al., 2020](https://arxiv.org/abs/1904.00962)).
  Lamb,
  /// Like Lamb but without bias correction.
  LambNoBias
};

// Implements Adam(W)
//   (Adam: A Method For Stochastic Optimization, Kingma & Ba, ICLR 2015)
//   (Fixing Weight Decay Regularization in Adam, Loshchilov & Hutter,
//    ICLR 2018 reject)
// and Lamb (with and without bias correction)
//   (Large Batch Optimization For Deep learning: Training BERT in 76 Minutes,
//    You et al., ICLR 2020)
//
//
// g = gradient computed in backward pass
// s = accumulated gradient
// x = updater term
// t = current step with Adam (FP32 counter)
// b1 = beta1 (Adam)
// b2 = beta2 (Adam)
// ls = loss scaling
// wd = weight decay
// eps = stability additive
// mwn = max weight norm (c.f. phi or scaling function in Lamb paper)
// lr = learning rate
// w = weight
// m = 1st momentum
// mc = bias corrected 1st momentum
// v = 2nd momentum
// vc = bias corrected 2nd momentum
// r1 = (Lamb) L2 norm of the weight (w)
// r2 = (Lamb) L2 norm of the updater term (x)
// rf = replication factor
// af = gradient accumulation factor
//
// If accumulationAndReplicationReductionType is set to ReductionType::Sum 'af'
// is set to 1
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
// first order momentum (FP16/FP32):
// m = b1 * m + (1 - b1) * s
//
// bias corrected:
// mc = m / (1 - b1 ** t)  (without correction: mc = m)
//
// second order momentum (FP16/FP32, Adam/Lamb):
// v = b2 * v + (1 - b2) * s^2
//
// second order momentum (FP16/FP32, AdaMax):
// v = max(b2 * v, abs(s))
//
// bias corrected:
// vc = v / (1 - b2 ** t)  (without correction: vc = v)
//
// updater term (FP16/FP32, with weight decay mode: decay and wd > 0.0):
// x = mc / (sqrt(vc) + eps) + wd * w
//
// updater term (FP16/FP32, without weight decay mode: decay):
// x = mc / (sqrt(vc) + eps)
//
//
// Lamb r1 (FP32):
// r1 = ||w||_2                          (without Lamb or r1 == 0: r1/r2 = 1)
//   special case: replicated weight sharding; every replica only stores a
//   shard of w, therefore the sum-of-squares is computed replicated, and
//   thereafter all-reduced before every replica takes the square root of r1sq
//
// Lamb r2 (FP32):
// r2 = ||x||_2                          (without Lamb or r2 == 0: r1/r2 = 1)
//   special case: replicated weight sharding; every replica only stores a
//   shard of x, therefore the sum-of-squares is computed replicated, and
//   thereafter all-reduced before every replica takes the square root of r2sq
//
// variable update:
// w -= min(r1, mwn) / r2 * lr * x
//      ^^^^^^^^^^^^^^
//      Lamb trust ratio
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
//                   [s''] [b1]-------------------.
//                     |    |            [b2]---. |
//    .----------------+-------------.    |     | |
//    |           [m]  |    |   [v]  |    |     | |
//    |            |   |    |    |   |    |     | |
//    |       (Accumulate(^1)) (Accumulate(^2)) | |    [wd] [eps] [ls]
//    |               |             |           | |     |     |    |
// (AccumulatorUpd)  [m']  [t]     [v']         | |     |     |    |
//    |               \     |       /          / /     /     /    /
//   [s''']       [w]--(AdamUpdater)------------------------------
//                 \        |
//  (gradient       \      [x]--------------------.
//   accumulation    \------|--------.            |
//   only)            \     |        |            |
//                     \    |      (LambSquare) (LambSquare) (Lamb only)
//                      \   |        |            |
//                       |  |       [r1sq]       [r2sq]
//                       |  |        |            |
//                       |  |       (AllReduce)  (AllReduce) (replicated weight
//                 [mwn] |  | [lr]  /            /            sharding only)
//                     \ |  |  |   / .-----------
//                     (AdamVarUpdate)
//                       |
//                      [w']

/**
 * AdamW, Lamb and AdaMax optimizer implementation.
 *
 * Akin to any optimizer implementation, this class is responsible for updating
 * each weight tensor (\f$w\f$) in the model using the gradient (\f$g\f$) of
 * the loss function with respect to the weight as calculated during the
 * backwards pass.
 *
 * The optimizer has the following **state** for each weight:
 *
 *  * *first-order momentum* (\f$m\f$)
 *  * *second-order momentum* (\f$v\f$)
 *  * *time step* (\f$t\f$)
 *
 * The optimizer has the following **hyper parameters**:
 *
 *  * *learning rate* (\f$\text{lr}\f$)
 *  * *weight decay* (\f$\text{wd}\f$)
 *  * *beta1* (\f$\beta_1\f$)
 *  * *beta2* (\f$\beta_2\f$)
 *  * *epsilon* (\f$\epsilon\f$)
 *  * *loss scaling* (\f$\text{ls}\f$)
 *  * *maximum weight norm* (\f$\text{mwn}\f$)
 *
 * The values of these parameters can be shared between all weights but some
 * can be overridden with weight-specific values (see Adam::insertSpecific).
 * Hyper parameters are captured using OptimizerValue objects and therefore
 * can be either a constant value or a non-constant value that can be adjusted
 * by the user.
 *
 * The values of #AdamMode and #WeightDecayMode passed to the constructor
 * determines how weights are updated (see below).
 *
 * In the following we will describe how this optimizer updates a weight
 * using a gradient. In the context of this description the gradient is
 * is the value of the gradient *after* any gradient accumulation has been
 * performed and *after* the application of a loss scaling factor to the
 * gradient has been corrected for.
 *
 * When the optimizer needs to update a weight, \f$w\f$, using a gradient,
 * \f$g\f$, it first computes a term \f$g_\text{tmp}\f$, which is effectively
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
 * Secondly, the optimizer updates the optimizer state as follows:
 *
 * \f[
 *    m' &:= \beta_1 * m + (1 - \beta_1) * g_\text{tmp} \\
 *    v' &:= \left\{\begin{aligned}
 *        \beta_2 * v + (1 - \beta_2) * g_\text{tmp}^2 & \text{ \;
 * (Adam/AdamNoBias) } \\
 *        \beta_2 * v + (1 - \beta_2) * g_\text{tmp}^2 & \text{ \;
 * (Lamb/LambNoBias) } \\
 *        \text{max}(\beta_2 * v, |g_\text{tmp}|)      & \text{ \; (AdaMax) } \\
 *    \end{aligned}\right.\\
 *    t' &:= t + 1 \\
 * \f]
 *
 * Next, it computes the following terms:
 *
 * \f[
 *    m_\text{tmp} &:= \left\{\begin{aligned}
 *        m'                            & \text{ \; (AdamNoBias/LambNoBias) } \\
 *        \frac{m'}{(1 - \beta_1^{t'})} & \text{ \; (Adam/Lamb/AdaMax) } \\
 *    \end{aligned}\right.\\
 *    v_\text{tmp} &:= \left\{\begin{aligned}
 *        v'                            & \text{ \; (AdamNoBias/LambNoBias) } \\
 *        \frac{v'}{(1 - \beta_2^{t'})} & \text{ \; (Adam/Lamb/AdaMax) } \\
 *    \end{aligned}\right.\\
 *    u_\text{tmp} &:= \left\{\begin{aligned}
 *        \frac{m_\text{tmp}}{(\sqrt{v_\text{tmp}} + \epsilon)} + \text{wd} * w
 * &\text{ \; (Decay) } \\
 *        \frac{m_\text{tmp}}{(\sqrt{v_\text{tmp}} + \epsilon)} &\text{ \;
 * (L2Regularization) } \\ \end{aligned}\right. \f]
 *
 * Finally, the optimizer updates the weight as follows:
 *
 * \f[
 *    w' := \left\{\begin{aligned}
 *        w - \text{lr} * u_\text{tmp} &\text{ \; (Adam/AdamNoBias/AdaMax) } \\
 *        w - \biggl(\frac{\text{min}(\lVert{w}\rVert,
 * \text{mwn})}{\lVert{u_\text{tmp}}\rVert}\biggr) *
 * \text{lr} *  u_\text{tmp} &\text{ \; (Lamb/LambNoBias) } \\
 *    \end{aligned}\right.
 * \f]
 *
 * In addition to the above, the *loss scaling* hyper parameter is similar in
 * nature to the velocity scaling parameter. It is a scaling value that is
 * applied to the loss gradient at the start of the the backwards pass and, at
 * the end of the backwards pass, this scaling is reversed by multiplying the
 * gradients for each weight with the inverse of the loss scaling value prior to
 * updating the optimizer state. Using loss scaling can also improve numerical
 * stability of the gradient calculations. If scaledOptimizerState is enabled
 * then the the lossScaling will not be removed before updating the optimizer
 * state. This can improve the numerical stability when accl1_type is set to
 * FLOAT16.
 *
 * **NOTE**: The maximum weight norm is referred to as \f$\phi\f$ in
 *        [You et al., 2020](https://arxiv.org/abs/1904.00962).
 */
class Adam : public Optimizer {

public:
  /// Default learning rate value.
  static OptimizerValue getUnsetLearningRate() {
    return {0.1f, true}; // a learning rate of 0.1 forever
  }

  /// Default weight decay value.
  static OptimizerValue getUnsetWeightDecay() {
    return {0.0f, true}; // no weight decay, ever
  }

  /// Default beta1 value.
  static OptimizerValue getUnsetBeta1() {
    return {0.9f, true}; // fixed beta1 of 0.9
  }

  /// Default beta2 value.
  static OptimizerValue getUnsetBeta2() {
    return {0.999f, true}; // fixed beta2 of 0.999
  }

  /// Default epsilon value.
  static OptimizerValue getUnsetEps() {
    return {1e-6f, true}; // a denominator stability term of 1e-6 forever
  }

  /// Default loss scaling value.
  static OptimizerValue getUnsetLossScaling() {
    return {1.0f, true}; // no loss scaling, ever
  }

  /// Default maximum weight norm value.
  static OptimizerValue getUnsetMaxWeightNorm() {
    return {10.0f, true}; // a maximum weight norm of 10.0f forever
  }

public:
  // Does "w" have specific OptimizerValues, or will it use default?
  bool hasSpecific(const Tensor &w) const final;

  // Do any weights have specific OptimizerValues, or do they all use default?
  bool hasSpecific() const final;

  TensorId getInverseLossScalingTensorId(const Tensor &weight) const final;

  /// Constructor.
  /// \param defaultLearningRate The learning rate value to use for weights
  ///     for which no weight-specific hyper parameter have been inserted.
  /// \param defaultWeightDecay The weight decay value to use for weights
  ///     for which no weight-specific hyper parameter have been inserted.
  /// \param defaultBeta1 The beta1 value to use for weights
  ///     for which no weight-specific hyper parameter have been inserted.
  /// \param defaultBeta2 The beta2 value value to use for weights
  ///     for which no weight-specific hyper parameter have been inserted.
  /// \param defaultEps The epsilon value to use for
  ///     weights for which no weight-specific hyper parameter have been
  ///     inserted.
  /// \param lossScaling The loss scaling value to use.
  /// \param maxWeightNorm The maxWeightNorm value to use.
  /// \param adamMode The AdamMode value to use.
  /// \param weightDecayMode The WeightDecayMode value to use.
  /// \param maxWeightNorm The maxWeightNorm value to use.
  /// \param accumType Data type to use for gradient accumulation.
  /// \param accl1Type Data type to use for tensor that stores first-order
  ///     momentum optimizer state.
  /// \param accl2Type Data type to use for tensor that stores second-order
  ///     momentum optimizer state.
  /// \param clipNormSettings A vector of ClipNormSettings (this can be used
  ///     to set maximum values for weights).
  /// \param scaledOptimizerState Experimental Option.
  ///     Does not remove lossScaling before updating the optimizer state. This
  ///     should have no effect on the update equation. However, it does ensure
  ///     a more numerically stable implementation when accl1_type is set to
  ///     DataType::FLOAT16. Note: When loading a model that includes
  ///     initialised optimizer state, ensure that accl1 and accl2 are scaled by
  ///     lossScaling and lossScaling^2 respectively.
  Adam(OptimizerValue defaultLearningRate,
       OptimizerValue defaultWeightDecay,
       OptimizerValue defaultBeta1,
       OptimizerValue defaultBeta2,
       OptimizerValue defaultEps,
       OptimizerValue lossScaling,
       OptimizerValue maxWeightNorm,
       AdamMode adamMode,
       WeightDecayMode weightDecayMode,
       DataType accumType,
       DataType accl1Type,
       DataType accl2Type,
       const std::vector<ClipNormSettings> &clipNormSettings = {},
       bool scaledOptimizerState                             = false);

  // Equivalent to calling Adam(defaultLearningRate, defaultWeightDecay,
  // defaultBeta1, defaultBeta2, defaultEps, lossScaling,
  // Adam::getUnsetMaxWeightNorm(), adamMode, weightDecayMode, accumType,
  // accl1Type, accl2Type, clipNormSettings).
  Adam(OptimizerValue defaultLearningRate,
       OptimizerValue defaultWeightDecay,
       OptimizerValue defaultBeta1,
       OptimizerValue defaultBeta2,
       OptimizerValue defaultEps,
       OptimizerValue lossScaling,
       AdamMode adamMode,
       WeightDecayMode weightDecayMode,
       DataType accumType,
       DataType accl1Type,
       DataType accl2Type,
       const std::vector<ClipNormSettings> &clipNormSettings = {},
       bool scaledOptimizerState                             = false);

  // Equivalent to calling Adam(defaultLearningRate, defaultWeightDecay,
  // defaultBeta1, defaultBeta2, defaultEps, lossScaling,
  // maxWeightNorm, adamMode, WeightDecayMode::Decay, accumType,
  // accl1Type, accl2Type, clipNormSettings).
  Adam(OptimizerValue defaultLearningRate,
       OptimizerValue defaultWeightDecay,
       OptimizerValue defaultBeta1,
       OptimizerValue defaultBeta2,
       OptimizerValue defaultEps,
       OptimizerValue lossScaling,
       OptimizerValue maxWeightNorm,
       AdamMode adamMode,
       DataType accumType,
       DataType accl1Type,
       DataType accl2Type,
       const std::vector<ClipNormSettings> &clipNormSettings = {},
       bool scaledOptimizerState                             = false);

  // Equivalent to calling Adam(defaultLearningRate, defaultWeightDecay,
  // defaultBeta1, defaultBeta2, defaultEps, lossScaling,
  // Adam::getUnsetMaxWeightNorm(), adamMode, WeightDecayMode::Decay,
  // accumType, accl1Type, accl2Type, clipNormSettings).
  Adam(OptimizerValue defaultLearningRate,
       OptimizerValue defaultWeightDecay,
       OptimizerValue defaultBeta1,
       OptimizerValue defaultBeta2,
       OptimizerValue defaultEps,
       OptimizerValue lossScaling,
       AdamMode adamMode,
       DataType accumType,
       DataType accl1Type,
       DataType accl2Type,
       const std::vector<ClipNormSettings> &clipNormSettings = {},
       bool scaledOptimizerState                             = false);

  /// Constructor.
  /// \param params A parameter map where keys are one of
  ///     `"defaultLearningRate"`, `"defaultWeightDecay"`, `"defaultBeta1"`,
  ///     `"defaultBeta2"`, `"defaultEps"`, `"lossScaling"` or
  ///     `"maxWeightNorm"`, and the map's values pairs of floats and booleans
  ///     representing OptimizerValue constructor arguments. The map does not
  ///     have to specify each hyper parameter as default values will be used
  ///     where parameters are missing.
  /// \param adamMode The AdamMode value to use.
  /// \param weightDecayMode The WeightDecayMode value to use.
  /// \param maxWeightNorm The maxWeightNorm value to use.
  /// \param accumType Data type to use for gradient accumulation.
  /// \param accl1Type Data type to use for tensor that stores first-order
  ///     momentum optimizer state.
  /// \param accl2Type Data type to use for tensor that stores second-order
  ///     momentum optimizer state.
  /// \param clipNormSettings A vector of ClipNormSettings (this can be used
  ///     to set maximum values for weights).
  /// \param scaledOptimizerState Experimental Option.
  ///     Does not remove lossScaling before updating the optimizer state. This
  ///     should have no effect on the update equation. However, it does ensure
  ///     a more numerically stable implementation when accl1_type is set to
  ///     DataType::FLOAT16. Note: When loading a model that includes
  ///     initialised optimizer state, ensure that accl1 and accl2 are scaled by
  ///     lossScaling and lossScaling^2 respectively.
  ///
  /// **EXAMPLE**:
  /// ```
  /// Adam({{"defaultLearningRate", {0.02, False}},
  ///       {"defaultBeta1", {0.9, True}},
  ///       {"defaultBeta2":{0.999, True}}},
  ///       AdamMode::Adam,
  ///       WeightDecayMode::Decay,
  ///       DataType::FLOAT,
  ///       DataType::FLOAT,
  ///       DataType::FLOAT);
  /// ```
  Adam(const std::map<std::string, std::pair<float, bool>> &params,
       AdamMode adamMode,
       WeightDecayMode weightDecayMode,
       DataType accumType,
       DataType accl1Type,
       DataType accl2Type,
       const std::vector<ClipNormSettings> &clipNormSettings = {},
       bool scaledOptimizerState                             = false);

  static Adam fromDefaultMap(const std::map<std::string, OptimizerValue> &,
                             AdamMode adamMode_,
                             WeightDecayMode decayMode_,
                             DataType accumType_,
                             DataType accl1Type_,
                             DataType accl2Type_);

  Adam(const Adam &) = default;
  ~Adam()            = default;

  OptimizerType type() const final { return OptimizerType::Adam; }
  std::string type_s() const final { return "Adam"; }

  std::unique_ptr<Optimizer> clone() const final;

  std::unique_ptr<Op> createOp(const Tensor &weight, Graph &) const final;

  /// The names of the inputs for the VarUpdateOp for the Variable Tensor
  /// "weight". In the returned vector, an empty string ("") is used as a
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
  /// \param beta1 The beta1 value to use for this specific
  ///     weight.
  /// \param beta2 The beta2 value to use for this specific
  ///     weight.
  /// \param eps The epsilon value to use for this
  ///     specific weight.
  void insertSpecific(const TensorId &weight,
                      OptimizerValue learningRate,
                      OptimizerValue weightDecay,
                      OptimizerValue beta1,
                      OptimizerValue beta2,
                      OptimizerValue eps);

  void setStep(int64_t step);
  void setStep(const TensorId &, int64_t step);
  void setStep(std::map<TensorId, int64_t> steps);

  // insert OptimizerValues specific to one Tensor. The keys of the map should
  // be the names of atomic optimizer scalars, such as "momentum",
  // "learningRate". The map does not need to be complete. If it is not
  // complete, the default values already set for the SGD will be used.

  /// Insert a weight-specific set of hyper parameters.
  /// \param weight The TensorId of the weight.
  /// \param params A parameter map where keys are one of
  ///     `"defaultLearningRate"`, `"defaultWeightDecay"`, `"defaultBeta1"`,
  ///     `"defaultBeta2"`, `"defaultEps"`, `"lossScaling"` or `"maxWeightNorm"`
  ///     and the map's values pairs of floats and booleans representing
  ///     OptimizerValue constructor arguments. The map does not have to
  ///     specify each hyper parameter as default values will be used where
  ///     parameters are missing.
  void
  insertSpecific(const TensorId &weight,
                 const std::map<std::string, std::pair<float, bool>> &params);

  const OptimizerValueMap &learningRates() const { return lrs; }
  const OptimizerValueMap &weightDecays() const { return wds; }
  const OptimizerValueMap &beta1s() const { return b1s; }
  const OptimizerValueMap &beta2s() const { return b2s; }
  const OptimizerValueMap &epss() const { return epsvs; }
  const OptimizerValueMap &maxWeightNorms() const { return mwns; }

  const WeightDecayMode &getWeightDecayMode() const { return decayMode; }
  bool useScaledOptimizerState() const { return scaledOptimizerState; }

  size_t hash() const final;

  void setFactorsFromOptions(const SessionOptions &) final;

private:
  void runValueChecks(OptimizerValue lr,
                      OptimizerValue wd,
                      OptimizerValue b1,
                      OptimizerValue b2,
                      OptimizerValue eps) const;

  // The atomic scalars
  // ------------------
  // learning rates
  OptimizerValueMap lrs;

  // weight decays
  OptimizerValueMap wds;

  // beta 1s
  OptimizerValueMap b1s;

  // beta 2s
  OptimizerValueMap b2s;

  // eps values
  OptimizerValueMap epsvs;

  // max weight norm.
  OptimizerValueMap mwns;

  // Adam settings
  AdamMode mode;
  WeightDecayMode decayMode;
  DataType accumType;
  DataType accl1Type;
  DataType accl2Type;
  bool scaledOptimizerState;

  // The compound scalars
  // --------------------
  AdamLearningRateHelper lrhelper;
  AdamWeightDecayHelper wdhelper;
  AdamBeta1Helper b1helper;
  AdamBeta2Helper b2helper;
  AdamEpsHelper epshelper;
  AdamLossScalingHelper lshelper;
  AdamMaxWeightNormHelper mwnhelper;
  AdamGradientScalingHelper gshelper;

  Adam(const std::map<std::string, OptimizerValue> &,
       AdamMode mode_,
       WeightDecayMode decayMode_,
       DataType accumType_,
       DataType accl1Type_,
       DataType accl2Type_,
       const std::vector<ClipNormSettings> &clipNormSettings,
       bool scaledOptimizerState_);

  static std::map<std::string, OptimizerValue>
  getComplete(const std::map<std::string, OptimizerValue> &);
};

} // namespace popart

#endif
