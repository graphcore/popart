// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_OPTIMIZER_HPP
#define GUARD_NEURALNET_OPTIMIZER_HPP

#include <memory>
#include <popart/compoundscalarhelper.hpp>
#include <popart/names.hpp>
#include <popart/optimizervalue.hpp>
#include <popart/optimizervaluemap.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

namespace popart {

struct SessionOptions;

/// Types of optimizers.
enum class OptimizerType { SGD = 0, Adam, Adaptive, NTYPES };

/// Replicated graph reduction mode (data parallel optimizer).
/// Determines which replicated collective operations are inserted into the
/// graph.
enum class OptimizerReductionType {
  /// No replicated graph reduction
  None = 0,
  /// Gradient reduction (every iteration, after a weight's gradient is
  /// produced)
  GradReduce,
  /// Momentum reduction (SGD1, every N-th iteration, gradient accumulation)
  AcclReduce,
  /// Accumulator reduction (Adam, every N-th iteration, gradient accumulation)
  AccumReduce
};

/***
 * Enum type for different types of weight decay.
 */
enum class WeightDecayMode {
  /// Weight decay (e.g. AdamW)
  Decay,
  /// L2 regularization (e.g. PyTorch-like Adam)
  L2Regularization
};

std::map<std::string, OptimizerValue>
getOptMap(const std::map<std::string, std::pair<float, bool>> &m);
/**
 * A data structure used to represent a maximum value constaint on
 * one or more weights.
 */
struct ClipNormSettings {
  /// Constructor.
  /// \param weightIds_ The weight tensor IDs that this constraint
  ///     applies to.
  /// \param maxNorm_ The maximum permissible value.
  ClipNormSettings(const std::vector<TensorId> &weightIds_, float maxNorm_)
      : weightIds(weightIds_), maxNorm(maxNorm_) {}

  std::vector<TensorId> weightIds;
  float maxNorm;

  bool operator==(const ClipNormSettings &) const;
  bool operator!=(const ClipNormSettings &other) const {
    return !(*this == other);
  }
};

class optimizer_replacement_error : public error {
public:
  template <typename... Args>
  explicit optimizer_replacement_error(const char *s, const Args &...args)
      : error(std::string("New optimizer is not a valid replacement. ") + s,
              args...) {}

  template <typename... Args>
  explicit optimizer_replacement_error(const std::string &s,
                                       const Args &...args)
      : error("New optimizer is not a valid replacement. " + s, args...) {}
};

/// The base Optimizer class
class Optimizer {
public:
  virtual ~Optimizer() = default;
  Optimizer(OptimizerValue lossScaling,
            const std::vector<ClipNormSettings> &clipNormSettings);
  Optimizer(const Optimizer &) = default;

  // If true, a graph that has been constructed with this optimizer can be
  // updated with \p other, without requiring a change to the compute graph.
  // For example, a VarUpdateOp which has a constant scaled learning rate
  // cannot be modified to have a variable scaled learning rate.
  virtual void validReplacement(const Optimizer &other) const;

  virtual OptimizerType type() const               = 0;
  virtual std::string type_s() const               = 0;
  virtual std::unique_ptr<Optimizer> clone() const = 0;

  // Set the data in tensor from a relevant value stored by this optimizer.
  // The particular value used is determined from the tensor's name and type.
  virtual void resetTensorData(Tensor &) const = 0;
  virtual void setTensorData(Tensor &) const   = 0;

  // Create a VarUpdateOp for a specific weight tensor using this optimizer,
  // and get the names of inputs to the VarUpdateOp for a specific tensor.
  virtual std::unique_ptr<Op> createOp(const Tensor &weight, Graph &) const = 0;

  virtual std::vector<TensorId> getInputIds(const Tensor &weight) const = 0;

  // Unique non-constant optimizers.
  virtual std::vector<std::tuple<TensorId, TensorInfo>>
  getOptimizerInputs(const Tensor &weight) const = 0;

  const OptimizerValue &lossScaling() const { return ls; }
  float getLossScalingVal() const { return ls.val(); }

  static TensorId getLossScalingTensorId(DataType);

  void setFactorsFromOptions(const SessionOptions &);

  bool gradientAccumulationEnabled() const;
  bool meanGradientAccumulationEnabled() const;
  int64_t getReplicatedGraphCount() const;
  int64_t getAccumulationFactor() const;

  const std::vector<ClipNormSettings> &getClipNormSettings() const {
    return clipNormSettings;
  }

  virtual size_t hash() const;

protected:
  // T can be of type OptimizerValue or OptimizerValueMap.
  template <typename T>
  void checkReplacementValue(const T &thisVal,
                             const T &replacementVal,
                             const char *valueName) const {
    logging::ir::debug("Checking {} for compatibility", valueName);
    try {
      thisVal.validReplacement(replacementVal);
    } catch (error &err) {
      std::string error_message =
          logging::format("Problem with {}. {}", valueName, err.what());
      throw optimizer_replacement_error(error_message);
    }
  }

private:
  OptimizerValue ls;
  std::vector<ClipNormSettings> clipNormSettings;

  bool enableGradientAccumulation;
  bool meanGradientAccumulation;
  int64_t accumulationFactor;
  int64_t replicatedGraphCount;

  bool factorsAreSetFromOptions{false};
};

// Equation derivation based on the non-Nesterov PyTorch implementation
// https://PyTorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD :
//
// g = gradient computed in backwards pass
// g = g + wd * w
// v = v * mm + (1 - dm) * g
// w = w - lr * v
//
// which is equivalent to
//
// g = gradient computed in backwards pass
// v = v * mm + (1 - dm) * g + (1 - dm) * wd * w
// w = w - lr * v
//
// if we include loss scaling, we factorise ls out of g before updating v:
//
// g = gradient computed in backwards pass * ls
// v = v * mm + (1 - dm) / ls * g + (1 - dm) * wd * w
// w = w - lr * v
//
// if we want to keep velocity (v) a factor vs larger throughout for numerical
// stability reasons, we
// (1) multiply the term added to it by scalar factor vs
// (2) make sure it is initialised with a factor vs larger (T12001)
// (3) divide lr by vs:
//
// v = v * mm + (1 - dm) * vs / ls * g + (1 - dm) * wd * vs * w
// w = w - lr / vs * v.
//
// if there is gradient accumulation, this becomes:
//
// v = v * mm + (1 - dm) * vs / ls * sum_micro_batches(g) +
//                                                  + (1 - dm) * wd * vs * w
// w = w - lr / vs * v.
//
// which has 2 parts, one part in the loop:
//    v <- v + (1 - dm) * vs / ls * g_i for each micro batch i's gradient
//    =    =                        =
//
// and one part out the loop:
//    w <- w - lr / vs * v
//    v <- v * mm + (1 - dm) * wd * vs * w.   (done once up front too,
//    =    =                             =               see test comments)
//
//
// if in addition there is data replication by factor rf, the equations become;
// in the loop:
//    v <- v + (1 - dm) * vs * rf / ls * g_i                        (include rf)
//    =    =                             =
//
// and outside the loop:
//    v <- sum reduce across IPUs of the v's                  (rf too large now)
//
//    w <- w - lr / ( vs * rf ) * v            (rf in denominator to compensate)
//    =    =                      =
//
//    v <- v * mm / rf + (1 - dm) * wd * vs * w.              (correction by rf)
//    =    =                                  =
//
// where the scalar factors corresponding to PyTorch are,
//   mm : momentum
//   dm : dampening
//   wd : weight decay
//   lr : learning rate
//
// the optional scaling factors to improve numerical stability are
//   ls : loss scaling
//   vs : velocity scaling
//
// and the terms to accelerate training is
//   rf : data replication factor.
//   af : gradient accumulation factor.
//
// If accumulationAndReplicationReductionType is set to ReductionType::Sum
// 'af' is set to 1
//
// Special case a)
// --------------
// In the case where there IS NO gradient accumulation and there IS NO momentum
// (mm = 0), there is no need for a persistant v Tensor, and the weight update
// reduces to,
//
// w <- w * {1 -  lr * (1 - dm) * wd} -  g * { lr * (1 - dm) / ls }   (1)
//          ^^^^^^^^^^^^^^^^^^^^^^^^^        ~~~~~~~~~~~~~~~~~~~~~~
//                    |                               |
//   weight decay scale factor 0                      |
//                                         scaled learning rate 0
//
// In this simple special case, everything is done in a single Op of type
// SGD0VarUpdateOp.
//
// Note that all compound scalar terms are always calculated on host.
//
// To summarise, there are atomic scalars and compound scalars.
//                         ------             --------
//
// The atomic scalars are mm, dm, wd, lr, ls, vs, rf, af.
//
// The compound scalars for the simple case of no persistent v tensor are,
//
// Compound scalars for the case where there is no gradient accumulation and no
// momentum (SGD0):
//
//  - weightDecayScaleFactor0 (wdsf0) =
//      1 - lr * (1 - dm) * wd
//
//  - scaledLearningRate0 (slr0) =
//      lr *  ( 1 - dm) / ls
//
// Compound scalars for the case where there is gradient accumulation and
// momentum (SGD1):
//                                            mm dm wd lr ls vs rf af
//                                            =======================
//  - scaledWeightDecay1 (swd1) =             .  x  x  .  .  x  .  .
//      (1 - dm) * wd * vs
//
//  - dampeningScaleFactor1 (dpsf1) =         .  x  .  .  x  x  x  x
//      (1 - dm) * vs * rf / (ls * af)
//
//  - scaledLearningRate1 (slr1) =            .  .  .  x  .  x  x  .
//      lr / ( vs * rf)
//
//  - scaledMomentum1 (smm1) =                x  .  .  .  .  .  .  .
//      mm / rf
//
//
// Note that the user sets atomic scalars (and not compound scalars)
//
// Note that all atomic scalar terms except loss scaling and replication factor
// can be Tensor specific.
//
// Constructing an SGD Optimizer is done in 2 steps;
//
// (1) Construct SGD with default values
// (2) Set Tensor specific values
//
// Any OptimizerValue can be set as isConst if it will not change during
// training. This can result in faster/smaller code. For a compound scalar to be
// isConst, all of its constituent atomic scalars must be isConst
//
// Currently rf != 1 is not supported for the case where mm != 0. The plan for
// enabling this: (1) make 1 Op which updates both w and g, i.e. does everything
// outside the loop. (2) support aliasing and modifying Ops with more than 1
// output. T12001 (above)
//
//
//          [dpfs1]
// [v]-|       |
//     |-(Accumulation)--[v']--(AcclReduce)--[v'']   [w]
// [g]-|                                       |  \/  |
//                                             |  /\  |  [swd1]
//                                             | /  \ |    |
//                              [slr1]--(VarUpdate)(AcclUpdate)-[smm1]
//                                            |       |
//                                           [w']   [v''']
//
// Note that ReplicationReduction will be a nop if replFactor = 1.
//

/**
 * Stochastic Gradient Descent (%SGD) optimizer.
 *
 * Akin to any optimizer implementation, this class is responsible for updating
 * each weight tensor (\f$w\f$) in the model using the gradient (\f$g\f$) of
 * the loss function with respect to the weight as calculated during the
 * backwards pass.
 *
 * The %SGD optimizer has the following **state** for each weight:
 *
 *  * *velocity* (\f$v\f$)
 *
 * The %SGD optimizer has the following **hyper parameters**:
 *
 *  * *learning rate* (\f$\text{lr}\f$)
 *  * *momentum* (\f$\text{mm}\f$)
 *  * *weight decay* (\f$\text{wd}\f$)
 *  * *dampening* (\f$\text{dm}\f$)
 *  * *velocity scaling* (\f$\text{vs}\f$)
 *  * *loss scaling* (\f$\text{ls}\f$)
 *  * *clip norm settings*
 *
 * The values of these parameters can be shared between all weights but some
 * can be overridden with weight-specific values (see SGD::insertSpecific).
 * Hyper parameters are captured using OptimizerValue objects and therefore
 * can be either a constant value or a non-constant value that can be adjusted
 * by the user.
 *
 * In the following we will describe how this optimizer updates a weight
 * using a gradient. In the context of this description the gradient is
 * is the value of the gradient *after* any gradient accumulation has been
 * performed and *after* the application of a loss scaling factor to the
 * gradient has been corrected for.
 *
 * When the optimizer needs to update a weight, \f$w\f$, using a gradient,
 * \f$g\f$, it first updates the optimizer state as follows:
 *
 * \f[
 *    v' := v * \text{mm} + (1 - \text{dm}) * (g + \text{wd} * w) \text{ \ . }
 * \f]
 *
 * Following the update of the optimizer state the optimizer uses said
 * state to update the weight:
 *
 * \f[
 *    w' := w - \text{lr} * v' \text{ \ . }
 * \f]
 *
 * In addition to the above, the *velocity scaling* hyper parameter is a scaling
 * factor that can provide improved numerical stability by ensuring the values
 * stored in the optimizer state, \f$v\f$, are scaled by this value. When using
 * this parameter PopART will automatically deal with the artificially scaled
 * velocity value during the weight update and other hyper parameters do not
 * need to be adjusted).
 *
 * In addition, the *loss scaling* hyper parameter is similar in nature to the
 * velocity scaling parameter. It is a scaling value that is applied to the loss
 * gradient at the start of the the backwards pass and, at the end of the
 * backwards pass, this scaling is reversed by multiplying the gradients for
 * each weight with the inverse of the loss scaling value prior to updating the
 * optimizer state. Using loss scaling can also improve numerical stability in
 * some cases.
 *
 * Finally, it is possible to add clip norm settings for this optimizer. These
 * clip norms compute the L2 norm for a group of weights and adds a scalar term
 * to the weight update that effectively divides it by the norm (or a constant
 * value that is provided as part of the clip norm, which ever is greater).
 */
class SGD : public Optimizer {

public:
  /// Default learning rate value.
  static OptimizerValue getUnsetLearningRate() {
    return {0.1f, true}; // a learning rate of 0.1 forever
  }

  /// Default weight decay value.
  static OptimizerValue getUnsetWeightDecay() {
    return {0.0f, true}; // no weight decay, ever
  }

  /// Default momentum value.
  static OptimizerValue getUnsetMomentum() {
    return {0.0f, true}; // no momentum, ever
  }

  /// Default dampening value.
  static OptimizerValue getUnsetDampening() {
    return {0.0f, true}; // no dampening, ever
  }

  /// Default velocity scaling value.
  static OptimizerValue getUnsetVelocityScaling() {
    return {1.0f, true}; // no velocity scaling, ever
  }

  /// Default loss scaling value.
  static OptimizerValue getUnsetLossScaling() {
    return {1.0f, true}; // no loss scaling, ever
  }

public:
  // Returns true if \p w has specific OptimizerValues, false if it will use
  // the default.
  bool hasSpecific(const Tensor &w) const;

  /// Constructor.
  /// \param defaultLearningRate The learning rate value to use for weights
  ///     for which no weight-specific hyper parameter have been inserted.
  /// \param defaultWeightDecay The weight decay value  to use for weights
  ///     for which no weight-specific hyper parameter have been inserted.
  /// \param defaultMomentum The momentum value to use for weights
  ///     for which no weight-specific hyper parameter have been inserted.
  /// \param defaultDampening The dampening value to use for weights
  ///     for which no weight-specific hyper parameter have been inserted.
  /// \param defaultVelocityScaling The velocity scaling value to use for
  ///     weights for which no weight-specific hyper parameter have been
  ///     inserted.
  /// \param lossScaling The loss scaling value to use.
  /// \param clipNormSettings A vector of ClipNormSettings (this can be used
  ///     to set maximum values for weights).
  SGD(OptimizerValue defaultLearningRate,
      OptimizerValue defaultWeightDecay,
      OptimizerValue defaultMomentum,
      OptimizerValue defaultDampening,
      OptimizerValue defaultVelocityScaling,
      OptimizerValue lossScaling,
      const std::vector<ClipNormSettings> &clipNormSettings = {});

  /// Constructor.
  /// \param params A parameter map where the keys are one or more of
  ///     `"defaultLearningRate"`, `"defaultWeightDecay"`, `"defaultMomentum"`,
  ///     `"defaultDampening"`, `"defaultVelocityScaling"` or `"lossScaling"`.
  ///     The map's values are pairs of floats and booleans representing
  ///     OptimizerValue constructor arguments. The map does not have to
  ///     specify each hyper parameter because default values will be used where
  ///     parameters are missing.
  /// \param clipNormSettings A vector of ClipNormSettings (this can be used
  ///     to set maximum values for weights).
  ///
  /// **EXAMPLE**:
  /// ```
  /// SGD({{"defaultLearningRate", {0.02, False}},
  ///     {"defaultMomentum":{0.6, True}}});
  /// ```
  /// This will create an SGD Optimizer which has a constant momentum of 0.6 and
  /// a changeable learning rate initially of 0.02. All OptimizerValues not
  /// present in the map will take values from the `getUnset`* functions.
  SGD(const std::map<std::string, std::pair<float, bool>> &params,
      const std::vector<ClipNormSettings> &clipNormSettings = {});
  static SGD fromDefaultMap(const std::map<std::string, OptimizerValue> &);

  /// Construct an SDG instance with default values.
  SGD(const SGD &) = default;
  ~SGD()           = default;

  OptimizerType type() const final { return OptimizerType::SGD; }
  std::string type_s() const final { return "SGD"; }

  std::unique_ptr<Optimizer> clone() const final;

  std::unique_ptr<Op> createOp(const Tensor &weight, Graph &) const final;

  /// The names of the inputs for the VarUpdateOp for the variable tensor
  /// \p weight. In the returned vector, an empty string ("") is used as a
  /// placeholder for constant inputs.
  std::vector<TensorId> getInputIds(const Tensor &weight) const final;

  /// The names and information for the optimizer tensors.
  std::vector<std::tuple<TensorId, TensorInfo>>
  getOptimizerInputs(const Tensor &weight) const final;

  void validReplacement(const Optimizer &other) const final;

  void resetTensorData(Tensor &) const final;
  void setTensorData(Tensor &) const final;

  /// Tensor "opt" has an id, which it uses to match a compound scalar which
  /// this object can compute from the atomic scalars.
  float getStoredValue(const TensorId &optId) const;

  /// Insert a weight-specific set of hyper parameters.
  /// \param weight The TensorId of the weight.
  /// \param learningRate The learning rate value to use for this specific
  ///     weight.
  /// \param weightDecay The weight decay value to use for this specific
  ///     weight.
  /// \param momentum The momentum value to use for this specific
  ///     weight.
  /// \param dampening The dampening value to use for this specific
  ///     weight.
  /// \param velocityScaling The velocity scaling value to use for this
  ///     specific weight.
  void insertSpecific(const TensorId &weight,
                      OptimizerValue learningRate,
                      OptimizerValue weightDecay,
                      OptimizerValue momentum,
                      OptimizerValue dampening,
                      OptimizerValue velocityScaling);

  /// Insert a weight-specific set of hyper parameters.
  /// \param weight The TensorId of the weight.
  /// \param params A parameter map where keys are one of
  ///     `"defaultLearningRate"`, `"defaultWeightDecay"`, `"defaultMomentum"`,
  ///     `"defaultDampening"`, `"defaultVelocityScaling"` or `"lossScaling"`
  ///     and the map's values pairs of floats and booleans representing
  ///     OptimizerValue constructor arguments. The map does not have to
  ///     specify each hyper parameter as default values will be used where
  ///     parameters are missing.
  void
  insertSpecific(const TensorId &weight,
                 const std::map<std::string, std::pair<float, bool>> &params);

  /// If velocity (accumulation) is required, either because of gradient
  /// accumulation or because of momentum, then return true otherwise return
  /// false.
  bool requiresAccl(const Tensor &weight) const;

  const OptimizerValueMap &learningRates() const { return lrs; }
  const OptimizerValueMap &weightDecays() const { return wds; }
  const OptimizerValueMap &momentums() const { return mms; }
  const OptimizerValueMap &dampenings() const { return dps; }
  const OptimizerValueMap &velocityScalings() const { return vss; }

  virtual size_t hash() const;

private:
  void runValueChecks(OptimizerValue lr,
                      OptimizerValue wd,
                      OptimizerValue mm,
                      OptimizerValue dp,
                      OptimizerValue vs) const;

  // The atomic scalars
  // ------------------
  // learning rates
  OptimizerValueMap lrs;

  // weight decays
  OptimizerValueMap wds;

  // momentums
  OptimizerValueMap mms;

  // dampenings
  OptimizerValueMap dps;

  // velocity scalings
  OptimizerValueMap vss;

  // The compound scalars
  // --------------------
  // No Accumulation Tensor needed (SGD0)
  ScaledLearningRate0Helper slr0helper;
  WeightDecayScaleFactor0Helper wdsf0helper;

  // Accumulation Tensor needed (SGD1)
  ScaledLearningRate1Helper slr1helper;
  ScaledWeightDecay1Helper swd1helper;
  DampeningScaleFactor1Helper dpsf1helper;
  ScaledMomentum1Helper smm1helper;

  // int argument only to disambiguate from the other SGD constructor
  SGD(const std::map<std::string, OptimizerValue> &,
      const std::vector<ClipNormSettings> &,
      int);

  static std::map<std::string, OptimizerValue>
  getComplete(const std::map<std::string, OptimizerValue> &);
};

/**
 * Stochastic Gradient Descent (SGD) optimizer with constant learning rate,
 * weight decay, loss scaling and clip norm settings (and default values for
 * momentum, dampening or velocity scaling).
 *
 * **NOTE**: See SGD for detailed meaning for these parameters.
 *
 * **NOTE**: This class exists for backwards compatibility with the Python API
 * and may be removed at some point in the future.
 */
class ConstSGD : public SGD {
public:
  /// Constructor.
  /// \param learningRate A constant learning rate.
  /// \param weightDecay A constant weight decay value.
  /// \param lossScaling A constant loss scaling value.
  /// \param clipNormSettings A vector of ClipNormSettings (this can be used
  ///     to set maximum values for weights).
  ConstSGD(float learningRate,
           float weightDecay                                     = 0,
           float lossScaling                                     = 1,
           const std::vector<ClipNormSettings> &clipNormSettings = {})
      : SGD({learningRate, true},
            {weightDecay, true},
            getUnsetMomentum(),
            getUnsetDampening(),
            getUnsetVelocityScaling(),
            {lossScaling, true},
            clipNormSettings) {}
};

} // namespace popart

namespace std {
template <> struct hash<popart::ClipNormSettings> {
  std::size_t operator()(const popart::ClipNormSettings &settings) const;
};

template <> struct hash<popart::Optimizer *> {
  std::size_t operator()(const popart::Optimizer *opt) const {
    return opt->hash();
  }
};

} // namespace std

namespace popart {
inline std::size_t hash_value(const ClipNormSettings &settings) {
  return std::hash<ClipNormSettings>()(settings);
}
} // namespace popart

#endif
