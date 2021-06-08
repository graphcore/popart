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

/// Reduction mode when doing data-parallel training over replicated graphs.
///
/// Depending on the optimizer used and its configuration, this option describes
/// how the reduction of gradients over replicas will occur. For example,
/// directly on the gradient, on the gradient accumulator, or on the momentum.
/// See the documentation of individual optimizers for more information.
enum class OptimizerReductionType {
  /// No replicated graph reduction
  None = 0,
  /// Gradient reduction (every iteration, after a weight's gradient is
  /// produced)
  GradReduce,
  /// Momentum reduction (SGD1, after the gradient accumulation loop, if
  /// applicable)
  AcclReduce,
  /// Accumulator reduction (Adam/SGD2 + gradient accumulation, after the
  /// gradient accumulation loop)
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
 * A data structure used to represent a maximum value constraint on
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
  explicit optimizer_replacement_error(const char *s, const Args &... args)
      : error(std::string("New optimizer is not a valid replacement. ") + s,
              args...) {}

  template <typename... Args>
  explicit optimizer_replacement_error(const std::string &s,
                                       const Args &... args)
      : error("New optimizer is not a valid replacement. " + s, args...) {}
};

/// Interface for describing an Optimizer and, internally, how to grow the
/// optimiser step for each weight.
///
///  - The end-user facing interface constructed by the user to describe what
///    kind of optimiser to use.
///  - Then also used internally by the Ir to grow the optimiser step for each
///    weight.
///  - Stores OptimizerValues for optimizer parameters like learning rate,
///    loss scaling, etc. \sa OptimiserValue.
///  - Optimizer stores the values for each weight - they can have different
///    values. There is a "default" for all weights, then you can specify
///    specific values for specific weights. This is encapsulated by an
///    OptimizerValueMap, which is a sparse map from weight to value, with
///    unspecified values implying the default. \sa OptimizerValueMap.
///  - At runtime, the user can dynamically update the Optimizer, e.g. by
///    setting new OptimizerValues. validReplacement determines whether the
///    new Optimizer is interchangable with the one the Ir was built for. For
///    example, trying to replace an SGD Optimizer with an Adam Optimizer
///    would throw.
class Optimizer {
  ///  - Optimizer class has a two-part initialisation. The ctor, used by the
  ///    end-user, and setFactorsFromOptions called by the Ir to finish
  ///    initialisation once we have all the relevant information during Ir
  ///    preparation.
  ///  - Some key methods used by the Ir to grow optimiser step for each weight
  ///    are createOp, getInputIds, optimizerInputs.
  ///  - If the OptimizerValue is const, no Ir tensor for that value is
  ///    created and the VarUpdateOp created for that weight will not have the
  ///    optional input for that tensor. The Opx of the VarUpdateOp will emit
  ///    poplar code that uses the provided value directly.
  ///
  ///    If the OptimizerValue is not const, an Ir tensor for that value is
  ///    created and the VarUpdateOp created for that weight will have the
  ///    optional input for that tensor. The tensor will be a stream tensor, so
  ///    that it can be updated later from host. The tensor will be streamed an
  ///    initial value of the OptimizerValue's value.
  ///  - It is common for Optimizer implementations to make use of "compound
  ///    scalars". Take for example the SGD0 weight update equation:
  ///        w <- w * (1 - lr * (1 - dm) * wd) -  g * (lr * (1 - dm) / ls)
  ///    w is the weights and g is the grads. lr, dm, wd, ls are all the
  ///    "atomic scalars". These are the scalars/hyperparameters of the
  ///    Optimizer that the user can set using OptimizerValues, as described
  ///    above.
  ///
  ///    Multiple atomic scalars appear in expressions together, and will be
  ///    operated on together before being used by an Op that also consumes a
  ///    tensor (in this case the weights or grads). For SGD0, they can be
  ///    grouped as follows:
  ///
  ///        w <- w * {1 -  lr * (1 - dm) * wd} -  g * { lr * (1 - dm) / ls }
  ///                 ^^^^^^^^^^^^^^^^^^^^^^^^^        ~~~~~~~~~~~~~~~~~~~~~~
  ///                            |                               |
  ///           weight decay scale factor 0                      |
  ///                                                   scaled learning rate 0
  ///
  ///    We call wdsf0 and slr0 the "compound scalars".
  ///
  ///    We can statically precompute the OptimizerValues for these compound
  ///    scalars using the OptimizerValues of the atomic scalars. This makes
  ///    the Ir simpler, as we now have only:
  ///
  ///        w <- w * wdsf0 - g * slr0
  ///
  ///    The CompoundScalarHelpers are used to precompute the compound scalar
  ///    values. \sa compoundscalarhelper.hpp
  ///
  ///    If any of the composite atomic scalars are non-const, the compound
  ///    scalar is non-const.
public:
  virtual ~Optimizer() = default;
  Optimizer(OptimizerValue lossScaling,
            const std::vector<ClipNormSettings> &clipNormSettings);
  Optimizer(const Optimizer &) = default;

  // If a graph that has been constructed with this optimizer can be
  // updated with \p other without requiring a change to the compute graph, does
  // nothing. Otherwise, throws optimizer_replacement_error.
  //
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

  // Create a VarUpdateOp for a specific \p weight tensor using this optimizer,
  // and get the names of inputs to the VarUpdateOp for a specific tensor.
  virtual std::unique_ptr<Op> createOp(const Tensor &weight, Graph &) const = 0;

  /// Returns the TensorIds of the input tensors to the VarUpdateOp this
  /// optimiser will create for the given \p weight .
  ///
  /// Specifically, The TensorId at index i will be the id of the input tensor
  /// at InIndex i of the VarUpdateOp. If the input is an OptimizerValue, if it
  /// is const, then "" will be returned, else the relevant reservered prefix
  /// for that OptimizerValue will be used, followed by the weight id. The
  /// prefixes are defined in tensornames.hpp, for example
  /// `reservedDefaultWeightDecayScaleFactor0Prefix` or
  /// `reservedSpecificScaledLearningRate1Prefix` (note there are different
  /// prefixes depending on if the weight has a specific or default value for
  /// that OptimizerValue).
  virtual std::vector<TensorId> getInputIds(const Tensor &weight) const = 0;

  // The TensorId and TensorInfo of all OptimizerValue input tensors to the
  // VarUpdateOp this Optimizer will create for the \p weight , in any order.
  // If an OptimizerValue is const, it is not an input so is not included.
  virtual std::vector<std::tuple<TensorId, TensorInfo>>
  getOptimizerInputs(const Tensor &weight) const = 0;

  const OptimizerValue &lossScaling() const { return ls; }
  float getLossScalingVal() const { return ls.val(); }

  static TensorId getLossScalingTensorId(DataType);
  virtual TensorId
  // Either the scalar tensor representing the inverse loss scale factor, or
  // compound scalar tensor which contains the inverse loss scale factor
  getInverseLossScalingTensorId(const Tensor &weight) const = 0;

  virtual void setFactorsFromOptions(const SessionOptions &);

  bool gradientAccumulationEnabled() const;
  bool meanGradientAccumulationEnabled() const;
  int64_t getReplicatedGraphCount() const;
  int64_t getAccumulationFactor() const;

  const std::vector<ClipNormSettings> &getClipNormSettings() const {
    return clipNormSettings;
  }

  // Returns true if \p w has specific OptimizerValues, false if it will use
  // the default.
  virtual bool hasSpecific(const Tensor &w) const = 0;

  // Do any weights have specific OptimizerValues, or do they all use default?
  virtual bool hasSpecific() const = 0;

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

/**
 * Strategy for implementing SGD with momentum and/or gradient accumulation.
 */
enum class SGDAccumulatorAndMomentum {
  /// Implement SGD using a single tensor for the gradient accumulator (accum)
  /// and momentum (accl) tensors.
  Combined = 0,

  /// Implement SGD using separate tensors for the gradient accumulator (accum)
  /// and momentum (accl) tensors
  Separate
};

/**
 * Write a representation of an SGDAccumulatorAndMomentum to an output stream.
 *
 * \param os Output stream.
 * \param sgdAccMm SGDAccumulatorAndMomentum reference.
 * \return The same output stream for chaining.
 */
std::ostream &operator<<(std::ostream &os,
                         const SGDAccumulatorAndMomentum &sgdAccMm);

// The SGD Optimizer
// =================
//
// Key
// ---
// The optimiser hyperparameters are
//   mm : momentum
//   dm : dampening
//   wd : weight decay
//   lr : learning rate
//
// The optional scaling factors to improve numerical stability are
//   ls : loss scaling
//   vs : velocity scaling
//
// The terms to accelerate training are
//   rf : replication factor
//   af : gradient accumulation factor
//
// Basic SGD Update Equation
// -------------------------
// Based on the non-Nesterov PyTorch implementation
// https://PyTorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD , the SGD
// update equation with weight decay, dampening, and momentum, is
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
// Loss Scaling
// ------------
// Loss scaling means the loss gradient is scaled by ls before starting the
// backwards pass, then each individual weight gradient is un-scaled during the
// weight update. Thus, the weight update becomes:
//
// g = gradient computed in backwards pass (scaled by ls)
// v = v * mm + (1 - dm) / ls * g + (1 - dm) * wd * w
// w = w - lr * v
//
// Velocity Scaling
// ----------------
// If we want to keep velocity (v) a factor vs larger throughout for numerical
// stability reasons (known as velocity scaling), we
// (1) multiply the term added to it by scalar factor vs
// (2) make sure it is initialised with a factor vs larger (T12001)
// (3) divide lr by vs
//
// # Where v_0 is the usual initial v
// v = v_0 * vs
//
// v = v * mm + vs * ((1 - dm) / ls * g + (1 - dm) * wd * w)
// w = w - lr / vs * v
//
// The above v equation can be rewritten as:
// v = v * mm + (1 - dm) * vs / ls * g + (1 - dm) * wd * vs * w
//
// Note on (2): In most cases v is initialised to 0, so the initial v is still
// the same. However, this is not always the case, for example when using
// SGDAccumulatorAndMomentum::Combined (explained below).
//
// Gradient Accumulation
// ---------------------
// Recall gradient accumulation, where gradients are accumulated across multiple
// "micro-batches" in a loop before applying the update (now with the
// accumulator tensor instead of the individual gradient tensor).
//
// a = 0
// for each micro-batch
//   a += g
// v = v * mm + (1 - dm) * vs / ls * a + (1 - dm) * wd * vs * w
// w = w - lr / vs * v
//
// Where a is the "accumulator" tensor.
//
// Reduction over Replicas
// -------------------------
// Recall, in data parallel trainining, multiple replicas of the model are run
// in parallel, each with their own copies of the weights and optimiser state.
// Before applying the weight update, the replicas reduce their gradients
// (by summing them) so they always train with identical weights. This
// reduction across replicas with the result being copied back to all replicas
// is known as an "all-reduce".
//
// The optimiser step becomes:
//
// a = 0
// for each micro-batch
//   a += g
// allReduce(a)
// v = v * mm + (1 - dm) * vs / ls * a + (1 - dm) * wd * vs * w
// w = w - lr / vs * v
//
// The reduction performed is ALWAYS a sum, even though you can mean gradients
// over replicas (see below).
//
// Note, because there is gradient accumulation, it is more efficient to
// all-reduce the accumulator once, than each individual gradient in the loop.
// The SGD Optimizer class will infer this automatically and set
// OptimizerReductionType::AccumReduce on the optmimiser Op it creates. If there
// was no gradient accumulation, the individual gradient would be marked for
// reduction using OptimizerReductionType::GradReduce.
//
// Lastly, note that this is invariant to the replication factor rf. Training
// should be mathematically identical no matter the amount of replication.
//
// Mean Accumulation and Reduction of Gradients
// --------------------------------------------
// The above sections describe how gradients are summed across the gradient
// accumulation loop and across replicas. Alternatively, you can opt to have the
// gradients be mean-averaged across both of these. Note, both must be sum, or
// both must be mean; you cannot mix them.
//
// This is controlled by the SessionOption
// `ReductionType accumulationAndReplicationReductionType`. Setting this to
// ReductionType::Mean will cause the gradients to be mean-ed.
//
// To implement the mean over the gradient accumulation loop, we divide by a
// new scalar af in the v update, which is set to the accumulation factor only
// if using mean reduction, otherwise 1:
//
// a = 0
// for each micro-batch
//   a += g
// allReduce(a)
// v = v * mm + (1 - dm) * vs / (ls * af) * a + (1 - dm) * wd * vs * w
// w = w - lr / vs * v
//
// To implement the mean reduction over replicas, we perform the division by the
// replication factor rf on the loss gradient before starting the backward pass.
// This is mathematically identical to performing the division on the weight
// gradients directly. Thus, the all-reduce during the optimiser step is still a
// summation. The division of the loss gradient by rf is done by the Autodiff
// transform. \sa GradGrowerLoss .
//
// SGDAccumulatorAndMomentum
// -------------------------
// When doing gradient accumulation, momentum, or both, Popart has an
// optional optimisation that combines the accumulator and momentum tensors to
// reduce liveness. This is controlled by:
//    enum class SGDAccumulatorAndMomentum { Combined, Separate }
// The following will explain how the Combined strategy is implemented, and some
// subtle caveats of this versus the separate tensor implementation.
//
// Recall the basic update equations with no gradient accumulation or
// replication reduction, which is split up into three addition terms:
//
//    |- 1 -|   |------------ 2 ------------|   |---------- 3 -------|
// v = v * mm + (1 - dm) * vs / (ls * af) * g + (1 - dm) * wd * vs * w
// w = w - lr / vs * v
//
// Every time a weight is updated, we have all the information required to
// immediately pre-compute 1 and 3 for the next time step. Thus, the actual
// v update for that step only needs to do 2. The optimiser step becomes:
//
// Initial v:
// v = v_0 * mm + (1- dm) * wd * vs * w_0   (1 and 3)
//   where v_0 and w_0 are the initial v and w, respectively. v_0 is 0 unless
//   the Ir's Onnx ModelProto has an initialiser for v.
//
// Then:
// v = (1 - dm) * vs / (ls * af) * g        (2)
// w = w - lr / vs * v
// v = v * mm + (1 - dm) * wd * vs * w      (precompute 1 and 3 for next step)
//
// ----------------------------------------------
// Gradient Accumulation
//
// The benefit of this is apparent when we add in gradient accumulation:
//
// a = 0
// for each micro batch
//   a += g
// v = (1 - dm) * vs / (ls * af) * g
// w = w - lr / vs * v
// v = v * mm + (1 - dm) * wd * vs * w
//
// We can elide the accumulator tensor by directly accumulating into v:
//
// for each micro batch
//   v += (1 - dm) * vs / (ls * af) * g
// w = w - lr / vs * v
// v = v * mm + (1 - dm) * wd * vs * w
//
// ----------------------------------------------
// Replication
//
// If we try to add back replication:
//
// for each micro batch
//   v += (1 - dm) * vs / ls * g
// # There is no accumulator to reduce
// allReduce(??)
// w = w - lr / vs * v
// v = v * mm + (1 - dm) * wd * vs * w
//
// There is no longer an accumulator/grad tensor to all-reduce. All-reducing v
// is not mathematically equivalent. However, we can find a workaround:
//
// Recall that the all-reduce is always a sum. At timestep t, just before the
// all-reduce, each replica is computing:
//
// (**)
// v_t = v_(t-1) * mm + wd * vs * w_(t-1) +
//                      sum_over_microbatches((1 - dm) * vs / (ls * af) * g_i)
//
// Where g_i is the gradient of the i^th microbatch the replica has computed.
//
// The desired v after reduction of gradients should be mathematically
// equivalent to this equation, but with the gradients summed over replicas. So
// the goal is to compute:
//
// v_t = v_(t-1) * mm + wd * vs * w_(t-1) +
//       sum_over_replicas(
//         sum_over_microbatches((1 - dm) * vs / (ls * af) * g_r_i)
//       )
//
// Where g_r_i is the gradient of the i^th microbatch computed on replica r.
//
// Let us see what happens if we naively try to all-reduce the v computed in
// (**):
//
// v_t = sum_over_replicas(
//         v_(t-1) * mm + wd * vs * w_(t-1) +
//             sum_over_microbatches((1 - dm) * vs / (ls * af) * g_r_i)
//       )
//
//   Split into two sums:
//     = sum_over_replicas(v_(t-1) * mm + wd * vs * w_(t-1)) +
//       sum_over_replicas(
//           sum_over_microbatches((1 - dm) * vs / (ls * af) * g_r_i)
//       )
//
//   Simplify first sum into multiplication as each iteration independent:
//     = rf * (v_(t-1) * mm + wd * vs * w_(t-1)) +
//       sum_over_replicas(
//         sum_over_microbatches((1 - dm) * vs / (ls * af) * g_r_i)
//       )
//
// We see that the first term is incorrectly scaled by rf. To compensate for
// this, we can also scale the second term by rf, then unscale during both the
// weight update and following velocity update:
//
// v_t = rf * (v_(t-1) * mm + wd * vs * w_(t-1)) +
//       rf * sum_over_replicas(
//         sum_over_microbatches((1 - dm) * vs / (ls * af) * g_r_i)
//       )
//
// Then later in the w update and second v update, divide by rf, so we have:
//
// v_t = v_(t-1) * mm + wd * vs * w_(t-1) +
//       sum_over_replicas(
//         sum_over_microbatches((1 - dm) * vs / (ls * af) * g_r_i)
//       )
//
// Which is the same as (**).
//
// Thus the code has become, with the all-reduce on v and the extra rf scaling:
//
// for each micro batch
//   v += (1 - dm) * vs * rf / (ls * af) * g
// allReduce(v)
// w = w - lr / (vs * rf) * v
// v = v * mm / rf + (1 - dm) * wd * vs * w
//
// This type of reduction (which as usual is inferred by the SGD class) is known
// as OptimizerReductionType::AcclReduce.
//
// ----------------------------------------------
// SGDAccumulatorAndMomentum::Combined Caveats
//
// 1. Lagged upate of optimizer parameters
// During their training loop, the user can dynamically update the optimizer
// parameters using Session::updateOptimizerFromHost. However, when using
// SGDAccumulatorAndMomentum::Combined, part of the v update for the next step
// has already occured using the old parameters. Thus, updating parameters
// during training is well-defined but not mathematically "correct". We say the
// update is "lagged" as the update will not come fully into effect until after
// one update.
//
// 2. Extra numerical instability due to rf scaling
// When doing replication, the velocity is now additionally scaled by rf. This
// exacerbates any potential numerical issues during training, and gets worse
// with higher replication factors.
//
// You can try to use velocity scaling to counteract this, but that is yet
// another layer of hyperparameter-tweaking complication for the user. With
// SGDAccumulatorAndMomentum::Separate, the loss scaling is undone after the
// all-reduce, before updating v, so velocity scaling is not really even
// required.
//
// 3. Un-portable hyperparameters
//
// Due to the above points, the hyperparameters you have found to work on the
// exact same model with SGDAccumulatorAndMomentum::Separate, or in a different
// framework, may not work with SGDAccumulatorAndMomentum::Combined.
//
// 4. Cannot separately offload velocity and accumulator
//
// With SGDAccumulatorAndMomentum::Separate, you have the freedom to
// individually offload the velocity and accumulator tensors when experimenting
// with different execution schemes. With SGDAccumulatorAndMomentum::Combined,
// you cannot do this as they are one tensor.
//
// 5. Harder to understand
//
// Evidently, SGDAccumulatorAndMomentum::Combined is much harder to understand
// and has more subtleties to consider than
// SGDAccumulatorAndMomentum::Separate. Thus, it is recommended to default to
// using SGDAccumulatorAndMomentum::Separate until you know you need the
// SGDAccumulatorAndMomentum::Combined memory optimisation.
//
//
// The compound scalars of SGD
// ---------------------------
// (\sa Optimizer documentation for what compound scalars are)
//
// ----------
//
// In the simplest case of SGD, there is no gradient accumulation or momentum
// (mm = 0), and thus no extra optimiser tensors. The optimiser step reduces to:
//
//   w <- w * (1 -  lr * (1 - dm) * wd) -  g * (lr * (1 - dm) / ls)
//
// and thus the compound scalars are:
//
//   - weightDecayScaleFactor0 (wdsf0) =
//       1 - lr * (1 - dm) * wd
//
//   - scaledLearningRate0 (slr0) =
//       lr *  ( 1 - dm) / ls
//
// Internally, we call this case SGD0, as there are 0 extra optimiser tensors.
//
// ----------
//
// For the case where there is gradient accumulation or momentum, and
// SGDAccumulatorAndMomentum::Combined, recall the optimiser step is:
//
// for each micro batch
//   v += (1 - dm) * vs * rf / (ls * af) * g
// allReduce(v)
// w = w - lr / (vs * rf) * v
// v = v * mm / rf + (1 - dm) * wd * vs * w
//
// and thus the compound scalars are:
//                                             mm dm wd lr ls vs rf af
//                                             =======================
//   - scaledWeightDecay1 (swd1) =             .  x  x  .  .  x  .  .
//       (1 - dm) * wd * vs
//
//   - dampeningScaleFactor1 (dpsf1) =         .  x  .  .  x  x  x  x
//       (1 - dm) * vs * rf / (ls * af)
//
//   - scaledLearningRate1 (slr1) =            .  .  .  x  .  x  x  .
//       lr / ( vs * rf)
//
//   - scaledMomentum1 (smm1) =                x  .  .  .  .  .  x  .
//       mm / rf
//
// Recall that af is the accumulation factor only if using gradient accumulation
// and the SessionOptions::accumulationAndReplicationReductionType was
// ReductionType::Mean; otherwise af is 1.
//
// Internally we call this case SGD1 as there is 1 extra optimiser tensor.
//
// ----------
//
// For the case where there is gradient accumulation or momentum, and
// SGDAccumulatorAndMomentum::Separate, recall the optimiser step is:
//
// a = 0
// for each micro-batch
//   a += g
// allReduce(a)
// v = v * mm + (1 - dm) * vs / (ls * af) * a + (1 - dm) * wd * vs * w
// w = w - lr / vs * v
//
// Thus the above compound scalars for SGDAccumulatorAndMomentum::Combined can
// actually be reused, but with rf = 1 always, as no rf scaling is required.
//
// Internally, we call this case SGD2, as there are 2 extra optimiser tensors.
//
// ----------
//
// Across all cases, the possible atomic scalars are:
//   mm, dm, wd, lr, ls, vs, rf, af.
//
// All atomic scalar terms except ls, rf, and af can be Tensor specific.
//
// The VarUpdateOps of SGD
// ---------------------------
// Recall from \sa Optimizer::createOp that the Optimizer implementation will
// create the VarUpdateOp used for updating each weight.
//
// For SGD0, this is an SGD0VarUpdateOp.
//
// For SGD1, this is an SGD1ComboOp. This will later be decomposed into a series
// of ops by the SGD1Decompose pattern. \sa SGD1Decompose.
//
// For SGD2, this is an SGD2ComboOp. This will later be decomposed into a series
// of ops by the SGD2Decompose pattern. \sa SGD2Decompose.

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
 *
 * See the SGD notes in optimizer.hpp for a more detailed and comprehensive
 * derivation of the SGD optimizer step in PopART.
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
  /// \param sgdAccMm The implementation strategy to use when gradient
  ///     accumulation and/or momentum are used, otherwise ignored. \sa
  ///     SGDAccumulatorAndMomentum. Defaults to
  ///     SGDAccumulatorAndMomentum::Combined.
  /// \param accumType The DataType of the accum tensor, when gradient
  ///     accumulation is used and sgdAccMm =
  ///     SGDAccumulatorAndMomentum::Separate, otherwise ignored. Only FLOAT,
  ///     FLOAT16 and UNDEFINED are supported. Defaults to UNDEFINED. If
  ///     UNDEFINED, the same type as the weights will be used. If accumType is
  ///     FLOAT16 and accl1Type is FLOAT, this parameter causes accum to be
  ///     upcasted before being passed to the op that updates accl1.
  /// \param accl1Type The DataType of the accl1 tensor, when gradient
  ///     accumulation is used and sgdAccMm =
  ///     SGDAccumulatorAndMomentum::Separate, otherwise ignored. Only FLOAT,
  ///     FLOAT16 and UNDEFINED are supported. Defaults to UNDEFINED. If
  ///     UNDEFINED, the same type as the weights will be used. If accumType is
  ///     FLOAT16 and accl1Type is FLOAT, this parameter causes accum to be
  ///     upcasted before being passed to the op that updates accl1.
  SGD(OptimizerValue defaultLearningRate,
      OptimizerValue defaultWeightDecay,
      OptimizerValue defaultMomentum,
      OptimizerValue defaultDampening,
      OptimizerValue defaultVelocityScaling,
      OptimizerValue lossScaling,
      const std::vector<ClipNormSettings> &clipNormSettings = {},
      SGDAccumulatorAndMomentum sgdAccMm = SGDAccumulatorAndMomentum::Combined,
      DataType accumType                 = DataType::UNDEFINED,
      DataType accl1Type                 = DataType::UNDEFINED);

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
  /// \param sgdAccMm The implementation strategy to use when gradient
  ///     accumulation and/or momentum are used, otherwise ignored. \sa
  ///     SGDAccumulatorAndMomentum. Defaults to
  ///     SGDAccumulatorAndMomentum::Combined.
  /// \param accumType The DataType of the accum tensor, when gradient
  ///     accumulation is used and sgdAccMm =
  ///     SGDAccumulatorAndMomentum::Separate, otherwise ignored. Only FLOAT,
  ///     FLOAT16 and UNDEFINED are supported. Defaults to UNDEFINED. If
  ///     UNDEFINED, the same type as the weights will be used. If accumType is
  ///     FLOAT16 and accl1Type is FLOAT, this parameter causes accum to be
  ///     upcasted before being passed to the op that updates accl1.
  /// \param accl1Type The DataType of the accl1 tensor, when gradient
  ///     accumulation is used and sgdAccMm =
  ///     SGDAccumulatorAndMomentum::Separate, otherwise ignored. Only FLOAT,
  ///     FLOAT16 and UNDEFINED are supported. Defaults to UNDEFINED. If
  ///     UNDEFINED, the same type as the weights will be used. If accumType is
  ///     FLOAT16 and accl1Type is FLOAT, this parameter causes accum to be
  ///     upcasted before being passed to the op that updates accl1.
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
      const std::vector<ClipNormSettings> &clipNormSettings = {},
      SGDAccumulatorAndMomentum sgdAccMm = SGDAccumulatorAndMomentum::Combined,
      DataType accumType                 = DataType::UNDEFINED,
      DataType accl1Type                 = DataType::UNDEFINED);
  static SGD fromDefaultMap(const std::map<std::string, OptimizerValue> &);

  /// Default constructor
  /// Creates SGD with default scalars (equivalent to getUnset<scalar>()
  /// methods), and other default parameters of main constructor.
  SGD() : SGD(std::map<std::string, std::pair<float, bool>>{}) {}

  /// Construct an SGD instance with default values.
  SGD(const SGD &) = default;
  ~SGD()           = default;

  OptimizerType type() const final { return OptimizerType::SGD; }
  std::string type_s() const final { return "SGD"; }

  SGDAccumulatorAndMomentum getSGDAccumulatorAndMomentum() const {
    return sgdAccMm;
  }

  std::unique_ptr<Optimizer> clone() const final;

  /// Returns the VarUpdateOp for the given \p weight . If no gradient
  /// accumulation of momentum, this will be a SGD0VarUpdateOp. Else, if
  /// `getSGDAccumulatorAndMomentum() == ::Combined`, this will be an
  /// SGD1ComboOp, else if `getSGDAccumulatorAndMomentum() ==
  /// ::Combined`SGD2ComboOp, an SGD2ComboOp.
  /// \sa Optimizer::createOp
  ///
  /// The required compound scalar OptimizerValues for the VarUpdateOp wil be
  /// computed and passed to the Op. See the SGD notes above this class for how
  /// they are derived. Recall that if non-const, the VarUpdateOp will take an
  /// input Tensor for the compound scalar.
  ///
  /// The OptimizerReductionType of the Op is derived as follows:
  ///   No replication              => None
  ///   Replication, no grad acc    => GradReduce
  ///   Replication, grad acc, SGD1 => AcclReduce
  ///   Replication, grad acc, SGD2 => AccumReduce
  /// See the SGD notes above this class for why this is.
  ///
  /// If SGD2, the DataType of the accum and accl1 tensors passed to the
  /// SGD2ComboOp will be as set in the SGD constructor. Recall
  /// DataType::UNDEFINED means use the same as the weight.
  ///
  /// An SGD1ComboOp will later be decomposed by SGD1Decompose pattern into a
  /// series of Ops and Tensors that implement the SGD1 optimiser step.
  /// \sa SGD1Decompose
  ///
  /// An SGD12ComboOp will later be decomposed by SGD2Decompose pattern into a
  /// series of Ops and Tensors that implement the SGD2 optimiser step.
  /// \sa SGD2Decompose
  std::unique_ptr<Op> createOp(const Tensor &weight, Graph &) const final;

  /// \sa Optimizer::getInputIds
  std::vector<TensorId> getInputIds(const Tensor &weight) const final;

  /// smm1 and wdsf0 have the same data type as the \p weight . Everything else
  // is float32. All shapes will be {}, so scalar.
  // \sa Optimizer::getOptimizerInputs.
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
  ///     `"learningRate"`, `"weightDecay"`, `"momentum"`, `"dampening"`, or
  ///     `"velocityScaling"` and the map's values pairs of floats and booleans
  ///     representing OptimizerValue constructor arguments. The map does not
  ///     have to specify each hyper parameter as default values will be used
  ///     where parameters are missing.
  void
  insertSpecific(const TensorId &weight,
                 const std::map<std::string, std::pair<float, bool>> &params);

  // Does "w" have specific OptimizerValues, or will it use default?
  bool hasSpecific(const Tensor &w) const final;

  // Do any weights have specific OptimizerValues, or do they all use default?
  bool hasSpecific() const final;

  TensorId getInverseLossScalingTensorId(const Tensor &weight) const;

  const OptimizerValueMap &learningRates() const { return lrs; }
  const OptimizerValueMap &weightDecays() const { return wds; }
  const OptimizerValueMap &momentums() const { return mms; }
  const OptimizerValueMap &dampenings() const { return dps; }
  const OptimizerValueMap &velocityScalings() const { return vss; }

  virtual size_t hash() const;

private:
  /// If velocity (accumulation) is required, either because of gradient
  /// accumulation or because of momentum, then return true otherwise return
  /// false.
  bool requiresAccl(const Tensor &weight) const;

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

  // SGD implementation strategy when accl/accum tensors needed (SGD1 or SGD2)
  SGDAccumulatorAndMomentum sgdAccMm;

  // SGD2 only: DataType of accum and accl1 tensors can be specified.
  DataType sgd2AccumType;
  DataType sgd2Accl1Type;

  // int argument only to disambiguate from the other SGD constructor
  SGD(const std::map<std::string, OptimizerValue> &,
      const std::vector<ClipNormSettings> &,
      SGDAccumulatorAndMomentum sgdAccMm,
      DataType accumType,
      DataType accl1Type,
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
