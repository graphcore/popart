// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_SGD_HPP_
#define POPART_WILLOW_INCLUDE_POPART_SGD_HPP_

#include <cstddef>
#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <popart/optimizer.hpp>
#include <popart/optimizervalue.hpp>

#include "popart/clipnormsettings.hpp"
#include "popart/compoundscalarhelper.hpp"
#include "popart/datatype.hpp"
#include "popart/debugcontext.hpp"
#include "popart/optimizervaluemap.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {
class Graph;
class Op;
class Tensor;
class TensorInfo;

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
// Based on the PyTorch implementation
// https://PyTorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD , the SGD
// update equation with weight decay, dampening, and momentum, is
//
// g = gradient computed in backwards pass
// g = g + wd * w
// v = v * mm + (1 - dm) * g
// if enable nesterov momentum:
//   g = g + mm * v
//   w = w - lr * g
// else:
//   w = w - lr * v
//
// which is equivalent to
//
// g = gradient computed in backwards pass
// v = v * mm + (1 - dm) * g + (1 - dm) * wd * w
// if enable nesterov momentum:
//   g = g + wd * w + mm * v
//   w = w - lr * g
// else:
//   w = w - lr * v
//
// Loss Scaling
// ------------
// Loss scaling means the loss gradient is scaled by ls before starting the
// backwards pass, then each individual weight gradient is un-scaled during the
// weight update. Thus, the weight update becomes:
//
// g = gradient computed in backwards pass (scaled by ls)
// v = v * mm + (1 - dm) / ls * g + (1 - dm) * wd * w
// if enable nesterov momentum:
//   g = 1 / ls * g + wd * w + mm * v
//   w = w - lr * g
// else:
//   w = w - lr * v
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
// if enable nesterov momentum:
//   g = vs * (1 / ls * g + wd * w) + mm * v
//   w = w - lr / vs * g
// else:
//   w = w - lr / vs * v
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
// if enable nesterov momentum:
//   a = vs * (1 / ls * a + wd * w) + mm * v
//   w = w - lr / vs * a
// else:
//   w = w - lr / vs * v
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
// if enable nesterov momentum:
//   a = vs * (1 / ls * a + wd * w) + mm * v
//   w = w - lr / vs * a
// else:
//   w = w - lr / vs * v
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
// if enable nesterov momentum:
//   a = vs * (1 / ls * a + wd * w) + mm * v
//   w = w - lr / vs * a
// else:
//   w = w - lr / vs * v
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
// if enable nesterov momentum:
//   g = vs * (1 / ls * g + wd * w) + mm * v
//   w = w - lr / vs * g
// else:
//   w = w - lr / vs * v
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
// if enable nesterov momentum:
//   g = vs * (1 / ls * g + wd * w) + mm * v
//   w = w - lr / vs * g
// else:
//   w = w - lr / vs * v
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
// v = (1 - dm) * vs / (ls * af) * a
// if enable nesterov momentum:
//   a = vs * (1 / ls * a + wd * w) + mm * v
//   w = w - lr / vs * a
// else:
//   w = w - lr / vs * v
// v = v * mm + (1 - dm) * wd * vs * w
//
// We can elide the accumulator tensor by directly accumulating into v:
//
// for each micro batch
//   v += (1 - dm) * vs / (ls * af) * g
//   if enable nesterov momentum:
//     a += g
// if enable nesterov momentum:
//   a = vs * (1 / ls * a + wd * w) + mm * v
//   w = w - lr / vs * a
// else:
//   w = w - lr / vs * v
// v = v * mm + (1 - dm) * wd * vs * w
//
// ----------------------------------------------
// Replication
//
// If we try to add back replication:
//
// for each micro batch
//   v += (1 - dm) * vs / ls * g
//   if enable nesterov momentum:
//     a += g
// # There is no accumulator to reduce
// allReduce(??)
// if enable nesterov momentum:
//   a = vs * (1 / ls * a + wd * w) + mm * v
//   w = w - lr / vs * a
// else:
//   w = w - lr / vs * v
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
//   if enable nesterov momentum:
//     a += g
// allReduce(v)
// if enable nesterov momentum:
//   a = (vs * rf) * (1 / ls * a + wd * w) + mm * v
//   w = w - lr / (vs * rf) * a
// else:
//   w = w - lr / (vs * rf) * v
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
//   if enable nesterov momentum:
//     a += g
// allReduce(v)
// if enable nesterov momentum:
//   a = (vs * rf) * (1 / ls * a + wd * w) + mm * v
//   w = w - lr / (vs * rf) * a
// else:
//   w = w - lr / (vs * rf) * v
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
//       lr / (vs * rf)
//
//   - scaledMomentum1 (smm1) =                x  .  .  .  .  .  x  .
//       mm / rf
//
//   - nesterovDampeningScaleFactor1 (ndsf1) = .  x  .  .  .  x  x  x
//       af / ((1 - dm) * vs * rf)
//       ndsf1 * dpsf1 = 1 / ls
//
//   - nesterovGradScalFactor1 (ngsf1) =       .  .  .  .  .  x  x  .
//       vs * rf
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
// v = v * mm + (1 - dm) * vs / (ls * af * rf) * a + (1 - dm) * wd * vs * w
// if enable nesterov momentum:
//   a = vs * (1 / ls * a + wd * w) + mm * v
//   w = w - lr / vs * a
// else:
//   w = w - lr / vs * v
//
// Thus the above compound scalars for SGDAccumulatorAndMomentum::Combined can
// actually be reused, but with rf = 1 always, as no rf scaling is required.
//
// The compound scalars are:
//                                             mm dm wd lr ls vs rf af
//                                             =======================
//   - scaledWeightDecay1 (swd1) =             .  x  x  .  .  x  .  .
//       (1 - dm) * wd * vs
//
//   - dampeningScaleFactor2 (dpsf2) =         .  x  .  .  x  x  .  x
//       (1 - dm) * vs / (ls * af * rf)
//
//   - scaledLearningRate2 (slr2) =            .  .  .  x  .  x  .  .
//       lr / vs
//
//   - scaledMomentum2 (smm2) =                x  .  .  .  .  .  .  .
//       mm
//
//   - nesterovDampeningScaleFactor2 (ndsf2) = .  x  .  .  .  x  x  x
//       af * rf / ((1 - dm) * vs)
//       ndsf2 * dpsf2 = 1 / ls
//
//   - nesterovGradScalFactor2 (ngsf2) =       .  .  .  .  .  x  .  .
//       vs
//
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
// For SGD0, this is an SGD0ComboOp. This will later be decomposed into a series
// of ops by the SGD0Decompose pattern. \sa SGD0Decompose.
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
 *  * *nesterov*
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
 * if nesterov is True:
 * \f[
 *    g' := g + \text{wd} * w + \text{mm} * v' \text{ \ . }
 * \f]
 * \f[
 *    w' := w - \text{lr} * g' \text{ \ . }
 * \f]
 * else:
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

  /// Default nesterov.
  static OptimizerValue getUnsetNesterov() {
    return {0.0f, true}; // no nesterov, ever
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
  /// \param nesterov Option to enable Nesterov momentum. Defaults to false.
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
  /// \param debugContext Optional debug context.
  SGD(OptimizerValue defaultLearningRate,
      OptimizerValue defaultWeightDecay,
      OptimizerValue defaultMomentum,
      OptimizerValue defaultDampening,
      OptimizerValue defaultVelocityScaling,
      OptimizerValue lossScaling,
      OptimizerValue nesterov,
      const std::vector<ClipNormSettings> &clipNormSettings = {},
      SGDAccumulatorAndMomentum sgdAccMm = SGDAccumulatorAndMomentum::Combined,
      DataType accumType                 = DataType::UNDEFINED,
      DataType accl1Type                 = DataType::UNDEFINED,
      const DebugContext &debugContext   = {});

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
  /// \param debugContext Optional debug context.
  SGD(OptimizerValue defaultLearningRate,
      OptimizerValue defaultWeightDecay,
      OptimizerValue defaultMomentum,
      OptimizerValue defaultDampening,
      OptimizerValue defaultVelocityScaling,
      OptimizerValue lossScaling,
      const std::vector<ClipNormSettings> &clipNormSettings = {},
      SGDAccumulatorAndMomentum sgdAccMm = SGDAccumulatorAndMomentum::Combined,
      DataType accumType                 = DataType::UNDEFINED,
      DataType accl1Type                 = DataType::UNDEFINED,
      const DebugContext &debugContext   = {});

  /// Constructor.
  /// \param params A parameter map where the keys are one or more of
  ///     `"defaultLearningRate"`, `"defaultWeightDecay"`, `"defaultMomentum"`,
  ///     `"defaultDampening"`, `"defaultVelocityScaling"`, `"lossScaling"`
  ///     or `"nesterov".
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
  /// \param debugContext Optional debug context.
  ///
  /// **EXAMPLE**:
  /// ```{.cpp}
  /// SGD({{"defaultLearningRate", {0.02, false}},
  ///     {"defaultMomentum", {0.6, true}}});
  /// ```
  /// This will create an SGD Optimizer which has a constant momentum of 0.6 and
  /// a changeable learning rate initially of 0.02. All OptimizerValues not
  /// present in the map will take values from the `getUnset`* functions.
  SGD(const std::map<std::string, std::pair<float, bool>> &params,
      const std::vector<ClipNormSettings> &clipNormSettings = {},
      SGDAccumulatorAndMomentum sgdAccMm = SGDAccumulatorAndMomentum::Combined,
      DataType accumType                 = DataType::UNDEFINED,
      DataType accl1Type                 = DataType::UNDEFINED,
      const DebugContext &debugContext   = {});
  static SGD fromDefaultMap(const std::map<std::string, OptimizerValue> &,
                            const DebugContext &debugContext = {});

  /// Default constructor
  /// Creates SGD with default scalars (equivalent to getUnset<scalar>()
  /// methods), and other default parameters of main constructor.
  SGD() : SGD(std::map<std::string, std::pair<float, bool>>{}) {}

  /// Copy constructor
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
  /// \param nesterov Option to enable Nesterov momentum. Defaults to false.
  void insertSpecific(const TensorId &weight,
                      OptimizerValue learningRate,
                      OptimizerValue weightDecay,
                      OptimizerValue momentum,
                      OptimizerValue dampening,
                      OptimizerValue velocityScaling,
                      OptimizerValue nesterov);

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
  const OptimizerValueMap &nesterov() const { return nts; }

  virtual size_t hash() const;

private:
  bool hasMomentum(const Tensor &weight) const;

  bool enableNesterov(const Tensor &weight) const;

  void runValueChecks(OptimizerValue lr,
                      OptimizerValue wd,
                      OptimizerValue mm,
                      OptimizerValue dp,
                      OptimizerValue vs,
                      OptimizerValue nt) const;

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

  // Enables Nesterov momentum.
  OptimizerValueMap nts;

  // The compound scalars
  // --------------------
  // No Momentum needed (SGD0)
  ScaledLearningRate0Helper slr0helper;
  WeightDecayScaleFactor0Helper wdsf0helper;

  // SGDAccumulatorAndMomentum::Combined (SGD1)
  ScaledLearningRate1Helper slr1helper;
  ScaledWeightDecay1Helper swd1helper;
  DampeningScaleFactor1Helper dpsf1helper;
  ScaledMomentum1Helper smm1helper;

  // SGDAccumulatorAndMomentum::Seperate (SGD2)
  ScaledLearningRate2Helper slr2helper;
  DampeningScaleFactor2Helper dpsf2helper;
  ScaledMomentum2Helper smm2helper;

  // SGD implementation strategy when accl/accum tensors needed (SGD1 or SGD2)
  SGDAccumulatorAndMomentum sgdAccMm;

  // SGD0 and SGD2 only: DataType of accum can be specified.
  DataType sgdAccumType;
  // SGD2 only: accl1 tensors can be specified.
  DataType sgd2Accl1Type;

  // Nesterov momentum only
  SGDMomentumHelper mmHelper;
  SGDWeightDecayHelper wdHelper;
  NesterovGradScaleFactor1Helper ngsf1helper;
  NesterovGradScaleFactor2Helper ngsf2helper;
  NesterovDampeningScaleFactor1Helper ndsf1helper;
  NesterovDampeningScaleFactor2Helper ndsf2helper;

  // int argument only to disambiguate from the other SGD constructor
  SGD(const std::map<std::string, OptimizerValue> &,
      const std::vector<ClipNormSettings> &,
      SGDAccumulatorAndMomentum sgdAccMm,
      DataType accumType,
      DataType accl1Type,
      int,
      const DebugContext &debugContext = {});

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
            getUnsetNesterov(),
            clipNormSettings,
            getSGDAccumulatorAndMomentum(),
            DataType::UNDEFINED,
            DataType::UNDEFINED,
            {}) {}
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_SGD_HPP_
