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

enum class AdamMode { Adam = 0, AdamNoBias, Lamb, LambNoBias };

// Implements Adam(W)
//   (Adam: A Method For Stochastic Optimization, Kingma & Ba, ICLR 2015)
//   (Fixing Weight Decay Regularization in Adam, Loshchilov & Hutter,
//    ICLR 2018 reject)
// and Lamb (with and without bias correction)
//   (Large Batch Optimization For Deep learning: Training BERT in 76 Minutes,
//    You et al., ICLR 2020)
//
// Note that for implementation reasons, weight decay with this Adam
// implementation always defaults to AdamW.
// Weight decay in the sense of L2-regularization (original Adam paper)
// is not supported.
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
// If accumulationReductionType is set to ReductionType::Sum 'af' is set to 1
//
// accumulator (only used if gradient accumulation is enabled):
// s = s + g                             (otherwise: s = g)
//
// Current gradient scaling
// gs = ls * af
//
// first order momentum (FP16/FP32):
// m = b1 * m + (1 - b1) * s
//
// bias corrected:
// mc = m / ((1 - b1 ** t) * gs)  (without correction: mc = m / gs)
//
// second order momentum (FP16/FP32):
// v = b2 * v + (1 - b2) * s
//
// bias corrected:
// vc = v / ((1 - b2 ** t) * gs ** 2)  (without correction: vc = v / gs ** 2)
//
// updater term (FP16/FP32):
// x = mc / (sqrt(vc) + eps) + wd * w
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
// w -= min(r1,mwn)/r2 * lr * x
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
//                       |  | [lr]  /            /            sharding only)
//                       |  |  |   / .-----------
//                     (AdamVarUpdate)
//                       |
//                      [w']

class Adam : public Optimizer {

public:
  static OptimizerValue getUnsetBeta1() {
    return {0.9f, true}; // fixed beta1 of 0.9
  }

  static OptimizerValue getUnsetBeta2() {
    return {0.999f, true}; // fixed beta2 of 0.999
  }

  static OptimizerValue getUnsetEps() {
    return {1e-6f, true}; // a denominator stability term of 1e-8 forever
  }

  static OptimizerValue getUnsetWeightDecay() {
    return {0.0f, true}; // no weight decay, ever
  }

  static OptimizerValue getUnsetLossScaling() {
    return {1.0f, true}; // no loss scaling, ever
  }

  static OptimizerValue getUnsetLearningRate() {
    return {0.1f, true}; // a learning rate of 0.1 forever
  }

  static OptimizerValue getUnsetMaxWeightNorm() {
    return {10.0f, true}; // a maximum weight norm of 10.0f forever
  }

public:
  // Does "w" have specific OptimizerValues, or will it use default?
  bool hasSpecific(const Tensor &w) const;

  // Adam constructor with all parameteers
  // ----------------
  Adam(OptimizerValue default_lr,
       OptimizerValue default_wd,
       OptimizerValue default_b1,
       OptimizerValue default_b2,
       OptimizerValue default_eps,
       OptimizerValue ls,
       OptimizerValue mwn,
       AdamMode adamMode_,
       DataType accumType_,
       DataType accl1Type_,
       DataType accl2Type_);

  // Adam constructor with all parameters except max weight norm.
  // ----------------
  Adam(OptimizerValue default_lr,
       OptimizerValue default_wd,
       OptimizerValue default_b1,
       OptimizerValue default_b2,
       OptimizerValue default_eps,
       OptimizerValue ls,
       AdamMode adamMode_,
       DataType accumType_,
       DataType accl1Type_,
       DataType accl2Type_);

  // Example:
  //
  // Adam({{"defaultLearningRate", {0.02, False}},
  //       {"defaultBeta1", {0.9, True}},
  //       {"defaultBeta2":{0.999, True}}});
  //
  // will create an Adam Optimizer which has a constant beta1/beta2 of 0.9/0.99
  // and a changeable learning rate initially of 0.02.
  // All OptimizerValues not present in the map will take values from the
  // getUnset* functions.
  //
  // Construct from pair instead of OptimizerValue for pybind11 support
  //
  Adam(const std::map<std::string, std::pair<float, bool>> &,
       AdamMode adamMode_,
       DataType accumType_,
       DataType accl1Type_,
       DataType accl2Type_);
  static Adam fromDefaultMap(const std::map<std::string, OptimizerValue> &,
                             AdamMode adamMode_,
                             DataType accumType_,
                             DataType accl1Type_,
                             DataType accl2Type_);

  Adam(const Adam &) = default;
  ~Adam()            = default;

  OptimizerType type() const final { return OptimizerType::Adam; }
  std::string type_s() const final { return "Adam"; }

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
                      OptimizerValue b1,
                      OptimizerValue b2,
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
  const OptimizerValueMap &beta1s() const { return b1s; }
  const OptimizerValueMap &beta2s() const { return b2s; }
  const OptimizerValueMap &epss() const { return epsvs; }
  const OptimizerValueMap &maxWeightNorm() const { return mwns; }

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
  DataType accumType;
  DataType accl1Type;
  DataType accl2Type;

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

  // int argument only to disambiguate from the other SGD constructor
  Adam(const std::map<std::string, OptimizerValue> &,
       AdamMode mode_,
       DataType accumType_,
       DataType accl1Type_,
       DataType accl2Type_,
       int);

  static std::map<std::string, OptimizerValue>
  getComplete(const std::map<std::string, OptimizerValue> &);
};

} // namespace popart

#endif
