// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_OPTIMIZER_HPP
#define GUARD_NEURALNET_OPTIMIZER_HPP

#include <memory>
#include <popart/clipnormsettings.hpp>
#include <popart/compoundscalarhelper.hpp>
#include <popart/error.hpp>
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

std::ostream &operator<<(std::ostream &os, const OptimizerReductionType &ort);

/***
 * Enum type for different types of weight decay.
 */
enum class WeightDecayMode {
  /// Weight decay (e.g. AdamW)
  Decay,
  /// L2 regularization (e.g. PyTorch-like Adam)
  L2Regularization
};

std::ostream &operator<<(std::ostream &os, const WeightDecayMode &wdm);

std::map<std::string, OptimizerValue>
getOptMap(const std::map<std::string, std::pair<float, bool>> &m);

template <typename... Args>
runtime_error optimizer_replacement_error(const std::string &s,
                                          const Args &...args) {
  return runtime_error("New optimizer is not a valid replacement. " + s,
                       args...);
}

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

  // The loss scaling value supplied by the user.
  float getLossScalingVal() const { return ls.val(); }

  // The compound scalar value of the loss scaling tensor added
  // to the graph. The user-defined loss scaling may be scaled
  // by the inverse of the graph replication factor, depending
  // on SessionOptions.
  float getFinalLossScalingVal() const;

  static TensorId getLossScalingTensorId(DataType);
  virtual TensorId
  // Either the scalar tensor representing the inverse loss scale factor, or
  // compound scalar tensor which contains the inverse loss scale factor
  getInverseLossScalingTensorId(const Tensor &weight) const = 0;

  virtual void setFactorsFromOptions(const SessionOptions &);

  bool gradientAccumulationEnabled() const;
  bool meanReductionEnabled() const;
  bool postMeanAccumulationEnabled() const;
  bool postMeanReplicationEnabled() const;
  bool lossMeanReplicationEnabled() const;
  int64_t getReplicatedGraphCount() const;
  int64_t getAccumulationFactor() const;

  // Deprecated
  bool meanGradientAccumulationEnabled() const;

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
  bool meanReduction;
  bool postMeanAccumulation;
  bool postMeanReplication;
  bool lossMeanReplication;
  int64_t accumulationFactor;
  int64_t replicatedGraphCount;

  bool factorsAreSetFromOptions{false};
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
