#ifndef GUARD_NEURALNET_OPTIMIZER_HPP
#define GUARD_NEURALNET_OPTIMIZER_HPP

#include <memory>
#include <popart/names.hpp>
#include <popart/optimizervalue.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

enum class OptimizerType { SGD = 0, NTYPES };

class OptimizerValueMap {
public:
  OptimizerValueMap(OptimizerValue g) : global(g) {}

  // Return the OptimizerValue specific to "id" if there is one, otherwise
  // return the global OptimizerValue
  OptimizerValue get(const TensorId &id) const;

  // Register a specific OptimizerValue for a Tensor
  void insertSpecific(const TensorId &, OptimizerValue);

  bool hasSpecific(const TensorId &id) const {
    return specifics.find(id) != specifics.end();
  }

  OptimizerValue getGlobal() const { return global; }

  // Check for compatibility of OptimizerValueMaps - can one replace another
  // after Graph construction without requiring changes to the compuatation
  // Graph?
  bool validReplacement(const OptimizerValueMap &rhs) const;

private:
  std::map<TensorId, OptimizerValue> specifics;

  // The fall-back for all Tensors without a specific OptimizerValue
  OptimizerValue global;
};

// The base Optimizer class
class Optimizer {
public:
  virtual ~Optimizer() = default;
  Optimizer(OptimizerValue lossScaling);
  Optimizer(const Optimizer &) = default;

  // If a Graph has been construted with this Optimizer, can it be updated with
  // "other", without requiring a change to compute Graph? For example, a
  // VarUpdate which has a constant scaled learning rate cannot be modified to
  // have a variable scaled learning rate
  virtual bool validReplacement(const Optimizer &other) const = 0;

  virtual OptimizerType type() const               = 0;
  virtual std::string type_s() const               = 0;
  virtual std::unique_ptr<Optimizer> clone() const = 0;

  // (re)set the data in Tensor from a relevant value stored by this Optimizer.
  // The particular value used is determined from the Tensor's name/type
  virtual void resetTensorData(Tensor &) const = 0;
  virtual void setTensorData(Tensor &) const   = 0;

  // Create a VarUpdate Op for a specific weight Tensor using this Optimizer,
  // and get the names of inputs to the VarUpdate Op fo a specific Tensor
  virtual std::unique_ptr<Op> createOp(const Tensor &weight, Graph &) const = 0;
  virtual std::vector<TensorId> getInputIds(const Tensor &weight) const     = 0;
  virtual std::vector<std::tuple<TensorId, TensorInfo>>
  getOptimizerInputs(const Tensor &weight) const = 0;

  const OptimizerValue &lossScaling() const { return ls; }
  TensorId getLossScalingTensorId(DataType) const;
  float getLossScalingVal() const { return ls.val(); }

private:
  OptimizerValue ls;
};

// matching the pytorch implementation for SGD,
//
// w <- w * (1 - lr * wd) - (lr/ls) * dw                      (1)
//          ^^^^^^^^^^^^^^^^^^   ~~~~~~~
//                  |               |
//                  |      scaled learning rate
//                  |
//                  |
//       weight decay scale factor
//
// where:
// lr : learning rate
// wd : weight decay
// ls : loss scaling
//
// Note that the 2 compound terms above, (1 - lr * wd )  and (lr/ls),
// are always calculated on host
//
// Note too that (1) lr and wd can be Tensor specific, (2) ls is only evert
// global.
//
// Constructing an SGD Optimizer is done in 2 steps;
// (1) Construct SGD with global values for lr, wd, ls.
// (2) Set Tensor specific values.
//
// Any OptimizerValue can be set as constant if it will not change during
// training. This can result in faster/smaller code.

class SGD : public Optimizer {
public:
  SGD(OptimizerValue globalLearningRate,
      OptimizerValue globalWeightDecay,
      OptimizerValue lossScaling);

  // All non-constant SGD constructor
  SGD(float lr, float wd = 0, float ls = 1)
      : SGD({lr, false}, {wd, false}, {ls, false}) {}

  SGD(const SGD &) = default;
  ~SGD()           = default;

  OptimizerType type() const final { return OptimizerType::SGD; }
  std::string type_s() const final { return "SGD"; }

  std::unique_ptr<Optimizer> clone() const final;

  std::unique_ptr<Op> createOp(const Tensor &weight, Graph &) const final;

  // The names of the inputs for the VarUpdateOp for "weight", with "" used as a
  // placeholder for missing (constant) inputs
  std::vector<TensorId> getInputIds(const Tensor &weight) const;

  // The names and infos of the optimizer Tensors (no ""s and not ordered by
  // input index)
  std::vector<std::tuple<TensorId, TensorInfo>>
  getOptimizerInputs(const Tensor &weight) const final;

  bool validReplacement(const Optimizer &other) const final;

  void resetTensorData(Tensor &) const final;
  void setTensorData(Tensor &) const final;

  float getStoredValue(const Tensor &opt) const;

  OptimizerValue learningRate(const Tensor &weight) const;

  // (lr/ls) Note that this scaled learning rate is const iff both the learning
  // rate and the loss scale factor are const
  OptimizerValue scaledLearningRate(const Tensor &weight) const;

  OptimizerValue weightDecay(const Tensor &weight) const;

  // (1 - lr* wd) Note that this weight decay scale factor is const iff the
  // learning rate and the weight decay are const
  OptimizerValue weightDecayScaleFactor(const Tensor &weight) const;

  void insertSpecific(const TensorId &,
                      OptimizerValue weightDecay,
                      OptimizerValue learningRate);

  float getGlobalLearningRateVal() const { return lrs.getGlobal().val(); }
  float getGlobalWeightDecayVal() const { return wds.getGlobal().val(); }

private:
  // These functions are pure string manipulation
  TensorId getScaledLearningRateId(const Tensor &weight) const;
  TensorId getWeightDecayScaleFactorId(const Tensor &weight) const;

  // if any of the constituent components of the weight decay scale factor (lr,
  // ls, wd) can be modified at run-time, then it is not const
  bool weightDecayScaleFactorIsConst(const Tensor &weight) const;
  float weightDecayScaleFactorVal(const Tensor &weight) const;

  bool scaledLearningRateIsConst(const Tensor &weight) const;
  float scaledLearningRateVal(const Tensor &weight) const;

  OptimizerValueMap lrs;
  OptimizerValueMap wds;

  std::string stripWeightIdFromSpecificLearningRate(const TensorId &) const;
  std::string stripWeightIdFromSpecificWeightDecay(const TensorId &) const;

  // We will add momentum here at a later point
};

// This class is kept to be backwards compatible with the Python API
class ConstSGD : public SGD {
public:
  ConstSGD(float lr, float wd = 0, float ls = 1)
      : SGD({lr, true}, {wd, true}, {ls, true}) {}
};

} // namespace popart

#endif
