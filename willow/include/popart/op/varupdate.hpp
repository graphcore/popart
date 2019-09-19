#ifndef GUARD_NEURALNET_VARUPDATE_HPP
#define GUARD_NEURALNET_VARUPDATE_HPP

#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/optimizervalue.hpp>

namespace popart {

class VarUpdateOp : public Op {
public:
  VarUpdateOp(const OperatorIdentifier &opid,
              TensorId,
              const Op::Settings &settings_);
  void setup() final;

  // the Var to be updated
  static InIndex getVarToUpdateInIndex() { return 0; }

  // the gradient or source of copy
  static InIndex getUpdaterInIndex() { return 1; }

  // Returns a reference to the updated Var
  static OutIndex getUpdatedVarOutIndex() { return 0; }

  // This Op aliases and modifies the input at index getVarIndex()
  view::Region aliases(InIndex) const final;
  view::Region modifies(InIndex) const final;
  const TensorId &getVarId() const { return varId; }

  float getSubgraphValue() const final;

  // Return a map of all optimizer specific input Tensors (learning rate, etc)
  virtual std::map<InIndex, TensorId> optimizerInputs() const = 0;

  // Create a clone, but with a new name
  virtual std::unique_ptr<Op> cloneWithNewName(const TensorId &) const = 0;

private:
  TensorId varId;
  TensorId varGradId;
};

class SGDVarUpdateOp : public VarUpdateOp {
public:
  SGDVarUpdateOp(const TensorId &, // the name of the Tensor to update
                 OptimizerValue initialScaledLearningRate,
                 OptimizerValue initialWeightDecayScaleFactor,
                 const Op::Settings &);

  std::unique_ptr<Op> clone() const final;
  std::unique_ptr<Op> cloneWithNewName(const TensorId &newName) const final;

  // If the scaled learning rate is not constant, this is the index at which it
  // will be consumed by this Op
  static InIndex getScaledLearningRateInIndex() { return 2; }

  // If the weight decay scale factor is not constant, this is the index at
  // which it will be consumed by this Op
  static InIndex getWeightDecayScaleFactorInIndex() { return 3; }

  // map of size 0/1/2, containing all non-const optimizer Tensors for this Op
  std::map<InIndex, TensorId> optimizerInputs() const final;

  const OptimizerValue initScaledLearningRate;
  const OptimizerValue initWeightDecayScaleFactor;

  void appendAttributes(OpSerialiserBase &) const final;
};

class CopyVarUpdateOp : public VarUpdateOp {
public:
  CopyVarUpdateOp(TensorId to, const Op::Settings &);
  std::unique_ptr<Op> clone() const final;

  std::unique_ptr<Op> cloneWithNewName(const TensorId &x) const final {
    return std::unique_ptr<Op>(new CopyVarUpdateOp(x, settings));
  }

  std::map<InIndex, TensorId> optimizerInputs() const final { return {}; }
};

} // namespace popart

#endif
