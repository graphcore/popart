#ifndef GUARD_NEURALNET_VARUPDATE_HPP
#define GUARD_NEURALNET_VARUPDATE_HPP

#include <poponnx/names.hpp>
#include <poponnx/op.hpp>

namespace poponnx {

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

  // Returns a reference to the updated Var. Note that this
  // is the only Op which modifies an input AND does not
  // have "Inplace"  in its name (16.04.2019).
  static OutIndex getUpdatedVarOutIndex() { return 0; }

  // This Op aliases and modifies the input at index getVarIndex()
  view::Region aliases(InIndex) const final;
  view::Region modifies(InIndex) const final;
  const TensorId &getVarId() const { return varId; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

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
  SGDVarUpdateOp(TensorId, const Op::Settings &);
  std::unique_ptr<Op> clone() const final;

  static InIndex getLearnRateInIndex() { return 2; }
  static InIndex getWeightDecayInIndex() { return 3; }

  std::map<InIndex, TensorId> optimizerInputs() const final {
    return {{getLearnRateInIndex(), inId(getLearnRateInIndex())},
            {getWeightDecayInIndex(), inId(getWeightDecayInIndex())}};
  }

  std::unique_ptr<Op> cloneWithNewName(const TensorId &id) const final {
    return std::unique_ptr<Op>(new SGDVarUpdateOp(id, settings));
  }
};

class ConstSGDVarUpdateOp : public VarUpdateOp {
public:
  ConstSGDVarUpdateOp(TensorId,
                      float learnRate,
                      float weightDecay,
                      const Op::Settings &);
  std::unique_ptr<Op> clone() const final;
  float getLearnRate() const;
  float getWeightDecay() const;
  void appendAttributes(OpSerialiserBase &is) const;

  std::unique_ptr<Op> cloneWithNewName(const TensorId &id) const final {
    return std::unique_ptr<Op>(
        new ConstSGDVarUpdateOp(id, learnRate, weightDecay, settings));
  }

  std::map<InIndex, TensorId> optimizerInputs() const final { return {}; }

private:
  float learnRate;
  float weightDecay;
};

class CopyVarUpdateOp : public VarUpdateOp {
public:
  CopyVarUpdateOp(TensorId to, const Op::Settings &);
  std::unique_ptr<Op> clone() const final;

  std::unique_ptr<Op> cloneWithNewName(const TensorId &id) const final {
    return std::unique_ptr<Op>(new CopyVarUpdateOp(id, settings));
  }

  std::map<InIndex, TensorId> optimizerInputs() const final { return {}; }
};

} // namespace poponnx

#endif
