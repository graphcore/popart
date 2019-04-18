#ifndef GUARD_NEURALNET_VARUPDATE_HPP
#define GUARD_NEURALNET_VARUPDATE_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class VarUpdateOp : public Op {
public:
  VarUpdateOp(const OperatorIdentifier &opid,
              TensorId,
              const Op::Settings &settings_);
  void setup() final;

  static InIndex getVarInIndex() { return 0; }
  static InIndex getVarGradInIndex() { return 1; }

  // Returns a reference to the updated Var. Note that this
  // is the only Op which modifies an input AND does not
  // have "Inplace"  in its name (16.04.2019).
  static OutIndex getUpdatedVarOutIndex() { return 0; }

  // This Op aliases and modifies the input at index getVarIndex()
  view::Region aliases(InIndex) const final;
  view::Region modifies(InIndex) const final;
  const TensorId &getVarId() const { return varId; }

  float getSubgraphValue() const final { return 0.1f; }

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

private:
  float learnRate;
  float weightDecay;
};

class CopyVarUpdateOp : public VarUpdateOp {
public:
  CopyVarUpdateOp(TensorId to, const Op::Settings &);
  std::unique_ptr<Op> clone() const final;

  static InIndex getVarToInIndex() { return 0; }
  static InIndex getVarFromInIndex() { return 1; }

private:
};

} // namespace poponnx

#endif
