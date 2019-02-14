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

  // This Op modifies the input at index getVarIndex()
  view::Region modifies(InIndex) const final;

private:
  TensorId varId;
  TensorId varGradId;
};

class SGDVarUpdateOp : public VarUpdateOp {
public:
  SGDVarUpdateOp(TensorId, const Op::Settings &);
  std::unique_ptr<Op> clone() const final;

  static InIndex getLearnRateInIndex() { return 2; }
};

class ConstSGDVarUpdateOp : public VarUpdateOp {
public:
  ConstSGDVarUpdateOp(TensorId, float learnRate, const Op::Settings &);
  std::unique_ptr<Op> clone() const final;
  float getLearnRate() const;

private:
  float learnRate;
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
