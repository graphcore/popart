#ifndef GUARD_NEURALNET_VARUPDATE_HPP
#define GUARD_NEURALNET_VARUPDATE_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class VarUpdateOp : public Op {
public:
  VarUpdateOp(const OperatorIdentifier &opid, TensorId, Ir *);
  void setup() final;

  static InIndex getVarInIndex() { return 0; }
  static InIndex getVarGradInIndex() { return 1; }

  // This Op modifies the input at index getVarIndex()
  std::map<InIndex, Region>
  modifies(const std::map<InIndex, Shape> &) const final;

  // there are no aliases created, VarUpdateOp has no output

private:
  TensorId varId;
  TensorId varGradId;
};

class SGDVarUpdateOp : public VarUpdateOp {
public:
  SGDVarUpdateOp(TensorId, Ir *);
  std::unique_ptr<Op> clone() const final;

  static InIndex getLearnRateInIndex() { return 2; }
};

class ConstSGDVarUpdateOp : public VarUpdateOp {
public:
  ConstSGDVarUpdateOp(TensorId, Ir *, float learnRate);
  std::unique_ptr<Op> clone() const final;
  float getLearnRate() const;

private:
  float learnRate;
};

} // namespace poponnx

#endif
