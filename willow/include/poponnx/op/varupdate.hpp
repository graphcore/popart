#ifndef GUARD_NEURALNET_VARUPDATE_HPP
#define GUARD_NEURALNET_VARUPDATE_HPP

#include <poponnx/ir.hpp>

namespace poponnx {

class VarUpdateOp : public Op {
public:
  // op_type: passed down from class which inherits
  VarUpdateOp(std::string op_type, TensorId, Ir *);
  void setup() final;
  static int getVarIndex();
  static int getVarGradIndex();
  // This Op modifies the input at index getVarIndex()
  virtual bool modifies(InIndex) const final;

private:
  TensorId varId;
  TensorId varGradId;
};

class SGDVarUpdateOp : public VarUpdateOp {
public:
  SGDVarUpdateOp(TensorId, Ir *);
  std::unique_ptr<Op> clone() const final;
  static int getLearnRateIndex();
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
