#ifndef GUARD_NEURALNET_VARUPDATE_HPP
#define GUARD_NEURALNET_VARUPDATE_HPP

#include <willow/ir.hpp>

namespace willow {

class VarUpdateOp : public Op {
public:
  // op_type: passed down from class which inherits
  VarUpdateOp(std::string op_type, TensorId, Ir *);
  virtual void setup() override final;
  static int getVarIndex();
  static int getVarGradIndex();
  // Impose the rule that:
  // This must be the final operation on the Variable
  // Tensor, as it is modified by this Op.
  void imposeTopoCons() override final;

private:
  TensorId varId;
  TensorId varGradId;
};

class SGDVarUpdateOp : public VarUpdateOp {
public:
  SGDVarUpdateOp(TensorId, Ir *);
  virtual std::unique_ptr<Op> clone() const override final;
  static int getLearnRateIndex();
};

class ConstSGDVarUpdateOp : public VarUpdateOp {
public:
  ConstSGDVarUpdateOp(TensorId, Ir *);
  virtual std::unique_ptr<Op> clone() const override final;
};

} // namespace willow

#endif
