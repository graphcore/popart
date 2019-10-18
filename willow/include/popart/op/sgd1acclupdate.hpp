#ifndef GUARD_NEURALNET_SGD1VARUPDATEACCLUPDATEOPOP_HPP
#define GUARD_NEURALNET_SGD1VARUPDATEACCLUPDATEOPOP_HPP

#include <popart/op/varupdate.hpp>
#include <popart/optimizervalue.hpp>

namespace popart {

class SGD1AcclUpdateOp : public VarUpdateOp {

public:
  SGD1AcclUpdateOp(const TensorId &varToUpdate,
                   OptimizerValue initMm1,
                   OptimizerValue initWdsf1,
                   const Op::Settings &);

  std::unique_ptr<Op> clone() const final;
  std::unique_ptr<Op> cloneWithNewName(const TensorId &newName) const final;
  std::map<InIndex, TensorId> optimizerInputs() const final;
  void appendAttributes(OpSerialiserBase &) const final;

  const OptimizerValue initMm1;
  const OptimizerValue initWdsf1;
  static InIndex getMm1InIndex() { return 2; }
  static InIndex getWdsf1InIndex() { return 3; }
};

} // namespace popart

#endif
