#ifndef GUARD_NEURALNET_SGD1VARUPDATEACCLUPDATEOPOP_HPP
#define GUARD_NEURALNET_SGD1VARUPDATEACCLUPDATEOPOP_HPP

#include <popart/op/varupdate.hpp>
#include <popart/optimizervalue.hpp>

namespace popart {

class SGD1AcclUpdateOp : public VarUpdateWithUpdaterOp {

public:
  SGD1AcclUpdateOp(const TensorId &varToUpdate,
                   OptimizerValue initSmm1,
                   OptimizerValue initSwd1,
                   const Op::Settings &);

  std::unique_ptr<Op> clone() const final;
  std::unique_ptr<Op> cloneWithNewName(const TensorId &newName) const final;
  std::map<InIndex, TensorId> optimizerInputs() const final;
  void appendOutlineAttributes(OpSerialiserBase &) const final;

  const OptimizerValue initSmm1;
  const OptimizerValue initSwd1;
  static InIndex getSmm1InIndex() { return 2; }
  static InIndex getSwd1InIndex() { return 3; }
};

} // namespace popart

#endif
