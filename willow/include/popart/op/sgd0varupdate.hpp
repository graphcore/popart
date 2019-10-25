#ifndef GUARD_NEURALNET_SGD0VARUPDATE_HPP
#define GUARD_NEURALNET_SGD0VARUPDATE_HPP

#include <popart/op/varupdate.hpp>

namespace popart {

// The "0" in the name signifies that there are no persistant Tensors required
// and associated to the Variable Tensor being updated. Specifically, there is
// no gradient accumulation and no momentum (momentum factor is 0)
class SGD0VarUpdateOp : public VarUpdateWithUpdaterOp {
public:
  SGD0VarUpdateOp(const TensorId &varToUpdate,
                  OptimizerValue initialSlr0,
                  OptimizerValue initialWdsf0,
                  const Op::Settings &);

  std::unique_ptr<Op> clone() const final;
  std::unique_ptr<Op> cloneWithNewName(const TensorId &newName) const final;

  // If the scaled learning rate is not constant, this is the index at which it
  // will be consumed by this Op
  static InIndex getSlr0InIndex() { return 2; }

  // If the weight decay scale factor is not constant, this is the index at
  // which it will be consumed by this Op
  static InIndex getWdsf0InIndex() { return 3; }

  // map of size 0/1/2, containing all non-const optimizer Tensors for this Op
  std::map<InIndex, TensorId> optimizerInputs() const final;

  // scaled learning rate
  const OptimizerValue initSlr0;

  // weight decay scaling factor
  const OptimizerValue initWdsf0;

  void appendAttributes(OpSerialiserBase &) const final;
};

} // namespace popart

#endif
