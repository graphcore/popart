#ifndef GUARD_NEURALNET_COPYVARUPDATE_HPP
#define GUARD_NEURALNET_COPYVARUPDATE_HPP

#include <popart/op/varupdate.hpp>

namespace popart {

class CopyVarUpdateOp : public VarUpdateOp {
public:
  CopyVarUpdateOp(TensorId to, const Op::Settings &);
  std::unique_ptr<Op> clone() const final;

  std::unique_ptr<Op> cloneWithNewName(const TensorId &updatedTo) const final {
    return std::unique_ptr<Op>(new CopyVarUpdateOp(updatedTo, settings));
  }

  std::map<InIndex, TensorId> optimizerInputs() const final { return {}; }
};

} // namespace popart

#endif
