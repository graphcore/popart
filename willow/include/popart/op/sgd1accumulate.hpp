#ifndef GUARD_NEURALNET_SGD1VARUPDATEACCUMULATEOP_HPP
#define GUARD_NEURALNET_SGD1VARUPDATEACCUMULATEOP_HPP

#include <popart/op/varupdate.hpp>
#include <popart/optimizervalue.hpp>

namespace popart {

class SGD1AccumulateOp : public VarUpdateWithUpdaterOp {

public:
  SGD1AccumulateOp(const TensorId &varToUpdate,
                   OptimizerValue initDpfs1,
                   const Op::Settings &);

  std::unique_ptr<Op> clone() const final;
  std::unique_ptr<Op> cloneWithNewName(const TensorId &newName) const final;
  std::map<InIndex, TensorId> optimizerInputs() const final;
  void appendOutlineAttributes(OpSerialiserBase &) const final;

  const OptimizerValue initDpsf1;
  static InIndex getDpsf1InIndex() { return 2; }
};

} // namespace popart

#endif
