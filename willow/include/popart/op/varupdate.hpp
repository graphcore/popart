#ifndef GUARD_NEURALNET_VARUPDATE_HPP
#define GUARD_NEURALNET_VARUPDATE_HPP

#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/optimizervalue.hpp>

namespace popart {

class VarUpdateOp : public Op {
public:
  VarUpdateOp(const OperatorIdentifier &opid,
              const TensorId &varId,
              const Op::Settings &settings_);

  void setup() final;

  // the Var to be updated received at this index
  static InIndex getVarToUpdateInIndex() { return 0; }

  // the gradient (SGD) or source of copy (CopyVarUpdate) received at this index
  static InIndex getUpdaterInIndex() { return 1; }

  // Return (a reference to) the updated Var at this index
  static OutIndex getUpdatedVarOutIndex() { return 0; }

  // This Op aliases and modifies the input at index getVarIndex()
  view::Region aliases(InIndex) const final;
  view::Region modifies(InIndex) const final;

  const TensorId &getVarId() const { return varId; }

  float getSubgraphValue() const final;

  // Return a map of all optimizer specific input Tensors (learning rate, etc)
  virtual std::map<InIndex, TensorId> optimizerInputs() const = 0;

  // Create a clone, but with a new name
  virtual std::unique_ptr<Op> cloneWithNewName(const TensorId &) const = 0;

private:
  TensorId varId;
  TensorId varGradId;
};

} // namespace popart

#endif
