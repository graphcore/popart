#ifndef GUARD_NEURALNET_RESTORE_HPP
#define GUARD_NEURALNET_RESTORE_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class RestoreOp : public Op {
public:
  RestoreOp(const OperatorIdentifier &, const Op::Settings &);

  std::unique_ptr<Op> clone() const override;
  void setup() final;

  // The activation tensor to restore
  static InIndex getActToRestoreInIndex() { return 0; }

  // The stash tensor from which to restore the activation tensor
  static InIndex getStashInIndex() { return 1; }

  // Returns a reference to the restored activation tensor
  static OutIndex getRestoredActOutIndex() { return 0; }

  TensorId getRestoredTensorId() const;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

class RestoreInplaceOp : public RestoreOp {
public:
  RestoreInplaceOp(const OperatorIdentifier &, const Op::Settings &);
  std::unique_ptr<Op> clone() const override;

  // This Op aliases and modifies the input at index getVarIndex()
  view::Region aliases(InIndex) const final;
  view::Region modifies(InIndex) const final;
};

} // namespace poponnx

#endif
