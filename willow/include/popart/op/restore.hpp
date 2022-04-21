// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RESTORE_HPP
#define GUARD_NEURALNET_RESTORE_HPP

#include <popart/op.hpp>

namespace popart {

class RestoreOp : public Op {
public:
  RestoreOp(const OperatorIdentifier &,
            int64_t stashSize,
            const Op::Settings &);

  std::unique_ptr<Op> clone() const override;
  void setup() final;

  // The stash tensor from which to restore the activation tensor
  static InIndex getStashInIndex() { return 0; }

  // Returns a reference to the restored activation tensor
  static OutIndex getRestoredActOutIndex() { return 0; }

  TensorId getRestoredTensorId() const;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  int64_t getStashSize() const { return stashSize; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  bool isOutlineable() const override { return false; }

private:
  int64_t stashSize;
};

class RestoreInplaceOp : public RestoreOp {
public:
  RestoreInplaceOp(const OperatorIdentifier &,
                   int64_t stashSize,
                   const Op::Settings &);
  std::unique_ptr<Op> clone() const override;

  // The activation tensor to restore
  static InIndex getActToRestoreInIndex() { return 1; }

  // This Op aliases and modifies the input at index getVarIndex()
  view::Regions aliases(InIndex in, OutIndex) const final;
  view::Regions modifies(InIndex) const final;

  bool requiredForRecompute = false;
  virtual void growAliasModel(AliasModel &m) const override {
    growAliasModelMulti(m);
  }
};

} // namespace popart

#endif
