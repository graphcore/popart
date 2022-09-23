// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_RESTORE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_RESTORE_HPP_

#include <cstdint>
#include <memory>
#include <popart/op.hpp>

#include "popart/names.hpp"

namespace popart {
class AliasModel;
class OpSerialiserBase;
struct OperatorIdentifier;

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
  void growAliasModel(AliasModel &m) const override { growAliasModelMulti(m); }
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_RESTORE_HPP_
