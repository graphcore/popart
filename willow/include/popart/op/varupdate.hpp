// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
/** \file varupdate.hpp
 * VarUpdate ops.
 */
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_VARUPDATE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_VARUPDATE_HPP_

#include <map>
#include <memory>
#include <popart/names.hpp>
#include <popart/op.hpp>

namespace popart {
class AliasModel;
struct OperatorIdentifier;

/**
 *  Base class used to define PopART ops that update variable tensors.
 */
class VarUpdateOp : public Op {
public:
  VarUpdateOp(const OperatorIdentifier &, const Op::Settings &);
  std::unique_ptr<Op> clone() const override = 0;

  // the Var to be updated received at this index
  static InIndex getVarToUpdateInIndex() { return 0; }

  // Return (a reference to) the updated Var at this index
  static OutIndex getUpdatedVarOutIndex() { return 0; }

  void setup() final;

  // This Op aliases and modifies the input at index getVarIndex()
  view::Regions aliases(InIndex in, OutIndex) const override;
  view::Regions modifies(InIndex) const override;

  // Return a map of all optimizer specific input Tensors (learning rate, etc)
  virtual std::map<InIndex, TensorId> optimizerInputs() const = 0;

  bool isOptimizerOp() const override { return true; }

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override;

  void growAliasModel(AliasModel &) const override;
};

class VarUpdateWithUpdaterOp : public VarUpdateOp {
public:
  VarUpdateWithUpdaterOp(const OperatorIdentifier &opid,
                         const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override = 0;

  // the gradient (for SGD) or source of copy (for CopyVarUpdate) or any other
  // tensor used to update the variable tensor is received at this index
  static InIndex getUpdaterInIndex() { return 1; }

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_VARUPDATE_HPP_
