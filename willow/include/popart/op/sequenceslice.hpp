// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SEQUENCESLICE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SEQUENCESLICE_HPP_

#include <memory>
#include <tuple>
#include <vector>
#include <poprithms/memory/inplace/proposal.hpp>
#include <popart/op.hpp>

#include "popart/names.hpp"

namespace popart {
class AliasModel;
struct OperatorIdentifier;

class SequenceSliceOp : public Op {
public:
  SequenceSliceOp(const OperatorIdentifier &,
                  bool zeroUnused,
                  const Op::Settings &);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  void setup() override;

  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  // The source tensor.
  static InIndex getSourceInIndex() { return 0; }
  // The destination tensor.
  static InIndex getDestinationInIndex() { return 1; }
  // The number of elements to copy.
  static InIndex getNInIndex() { return 2; }
  // The first element read from source.
  static InIndex getSourceOffsetInIndex() { return 3; }
  // The first element written to destination.
  static InIndex getDestOffsetInIndex() { return 4; }

  static OutIndex getOutIndex() { return 0; }

  void growAliasModel(AliasModel &) const override;

  poprithms::memory::inplace::Proposal
  mapInplaceProposal(const AliasModel &, OperatorIdentifier) const override;

  const bool zeroUnused;
};

class SequenceSliceInplaceOp : public SequenceSliceOp {
public:
  SequenceSliceInplaceOp(const OperatorIdentifier &,
                         bool zeroUnused,
                         const Op::Settings &);

  std::unique_ptr<Op> clone() const override;

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  // This Op aliases and modifies the input
  view::Regions aliases(InIndex, OutIndex) const final;
  view::Regions modifies(InIndex) const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SEQUENCESLICE_HPP_
