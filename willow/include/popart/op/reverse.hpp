// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_REVERSE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_REVERSE_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <vector>
#include <poprithms/memory/inplace/proposal.hpp>
#include <popart/op.hpp>
#include <popart/region.hpp> // IWYU pragma: keep

#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/operatoridentifier.hpp"

namespace popart {
class AliasModel;
class OpSerialiserBase;

// This Op matches the snap::Tensor::reverse function, except it allows
// you to reverse along multiple dimensions in one go

class ReverseBaseOp : public Op {
public:
  ReverseBaseOp(const OperatorIdentifier &_opid,
                const Op::Settings &settings_,
                const std::vector<int64_t> &dimensions_)
      : Op(_opid, settings_), dimensions(dimensions_) {}
  std::unique_ptr<Op> clone() const override;

  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  bool canBeReplacedByIdentity() const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  std::vector<int64_t> getDimensions() const { return dimensions; }

  void growAliasModel(AliasModel &) const override;

private:
  std::vector<int64_t> dimensions;
};

class ReverseOp : public ReverseBaseOp {
public:
  ReverseOp(const OperatorIdentifier &_opid,
            const Op::Settings &settings_,
            const std::vector<int64_t> &dimensions_)
      : ReverseBaseOp(_opid, settings_, dimensions_) {}

  std::unique_ptr<Op> clone() const override;

  std::vector<std::unique_ptr<Op>> getGradOps() final;

  poprithms::memory::inplace::Proposal
  mapInplaceProposal(const AliasModel &, OperatorIdentifier) const override;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final {
    return {{Onnx::CustomOperators::ReverseInplace, 10}};
  }
};

class ReverseInplaceOp : public ReverseBaseOp {
public:
  ReverseInplaceOp(const ReverseOp &op)
      : ReverseBaseOp(Onnx::CustomOperators::ReverseInplace,
                      op.settings,
                      op.getDimensions()) {}

  std::unique_ptr<Op> clone() const final;

  // modifies and uses are still the defaults, but aliases changes
  // to be the same as uses (the full out region)
  view::Regions aliases(InIndex in, OutIndex) const final { return uses(in); }
};

// The gradient of reverse is also a reverse
class ReverseGradOp : public ReverseOp {
public:
  ReverseGradOp(const ReverseOp &);
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_REVERSE_HPP_
