// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SUBSAMPLE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SUBSAMPLE_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <vector>
#include <poprithms/memory/inplace/proposal.hpp>
#include <popart/op.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
class AliasModel;
class OpSerialiserBase;

class SubsampleBaseOp : public Op {
public:
  SubsampleBaseOp(const OperatorIdentifier &_opid,
                  const std::vector<int64_t> &strides_,
                  const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override;
  void setup() override;

  std::vector<std::unique_ptr<Op>> getGradOps() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  std::vector<int64_t> getStrides() const { return strides; }

  // The stride is a vector whose length is the rank of the input tensor
  // If strides is defined as {1,..,1} the the input tensor will not be changed
  std::vector<uint32_t> strides_u32() const;

  // Returns true if all the strides at 1
  bool strideSizeOne() const;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  bool canBeReplacedByIdentity() const override;

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  virtual void growAliasModel(AliasModel &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

public:
  std::vector<int64_t> strides;
};

class SubsampleOp : public SubsampleBaseOp {
public:
  SubsampleOp(const OperatorIdentifier &_opid,
              const std::vector<int64_t> &strides_,
              const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override;
  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final {
    return {{Onnx::CustomOperators::SubsampleInplace, 10}};
  }

  poprithms::memory::inplace::Proposal
  mapInplaceProposal(const AliasModel &, OperatorIdentifier) const override;
};

class SubsampleInplaceOp : public SubsampleBaseOp {
public:
  SubsampleInplaceOp(const SubsampleOp &);
  std::unique_ptr<Op> clone() const final;

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final {
    // this throws an error
    return Op::getInplaceVariant(o);
  }

  // modifies and uses are still the defaults, but aliases changes
  // to be the same as uses (the full out region)
  view::Regions aliases(InIndex in, OutIndex) const final;
};

class SubsampleGradOp : public Op {
public:
  SubsampleGradOp(const SubsampleBaseOp &fwdOp);
  std::unique_ptr<Op> clone() const final;
  void setup() override;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  std::vector<int64_t> getStrides() const { return strides; }

  std::vector<uint32_t> strides_u32() const;

  const Shape &getFwdInputShape() const { return fwdOpInfo.shape(); }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  std::vector<int64_t> strides;
  TensorInfo fwdOpInfo;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SUBSAMPLE_HPP_
