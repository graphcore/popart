// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_GATHER_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_GATHER_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <vector>
#include <popart/op.hpp>
#include <popart/vendored/optional.hpp>

#include "popart/names.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

class GatherOp : public Op {
public:
  GatherOp(const OperatorIdentifier &_opid,
           int64_t axis_,
           const Op::Settings &settings_,
           const nonstd::optional<float> &available_memory_proportion_ =
               nonstd::nullopt,
           bool zeroOutOfRangeIndices_ = false);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() override;
  void setup() final;

  // Which axis to gather on.
  int64_t getAxis() const;

  static InIndex dataInIndex() { return 0; }
  static InIndex indicesInIndex() { return 1; }
  static OutIndex outIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  bool canBeReplacedByIdentity() const override;

  float getSubgraphValue() const override { return getLowSubgraphValue(); }

  bool canShard() const override { return true; }

  nonstd::optional<float> getAvailableMemoryProportion() const {
    return available_memory_proportion;
  }

  void setAvailableMemoryProportion(const nonstd::optional<float> v) {
    available_memory_proportion = v;
  }

  bool zeroOutOfRangeIndices() const { return zeroOutOfRangeIndices__; }

private:
  int64_t axis = 0;
  nonstd::optional<float> available_memory_proportion;
  bool zeroOutOfRangeIndices__;
};

class GatherGradOp : public Op {
public:
  GatherGradOp(const GatherOp &op, int64_t axis);

  std::unique_ptr<Op> clone() const override;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  // Which axis to gather on.
  int64_t getAxis() const;

  static InIndex gradInIndex() { return 0; }
  static InIndex indicesInIndex() { return 1; }
  static OutIndex gradOutIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  int getInBatchAxis(InIndex i) const override { return 0; }
  int getOutBatchAxis(OutIndex) const override { return -1; }

  bool canShard() const override { return true; }

  nonstd::optional<float> getAvailableMemoryProportion() const {
    return available_memory_proportion;
  }
  void setAvailableMemoryProportion(const nonstd::optional<float> v) {
    available_memory_proportion = v;
  }

private:
  int64_t axis;
  TensorInfo fwdDataInfo;
  nonstd::optional<float> available_memory_proportion;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_GATHER_HPP_
