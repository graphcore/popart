// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SCATTER_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SCATTER_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <vector>
#include <popart/op.hpp>
#include <popart/vendored/optional.hpp>

#include "popart/names.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

class ScatterOp : public Op {
public:
  ScatterOp(const OperatorIdentifier &_opid,
            int64_t axis_,
            const Op::Settings &settings_,
            const nonstd::optional<float> &available_memory_proportion_ =
                nonstd::nullopt);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  // Which axis to scatter on.
  int64_t getAxis() const;

  static InIndex dataInIndex() { return 0; }
  static InIndex indicesInIndex() { return 1; }
  static InIndex updatesInIndex() { return 2; }
  static OutIndex outIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  nonstd::optional<float> getAvailableMemoryProportion() const {
    return available_memory_proportion;
  }

  void setAvailableMemoryProportion(float v) {
    available_memory_proportion = v;
  }

private:
  int64_t axis = 0;
  nonstd::optional<float> available_memory_proportion;
};

// This is a scatter of zeros into the grad input. This is because these
// elements are replaced in the forward op by the update input tensor.
class ScatterDataGradOp : public Op {
public:
  ScatterDataGradOp(const ScatterOp &op, int64_t axis);

  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  // Which axis the forward op scattered on.
  int64_t getAxis() const;

  static InIndex gradInIndex() { return 0; }
  static InIndex indicesInIndex() { return 1; }
  static OutIndex gradOutIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  nonstd::optional<float> getAvailableMemoryProportion() const {
    return available_memory_proportion;
  }

private:
  int64_t axis;
  nonstd::optional<float> available_memory_proportion;
};

// This is a gather of elements from the grad input based on the indices used in
// the forward op.
class ScatterUpdateGradOp : public Op {
public:
  ScatterUpdateGradOp(const ScatterOp &op, int64_t axis);

  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  // Which axis the forward op scattered on.
  int64_t getAxis() const;

  static InIndex gradInIndex() { return 0; }
  static InIndex indicesInIndex() { return 1; }
  static OutIndex gradOutIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  nonstd::optional<float> getAvailableMemoryProportion() const {
    return available_memory_proportion;
  }

private:
  int64_t axis;
  nonstd::optional<float> available_memory_proportion;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SCATTER_HPP_
