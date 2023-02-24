// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SCATTER_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SCATTER_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <vector>
#include <popart/op.hpp>
#include <popart/op/scatterreduce.hpp>
#include <popart/vendored/optional.hpp>

#include "popart/names.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

class ScatterOp : public ScatterReduceOp {
public:
  ScatterOp(const OperatorIdentifier &_opid,
            int64_t axis_,
            const Op::Settings &settings_,
            const nonstd::optional<float> &available_memory_proportion_ =
                nonstd::nullopt);

  static InIndex dataInIndex() { return 0; }
  static InIndex indicesInIndex() { return 1; }
  static InIndex updatesInIndex() { return 2; }
  static OutIndex outIndex() { return 0; }
  // The position of the corresponding tensors differs between scatter and
  // scatter reduce ops.
  InIndex srcDataInIndex() const noexcept override;
  InIndex initialValuesInIndex() const noexcept override;

  std::unique_ptr<Op> clone() const override final;
  std::vector<std::unique_ptr<Op>> getGradOps() override final;

  void appendOutlineAttributes(OpSerialiserBase &) const override;
};

// This is a scatter of zeros into the grad input. This is because these
// elements are replaced in the forward op by the update input tensor.
class ScatterDataGradOp : public Op {
public:
  ScatterDataGradOp(const ScatterOp &op, int64_t axis);

  static InIndex gradInIndex() { return 0; }
  static InIndex indicesInIndex() { return 1; }
  static OutIndex gradOutIndex() { return 0; }

  std::unique_ptr<Op> clone() const override final;
  const std::vector<GradInOutMapper> &gradInputInfo() const override final;
  const std::map<int, int> &gradOutToNonGradIn() const override final;
  void setup() override final;
  void appendOutlineAttributes(OpSerialiserBase &) const override;
  float getSubgraphValue() const override final;

  int64_t getAxis() const noexcept;
  nonstd::optional<float> getAvailableMemoryProportion() const noexcept;

private:
  int64_t axis;
  nonstd::optional<float> available_memory_proportion;
};

// This is a gather of elements from the grad input based on the indices used in
// the forward op.
class ScatterUpdateGradOp : public Op {
public:
  ScatterUpdateGradOp(const ScatterOp &op, int64_t axis);

  static InIndex gradInIndex() { return 0; }
  static InIndex indicesInIndex() { return 1; }
  static OutIndex gradOutIndex() { return 0; }

  std::unique_ptr<Op> clone() const override final;
  void setup() override final;
  const std::vector<GradInOutMapper> &gradInputInfo() const override final;
  const std::map<int, int> &gradOutToNonGradIn() const override final;
  void appendOutlineAttributes(OpSerialiserBase &) const override;
  float getSubgraphValue() const override final;

  int64_t getAxis() const noexcept;
  nonstd::optional<float> getAvailableMemoryProportion() const noexcept;

private:
  int64_t axis;
  nonstd::optional<float> available_memory_proportion;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SCATTER_HPP_
