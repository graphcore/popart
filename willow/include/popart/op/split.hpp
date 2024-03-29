// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SPLIT_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SPLIT_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <vector>
#include <popart/op.hpp>

#include "popart/names.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
struct OperatorIdentifier;

class SplitOp : public Op {
public:
  SplitOp(const OperatorIdentifier &,
          int64_t axis_,
          const std::vector<int64_t> split_,
          const Op::Settings &);

  void setup() final;
  std::unique_ptr<Op> clone() const final;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::vector<int64_t> getSplitSizes() const;
  int64_t getAxis() const { return axis; }

  static InIndex getInIndex() { return 0; }

  bool canShard() const override { return true; }

private:
  const int64_t axis;
  const std::vector<int64_t> split;
};

class SplitGradOp : public Op {
public:
  SplitGradOp(const SplitOp &, const Op::Settings &);

  void setup() final;
  std::unique_ptr<Op> clone() const override;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  int64_t getAxis() const { return axis; }

  static OutIndex getOutIndex() { return 0; }

private:
  std::vector<GradInOutMapper> gradInInfo;
  std::map<int, int> outInfoMap;

  const TensorInfo fwdOpInInfo;
  const int64_t axis;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SPLIT_HPP_
