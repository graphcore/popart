// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef POPART_WILLOW_INCLUDE_POPART_OP_TIEDGATHER_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_TIEDGATHER_HPP_

#include <cstdint>
#include <memory>
#include <vector>
#include <popart/op/gather.hpp>

#include "popart/op.hpp"
#include "popart/vendored/optional.hpp"

namespace popart {

class TiedGatherOp final : public GatherOp {
public:
  TiedGatherOp(int64_t axis_,
               const Op::Settings &settings_,
               const nonstd::optional<float> available_memory_proportion_ =
                   nonstd::nullopt,
               bool zeroOutOfRangeIndices_ = false);

  std::unique_ptr<Op> clone() const final;

  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class TiedGatherGradOp final : public GatherGradOp {
public:
  TiedGatherGradOp(const TiedGatherOp *fwdOp, int64_t axis);

  std::unique_ptr<Op> clone() const final;

  const TiedGatherOp *fwdOp;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_TIEDGATHER_HPP_
