// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SORT_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SORT_HPP_

#include <popart/op/topk.hpp>

namespace popart {

class SortOp : public TopKOp {
public:
  SortOp(const OperatorIdentifier &opid,
         int64_t axis,
         int64_t axis_size,
         bool descending,
         bool stable,
         const Op::Settings &settings,
         const nonstd::optional<float> &available_memory_proportion =
             nonstd::nullopt);
  std::unique_ptr<Op> clone() const override;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SORT_HPP_
