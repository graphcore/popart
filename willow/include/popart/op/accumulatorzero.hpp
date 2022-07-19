// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_ACCUMULATORZERO_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_ACCUMULATORZERO_HPP_

#include <memory>
#include <popart/op/accumulatorscale.hpp>
#include <popart/optimizervalue.hpp>

#include "popart/op.hpp"

namespace popart {

/**
 * @brief An AccumulatorScaleOp with a factor of 0, so zeroes the input tensor.
 *
 */
class AccumulatorZeroOp : public AccumulatorScaleOp {
public:
  AccumulatorZeroOp(const Op::Settings &settings)
      : AccumulatorScaleOp(0.0, settings) {}
  std::unique_ptr<Op> clone() const override;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_ACCUMULATORZERO_HPP_
