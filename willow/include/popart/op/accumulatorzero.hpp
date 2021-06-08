// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ACCUMULATORZEROOP_HPP
#define GUARD_NEURALNET_ACCUMULATORZEROOP_HPP

#include <popart/op/accumulatorscale.hpp>
#include <popart/optimizervalue.hpp>

namespace popart {

/**
 * @brief An AccumulatorScaleOp with a factor of 0, so zeroes the input tensor.
 *
 */
class AccumulatorZeroOp : public AccumulatorScaleOp {
public:
  AccumulatorZeroOp(const Op::Settings &settings)
      : AccumulatorScaleOp(0.0, settings) {}
};

} // namespace popart

#endif
