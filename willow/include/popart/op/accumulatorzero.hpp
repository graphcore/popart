// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ACCUMULATORZEROOP_HPP
#define GUARD_NEURALNET_ACCUMULATORZEROOP_HPP

#include <popart/op/accumulatorscale.hpp>
#include <popart/optimizervalue.hpp>

namespace popart {

// Reset all values of the accumulator to zero
class AccumulatorZeroOp : public AccumulatorScaleOp {
public:
  AccumulatorZeroOp(const Op::Settings &settings)
      : AccumulatorScaleOp(0.0, settings) {}
};

} // namespace popart

#endif
