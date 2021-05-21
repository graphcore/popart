// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ACCUMULATORSCALEOPX_HPP
#define GUARD_NEURALNET_ACCUMULATORSCALEOPX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/varupdatex.hpp>

namespace popart {
namespace popx {

class AccumulatorScaleOpx : public VarUpdateOpx {
public:
  AccumulatorScaleOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
