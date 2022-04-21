// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ALLREDUCEX_HPP
#define GUARD_NEURALNET_ALLREDUCEX_HPP

#include <popart/names.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class AllReduceOpx : public PopOpx {
public:
  AllReduceOpx(Op *, Devicex *);

  void grow(snap::program::Sequence &) const;

  InputCreatorType getInputCreatorType(int index0) const;

  snap::Tensor unwindTensorLayout(snap::Tensor tensor, InIndex, OutIndex) const;

  view::RegMap unwindRegion(InIndex, OutIndex) const;

protected:
  CollectiveOperator op;
  int numInputs;
};

} // namespace popx
} // namespace popart

#endif
