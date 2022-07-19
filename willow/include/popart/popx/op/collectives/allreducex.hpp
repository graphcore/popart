// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_ALLREDUCEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_ALLREDUCEX_HPP_

#include <snap/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/popx/popopx.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

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

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_ALLREDUCEX_HPP_
