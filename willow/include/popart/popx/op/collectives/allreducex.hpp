// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_ALLREDUCEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_ALLREDUCEX_HPP_

#include <poplar/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/popx/opx.hpp>

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class AllReduceOpx : public Opx {
public:
  AllReduceOpx(Op *, Devicex *);

  void grow(poplar::program::Sequence &) const;

  InputCreatorType getInputCreatorType(int index0) const;

  poplar::Tensor
  unwindTensorLayout(poplar::Tensor tensor, InIndex, OutIndex) const;

  view::RegMap unwindRegion(InIndex, OutIndex) const;

protected:
  CollectiveOperator op;
  int numInputs;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_ALLREDUCEX_HPP_
