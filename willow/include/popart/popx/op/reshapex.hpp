// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_RESHAPEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_RESHAPEX_HPP_

#include <poplar/Tensor.hpp>
#include <popart/names.hpp>
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

class ReshapeBaseOpx : public Opx {
public:
  ReshapeBaseOpx(Op *, Devicex *);
  InputCreatorType getInputCreatorType(InIndex) const final;
  poplar::Tensor unwindTensorLayout(poplar::Tensor tensor,
                                    InIndex inIndex,
                                    OutIndex outIndex) const final;
  view::RegMap unwindRegion(InIndex inIndex, OutIndex outIndex) const final;
};

class ReshapeOpx : public ReshapeBaseOpx {
public:
  ReshapeOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class ReshapeInplaceOpx : public ReshapeBaseOpx {
public:
  ReshapeInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

// The gradient of a reshape is the reshape in reverse
class ReshapeGradOpx : public ReshapeOpx {
public:
  ReshapeGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_RESHAPEX_HPP_
