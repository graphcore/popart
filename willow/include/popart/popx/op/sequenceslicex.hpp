// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SEQUENCESLICEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SEQUENCESLICEX_HPP_

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

class SequenceSliceOpx : public Opx {
public:
  SequenceSliceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class SequenceSliceInplaceOpx : public Opx {
public:
  SequenceSliceInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SEQUENCESLICEX_HPP_
