// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SEQUENCESLICEX_HPP
#define GUARD_NEURALNET_SEQUENCESLICEX_HPP

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

class SequenceSliceOpx : public PopOpx {
public:
  SequenceSliceOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class SequenceSliceInplaceOpx : public PopOpx {
public:
  SequenceSliceInplaceOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
