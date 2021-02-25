// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SEQUENCESLICEX_HPP
#define GUARD_NEURALNET_SEQUENCESLICEX_HPP

#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

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

#endif
