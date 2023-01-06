// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_LRNX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_LRNX_HPP_

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

class LRNOpx : public Opx {
public:
  LRNOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class LRNGradOpx : public Opx {
public:
  LRNGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_LRNX_HPP_
