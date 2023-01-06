// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SGD1ACCLUPDATEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SGD1ACCLUPDATEX_HPP_

#include <popart/popx/op/varupdatex.hpp>

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class SGD1AcclUpdateOpx : public VarUpdateOpx {
public:
  SGD1AcclUpdateOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SGD1ACCLUPDATEX_HPP_
