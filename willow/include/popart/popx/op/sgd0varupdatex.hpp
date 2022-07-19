// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SGD0VARUPDATEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SGD0VARUPDATEX_HPP_

#include <popart/popx/op/varupdatex.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class SGD0VarUpdateOpx : public VarUpdateOpx {
public:
  SGD0VarUpdateOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  // does not create inputs
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SGD0VARUPDATEX_HPP_
