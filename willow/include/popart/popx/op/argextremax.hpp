// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_ARGEXTREMAX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_ARGEXTREMAX_HPP_

#include <poplar/Tensor.hpp>
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

class ArgExtremaOpx : public Opx {
public:
  ArgExtremaOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;

private:
  virtual poplar::Tensor extremaOp(poplar::program::Sequence &,
                                   const poplar::Tensor &) const = 0;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_ARGEXTREMAX_HPP_
