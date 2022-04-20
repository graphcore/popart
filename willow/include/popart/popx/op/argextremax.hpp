// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ARGEXTREMAX_HPP
#define GUARD_NEURALNET_ARGEXTREMAX_HPP

#include <snap/Tensor.hpp>
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

class ArgExtremaOpx : public PopOpx {
public:
  ArgExtremaOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;

private:
  virtual snap::Tensor extremaOp(snap::program::Sequence &,
                                 const snap::Tensor &) const = 0;
};

} // namespace popx
} // namespace popart

#endif
