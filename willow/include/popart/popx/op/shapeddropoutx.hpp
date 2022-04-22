// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SHAPEDDROPOUTX_HPP
#define GUARD_NEURALNET_SHAPEDDROPOUTX_HPP

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

class ShapedDropoutOpx : public PopOpx {
public:
  ShapedDropoutOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;

private:
  snap::Tensor getReferenceTensor() const;
};

} // namespace popx
} // namespace popart

#endif
