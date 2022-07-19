// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SHAPEDDROPOUTX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SHAPEDDROPOUTX_HPP_

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

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SHAPEDDROPOUTX_HPP_
