// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SHAPEDDROPOUTX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SHAPEDDROPOUTX_HPP_

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

class ShapedDropoutOpx : public Opx {
public:
  ShapedDropoutOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;

private:
  poplar::Tensor getReferenceTensor() const;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SHAPEDDROPOUTX_HPP_
