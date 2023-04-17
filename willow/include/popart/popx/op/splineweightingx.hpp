// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SPLINEWEIGHTINGX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SPLINEWEIGHTINGX_HPP_

#include "popart/popx/opx.hpp"
#include <popart/names.hpp>

namespace popart {
class Op;

namespace popx {
class Devicex;

class SplineWeightingx : public Opx {
public:
  SplineWeightingx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override final;
  bool outputCreatedExternally(OutIndex index) const override;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SPLINEWEIGHTINGX_HPP_
