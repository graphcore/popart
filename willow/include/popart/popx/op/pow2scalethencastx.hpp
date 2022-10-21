// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_POW2SCALETHENCASTX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_POW2SCALETHENCASTX_HPP_

#include <popart/popx/opx.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

/**
 * Opx for the Pow2ScaleThenCast op.
 *
 */
class Pow2ScaleThenCastOpx : public Opx {
public:
  Pow2ScaleThenCastOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  /**
   * Get the Destination Format for the output tensor (index 0).
   *
   * \returns poplar::QuarterMetadata::Format The poplar format of the output
   * tensor.
   */
  poplar::QuarterMetadata::Format getDestinationFormat() const;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_POW2SCALETHENCASTX_HPP_
