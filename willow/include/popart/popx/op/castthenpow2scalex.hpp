// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_CASTTHENPOW2SCALEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_CASTTHENPOW2SCALEX_HPP_

#include "popart/names.hpp"
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
 * Opx for the CastThenPow2Scale op.
 *
 */
class CastThenPow2ScaleOpx : public Opx {
public:
  CastThenPow2ScaleOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  /**
   * Note to create the input tensor at index 0, the metadata tensor must
   * already exist.
   *
   * \returns std::set<TensorId> The scale bias tensor if InIndex 0 is passed,
   * otherwise {}.
   */
  std::set<TensorId> mustExistBeforeCreate(InIndex) const final;

  /**
   * Get the Source Format for the input tensor (index 0).
   *
   * \returns poplar::QuarterMetadata::Format The poplar format of the input
   * tensor.
   */
  poplar::QuarterMetadata::Format getSourceFormat() const;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_CASTTHENPOW2SCALEX_HPP_
