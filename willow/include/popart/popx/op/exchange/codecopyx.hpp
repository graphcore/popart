// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CODECOPYX_HPP
#define GUARD_NEURALNET_CODECOPYX_HPP

#include <popart/popx/op/exchange/exchangex.hpp>

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
 * Opx for the external code copy op.
 *
 */
class ExternalCodeCopyOpx : public ExchangeBaseOpx {
public:
  ExternalCodeCopyOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
