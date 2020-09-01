// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_WHEREX_HPP
#define GUARD_NEURALNET_WHEREX_HPP

#include <popops/ElementWise.hpp>
#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class WhereOpx : public Opx {
public:
  WhereOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
