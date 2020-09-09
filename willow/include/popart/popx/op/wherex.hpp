// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_WHEREX_HPP
#define GUARD_NEURALNET_WHEREX_HPP

#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class WhereOpx : public Opx {
public:
  WhereOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class WhereXGradOpx : public Opx {
public:
  WhereXGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class WhereYGradOpx : public Opx {
public:
  WhereYGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
