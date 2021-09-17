// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_WHEREX_HPP
#define GUARD_NEURALNET_WHEREX_HPP

#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class WhereOpx : public PopOpx {
public:
  WhereOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class WhereXGradOpx : public PopOpx {
public:
  WhereXGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class WhereYGradOpx : public PopOpx {
public:
  WhereYGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
