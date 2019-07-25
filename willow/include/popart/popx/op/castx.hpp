#ifndef GUARD_NEURALNET_CASTX_HPP
#define GUARD_NEURALNET_CASTX_HPP

#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {

class CastOpx : public Opx {
public:
  CastOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class CastGradOpx : public CastOpx {
public:
  CastGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
