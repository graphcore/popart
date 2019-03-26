#ifndef GUARD_NEURALNET_CASTX_HPP
#define GUARD_NEURALNET_CASTX_HPP

#include <poponnx/popx/opx.hpp>

namespace poponnx {

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
} // namespace poponnx

#endif
