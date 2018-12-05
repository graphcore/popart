#ifndef GUARD_NEURALNET_SINX_HPP
#define GUARD_NEURALNET_SINX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

namespace popx {

class SinOpx : public Opx {
public:
  SinOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
