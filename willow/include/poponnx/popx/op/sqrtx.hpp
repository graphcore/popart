#ifndef GUARD_NEURALNET_SQRTX_HPP
#define GUARD_NEURALNET_SQRTX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

namespace popx {

class SqrtOpx : public Opx {
public:
  SqrtOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
