#ifndef GUARD_NEURALNET_GRADIENTACCLX_HPP
#define GUARD_NEURALNET_GRADIENTACCLX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {
namespace popx {

class GradientAcclOpx : public Opx {
public:
  GradientAcclOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class ResetAcclOpx : public Opx {
public:
  ResetAcclOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
