#ifndef GUARD_NEURALNET_IFX_HPP
#define GUARD_NEURALNET_IFX_HPP

#include <poponnx/popx/opx.hpp>

namespace poponnx {

namespace popx {

class IfOpx : public Opx {
public:
  IfOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
