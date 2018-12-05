#ifndef GUARD_NEURALNET_COSX_HPP
#define GUARD_NEURALNET_COSX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

namespace popx {

class CosOpx : public Opx {
public:
  CosOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
