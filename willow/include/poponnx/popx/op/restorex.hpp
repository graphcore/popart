#ifndef GUARD_NEURALNET_RESTOREX_HPP
#define GUARD_NEURALNET_RESTOREX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class RestoreOp;

namespace popx {

class RestoreOpx : public Opx {
public:
  RestoreOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
