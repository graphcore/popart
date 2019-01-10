#ifndef GUARD_NEURALNET_UNSQUEEZEX_HPP
#define GUARD_NEURALNET_UNSQUEEZEX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class UnsqueezeOp;
class UnsqueezeGradOp;

namespace popx {

class UnsqueezeOpx : public Opx {
public:
  UnsqueezeOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class UnsqueezeGradOpx : public Opx {
public:
  UnsqueezeGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
