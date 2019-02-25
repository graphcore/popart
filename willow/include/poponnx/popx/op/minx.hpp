#ifndef GUARD_NEURALNET_MINX_HPP
#define GUARD_NEURALNET_MINX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {

namespace popx {

class MinOpx : public ElementWiseUnaryOpx {
public:
  MinOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class MinGradOpx : public Opx {
public:
  MinGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
