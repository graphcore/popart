#ifndef GUARD_NEURALNET_MAXX_HPP
#define GUARD_NEURALNET_MAXX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {

namespace popx {

class MaxOpx : public ElementWiseUnaryOpx {
public:
  MaxOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class MaxGradOpx : public Opx {
public:
  MaxGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
