#ifndef GUARD_NEURALNET_SQUAREX_HPP
#define GUARD_NEURALNET_SQUAREX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {

namespace popx {

class SquareOpx : public ElementWiseUnaryOpx {
public:
  SquareOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
