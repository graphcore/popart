#ifndef GUARD_NEURALNET_MULX_HPP
#define GUARD_NEURALNET_MULX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>
#include <poponnx/popx/op/reducesumx.hpp>

namespace poponnx {

class MulOp;

namespace popx {

class MulOpx : public ElementWiseBinaryOpx {
public:
  MulOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
