#ifndef GUARD_NEURALNET_POWX_HPP
#define GUARD_NEURALNET_POWX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>
#include <poponnx/popx/op/reducesumx.hpp>

namespace poponnx {

class PowOp;

namespace popx {

class PowOpx : public ElementWiseBinaryOpx {
public:
  PowOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
