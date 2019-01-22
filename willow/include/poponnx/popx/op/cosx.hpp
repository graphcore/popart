#ifndef GUARD_NEURALNET_COSX_HPP
#define GUARD_NEURALNET_COSX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {

namespace popx {

class CosOpx : public ElementWiseUnaryOpx {
public:
  CosOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
