#ifndef GUARD_NEURALNET_GREATERX_HPP
#define GUARD_NEURALNET_GREATERX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {

class GreaterOp;

namespace popx {

class GreaterOpx : public BinaryComparisonOpx {
public:
  GreaterOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
