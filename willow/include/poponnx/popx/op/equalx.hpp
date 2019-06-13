#ifndef GUARD_NEURALNET_EQUALX_HPP
#define GUARD_NEURALNET_EQUALX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {

class EqualOp;

namespace popx {

class EqualOpx : public BinaryComparisonOpx {
public:
  EqualOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
