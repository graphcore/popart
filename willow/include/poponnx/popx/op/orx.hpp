#ifndef GUARD_NEURALNET_ORX_HPP
#define GUARD_NEURALNET_ORX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {

class OrOp;

namespace popx {

class OrOpx : public BinaryComparisonOpx {
public:
  OrOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
