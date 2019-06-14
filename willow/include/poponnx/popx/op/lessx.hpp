#ifndef GUARD_NEURALNET_LESSX_HPP
#define GUARD_NEURALNET_LESSX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {

class LessOp;

namespace popx {

class LessOpx : public BinaryComparisonOpx {
public:
  LessOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
