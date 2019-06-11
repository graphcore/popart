#ifndef GUARD_NEURALNET_REDUCESUMSQUAREX_HPP
#define GUARD_NEURALNET_REDUCESUMSQUAREX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class ReduceSumSquareOp;

namespace popx {

class ReduceSumSquareOpx : public Opx {
public:
  ReduceSumSquareOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

class ReduceSumSquareGradOpx : public Opx {
public:
  ReduceSumSquareGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

} // namespace popx
} // namespace poponnx

#endif
