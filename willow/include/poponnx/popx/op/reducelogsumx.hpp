#ifndef GUARD_NEURALNET_REDUCELOGSUMX_HPP
#define GUARD_NEURALNET_REDUCELOGSUMX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class ReduceLogSumOp;

namespace popx {

class ReduceLogSumOpx : public Opx {
public:
  ReduceLogSumOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

class ReduceLogSumGradOpx : public Opx {
public:
  ReduceLogSumGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

} // namespace popx
} // namespace poponnx

#endif
