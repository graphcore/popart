#ifndef GUARD_NEURALNET_REDUCELOGSUMEXPX_HPP
#define GUARD_NEURALNET_REDUCELOGSUMEXPX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class ReduceLogSumExpOp;

namespace popx {

class ReduceLogSumExpOpx : public Opx {
public:
  ReduceLogSumExpOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

class ReduceLogSumExpGradOpx : public Opx {
public:
  ReduceLogSumExpGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

} // namespace popx
} // namespace poponnx

#endif
