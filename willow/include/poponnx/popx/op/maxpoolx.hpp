#ifndef GUARD_NEURALNET_MAXPOOLX_HPP
#define GUARD_NEURALNET_MAXPOOLX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/poolx.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class MaxPoolOp;
class MaxPoolGradOp;

namespace popx {

class MaxPoolOpx : public PoolOpx {
public:
  MaxPoolOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class MaxPoolGradOpx : public PoolOpx {
public:
  MaxPoolGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
