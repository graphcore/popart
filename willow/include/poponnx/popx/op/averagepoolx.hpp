#ifndef GUARD_NEURALNET_AVERAGEPOOLX_HPP
#define GUARD_NEURALNET_AVERAGEPOOLX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/poolx.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class AveragePoolOp;
class AveragePoolGradOp;

namespace popx {

class AveragePoolOpx : public PoolOpx {
public:
  AveragePoolOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class AveragePoolGradOpx : public PoolOpx {
public:
  AveragePoolGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
