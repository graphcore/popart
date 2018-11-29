#ifndef GUARD_NEURALNET_REDUCESUMX_HPP
#define GUARD_NEURALNET_REDUCESUMX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class ReduceSumOp;

namespace popx {

class ReduceSumOpx : public Opx {
public:
  ReduceSumOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

class ReduceSumGradOpx : public Opx {
public:
  ReduceSumGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

} // namespace popx
} // namespace poponnx

#endif
