#ifndef GUARD_NEURALNET_REDUCEMEANX_HPP
#define GUARD_NEURALNET_REDUCEMEANX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class ReduceMeanOp;

namespace popx {

class ReduceMeanOpx : public Opx {
public:
  ReduceMeanOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

class ReduceMeanGradOpx : public Opx {
public:
  ReduceMeanGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
};

} // namespace popx
} // namespace poponnx

#endif
