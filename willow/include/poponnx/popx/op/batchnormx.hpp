#ifndef GUARD_NEURALNET_BATCHNORMX_HPP
#define GUARD_NEURALNET_BATCHNORMX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class BatchNormOp;
class BatchNormGradOp;

namespace popx {

class BatchNormOpx : public Opx {
public:
  BatchNormOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  poplar::Tensor batchNormalise(poplar::program::Sequence &prog,
                                const poplar::Tensor &x,
                                const poplar::Tensor &scale,
                                const poplar::Tensor &b,
                                const poplar::Tensor &mean,
                                const poplar::Tensor &invSd) const;
};

class BatchNormGradOpx : public Opx {
public:
  BatchNormGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor>
  batchNormaliseGrad(poplar::program::Sequence &prog,
                     const poplar::Tensor &x,
                     const poplar::Tensor &scale,
                     const poplar::Tensor &mean,
                     const poplar::Tensor &invSd,
                     const poplar::Tensor &yGrad) const;
};

} // namespace popx
} // namespace poponnx

#endif
