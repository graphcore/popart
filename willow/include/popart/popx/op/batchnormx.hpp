// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_BATCHNORMX_HPP
#define GUARD_NEURALNET_BATCHNORMX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/normx.hpp>

#include <popart/vendored/optional.hpp>

namespace popart {

class BatchNormOp;
class BatchNormGradOp;

namespace popx {

class BatchNormOpx : public NormOpx {
public:
  BatchNormOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  // Output type for growSpatial.
  struct GrowSpatialOutput {
    snap::Tensor y;
    nonstd::optional<snap::Tensor> mean;
    nonstd::optional<snap::Tensor> var;
    nonstd::optional<snap::Tensor> savedMean;
    nonstd::optional<snap::Tensor> savedVar;
  };

  snap::Tensor batchNormalise(poplar::program::Sequence &prog,
                              const snap::Tensor &x,
                              const snap::Tensor &scale,
                              const snap::Tensor &b,
                              const snap::Tensor &mean,
                              const snap::Tensor &invSd) const;

  GrowSpatialOutput growSpatial(poplar::program::Sequence &prog,
                                BatchNormOp &op,
                                snap::Tensor &x,
                                snap::Tensor &scale,
                                snap::Tensor &b,
                                snap::Tensor &mean,
                                snap::Tensor &var) const;
};

class BatchNormGradOpx : public NormOpx {
public:
  BatchNormGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  // Output type for growSpatial.
  struct GrowSpatialOutput {
    snap::Tensor xGrad;
    snap::Tensor scaleGrad;
    snap::Tensor bGrad;
  };

  std::tuple<snap::Tensor, snap::Tensor, snap::Tensor>
  batchNormaliseGrad(poplar::program::Sequence &prog,
                     const snap::Tensor &x,
                     const snap::Tensor &scale,
                     const snap::Tensor &mean,
                     const snap::Tensor &invSd,
                     const snap::Tensor &yGrad) const;

  GrowSpatialOutput growSpatial(poplar::program::Sequence &prog,
                                BatchNormGradOp &op,
                                snap::Tensor &x,
                                snap::Tensor &scale,
                                snap::Tensor &mean,
                                snap::Tensor &var,
                                snap::Tensor &yGrad) const;
};

} // namespace popx
} // namespace popart

#endif
