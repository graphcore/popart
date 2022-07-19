// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_BATCHNORMX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_BATCHNORMX_HPP_

#include <snap/Tensor.hpp>
#include <tuple>
#include <popart/popx/op/normx.hpp>
#include <popart/vendored/optional.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {

class BatchNormOp;
class BatchNormGradOp;
class Op;

namespace popx {
class Devicex;

class BatchNormOpx : public NormOpx {
public:
  BatchNormOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

private:
  // Output type for growSpatial.
  struct GrowSpatialOutput {
    snap::Tensor y;
    nonstd::optional<snap::Tensor> mean;
    nonstd::optional<snap::Tensor> var;
    nonstd::optional<snap::Tensor> savedMean;
    nonstd::optional<snap::Tensor> savedVar;
  };

  snap::Tensor batchNormalise(snap::program::Sequence &prog,
                              const snap::Tensor &x,
                              const snap::Tensor &scale,
                              const snap::Tensor &b,
                              const snap::Tensor &mean,
                              const snap::Tensor &invSd) const;

  GrowSpatialOutput growSpatial(snap::program::Sequence &prog,
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
  void grow(snap::program::Sequence &) const final;

private:
  // Output type for growSpatial.
  struct GrowSpatialOutput {
    snap::Tensor xGrad;
    snap::Tensor scaleGrad;
    snap::Tensor bGrad;
  };

  std::tuple<snap::Tensor, snap::Tensor, snap::Tensor>
  batchNormaliseGrad(snap::program::Sequence &prog,
                     const snap::Tensor &x,
                     const snap::Tensor &scale,
                     const snap::Tensor &mean,
                     const snap::Tensor &invSd,
                     const snap::Tensor &yGrad) const;

  GrowSpatialOutput growSpatial(snap::program::Sequence &prog,
                                BatchNormGradOp &op,
                                snap::Tensor &x,
                                snap::Tensor &scale,
                                snap::Tensor &mean,
                                snap::Tensor &var,
                                snap::Tensor &yGrad) const;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_BATCHNORMX_HPP_
