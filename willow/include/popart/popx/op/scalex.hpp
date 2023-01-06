// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCALEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCALEX_HPP_

#include <memory>
#include <string>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popart/popx/op/elementwisex.hpp>

#include "popart/popx/debugcontextx.hpp"

namespace poplar {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class ScaleComputex : public EwuComputex {

public:
  ScaleComputex(double x) : scale_factor(x) {}

  poplar::Tensor outplace(poplar::program::Sequence &,
                          poplar::Graph &,
                          const poplar::Tensor &tensor,
                          const poplar::DebugNameAndId &,
                          const std::string &) const final;

  void inplace(poplar::program::Sequence &,
               poplar::Graph &,
               const poplar::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get(float x) {
    return std::unique_ptr<EwuComputex>(
        new ScaleComputex(static_cast<double>(x)));
  }

  poplar::Tensor getScaleTensor(const poplar::Type &type,
                                poplar::Graph &graph) const;

  static float getFromScaleOp(Op *op);
  static float getFromScaleInplaceOp(Op *op);

private:
  double scale_factor;
};

class ScaleOpx : public ElementWiseUnaryOutplaceOpx {
public:
  ScaleOpx(Op *, Devicex *);
};

class ScaleInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  ScaleInplaceOpx(Op *, Devicex *);
};

class ScaleGradOpx : public ScaleOpx {
public:
  ScaleGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCALEX_HPP_
