// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SCALEX_HPP
#define GUARD_NEURALNET_SCALEX_HPP

#include "popart/popx/debugcontextx.hpp"
#include <memory>
#include <snap/Tensor.hpp>
#include <string>
#include <poplar/Type.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace snap {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class ScaleComputex : public EwuComputex {

public:
  ScaleComputex(double x) : scale_factor(x) {}

  snap::Tensor outplace(snap::program::Sequence &,
                        snap::Graph &,
                        const snap::Tensor &tensor,
                        const poplar::DebugNameAndId &,
                        const std::string &) const final;

  void inplace(snap::program::Sequence &,
               snap::Graph &,
               const snap::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get(float x) {
    return std::unique_ptr<EwuComputex>(
        new ScaleComputex(static_cast<double>(x)));
  }

  snap::Tensor getScaleTensor(const poplar::Type &type,
                              snap::Graph &graph) const;

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

#endif
