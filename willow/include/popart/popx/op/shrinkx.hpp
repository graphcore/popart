// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SHRINKX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SHRINKX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <memory>
#include <snap/Tensor.hpp>
#include <string>
#include <popart/popx/op/elementwisex.hpp>

#include "popart/popx/popopx.hpp"

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

class ShrinkComputex : public EwuComputex {
public:
  ShrinkComputex(float lambd, float bias) : lambd_(lambd), bias_(bias) {}

  snap::Tensor outplace(snap::program::Sequence &,
                        snap::Graph &,
                        const snap::Tensor &,
                        const poplar::DebugNameAndId &,
                        const std::string &) const final;

  void inplace(snap::program::Sequence &,
               snap::Graph &,
               const snap::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get(float lambd, float bias) {
    return std::unique_ptr<EwuComputex>(new ShrinkComputex(
        static_cast<float>(lambd), static_cast<float>(bias)));
  }

  float lambd() const { return lambd_; }
  float bias() const { return bias_; }

private:
  float lambd_;
  float bias_;
};

class ShrinkOpx : public ElementWiseUnaryOutplaceOpx {
public:
  ShrinkOpx(Op *, Devicex *);
};

class ShrinkInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  ShrinkInplaceOpx(Op *, Devicex *);
};

class ShrinkGradOpx : public PopOpx {
public:
  ShrinkGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SHRINKX_HPP_
