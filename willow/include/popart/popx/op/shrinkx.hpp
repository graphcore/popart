// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SHRINKX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SHRINKX_HPP_

#include <memory>
#include <string>
#include <poplar/Tensor.hpp>
#include <popart/popx/op/elementwisex.hpp>

#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/opx.hpp"

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

class ShrinkComputex : public EwuComputex {
public:
  ShrinkComputex(float lambd, float bias) : lambd_(lambd), bias_(bias) {}

  poplar::Tensor outplace(poplar::program::Sequence &,
                          poplar::Graph &,
                          const poplar::Tensor &,
                          const poplar::DebugNameAndId &,
                          const std::string &) const final;

  void inplace(poplar::program::Sequence &,
               poplar::Graph &,
               const poplar::Tensor &,
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

class ShrinkGradOpx : public Opx {
public:
  ShrinkGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SHRINKX_HPP_
