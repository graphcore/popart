// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SOFTSIGNX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SOFTSIGNX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <string>
#include <popart/popx/op/elementwisex.hpp>

#include "popart/popx/opx.hpp"

namespace poplar {
class Graph;
class Tensor;

namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class SoftSignComputex : public EwuComputex {
public:
  SoftSignComputex() {}

  void inplace(poplar::program::Sequence &,
               poplar::Graph &,
               const poplar::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;
};

class SoftSignOpx : public ElementWiseUnaryOutplaceOpx {
public:
  SoftSignOpx(Op *, Devicex *);
};

class SoftSignInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  SoftSignInplaceOpx(Op *, Devicex *);
};

class SoftSignGradOpx : public Opx {
public:
  SoftSignGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SOFTSIGNX_HPP_
