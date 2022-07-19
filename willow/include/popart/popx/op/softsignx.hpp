// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SOFTSIGNX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SOFTSIGNX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <string>
#include <popart/popx/op/elementwisex.hpp>

#include "popart/popx/popopx.hpp"

namespace snap {
class Graph;
class Tensor;

namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class SoftSignComputex : public EwuComputex {
public:
  SoftSignComputex() {}

  void inplace(snap::program::Sequence &,
               snap::Graph &,
               const snap::Tensor &,
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

class SoftSignGradOpx : public PopOpx {
public:
  SoftSignGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SOFTSIGNX_HPP_
