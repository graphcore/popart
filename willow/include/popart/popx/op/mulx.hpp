// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_MULX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_MULX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <snap/Tensor.hpp>
#include <string>
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

class MulComputex : public EwbComputex {
public:
  explicit MulComputex(EwbComputex::InplacePolicy ip);

  snap::Tensor outplace(snap::program::Sequence &,
                        snap::Graph &,
                        const snap::Tensor &,
                        const snap::Tensor &,
                        const poplar::DebugNameAndId &,
                        const std::string &) const final;

  snap::Tensor maybeInplace(snap::program::Sequence &,
                            snap::Graph &,
                            const snap::Tensor &,
                            const snap::Tensor &,
                            const poplar::DebugNameAndId &,
                            const std::string &) const final;
};

class MulOpx : public ElementWiseBinaryOutplaceOpx {
public:
  MulOpx(Op *, Devicex *);
};

class MulLhsInplaceOpx : public ElementWiseBinaryInplaceOpx {
public:
  MulLhsInplaceOpx(Op *, Devicex *);
};

class MulRhsInplaceOpx : public ElementWiseBinaryInplaceOpx {
public:
  MulRhsInplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_MULX_HPP_
