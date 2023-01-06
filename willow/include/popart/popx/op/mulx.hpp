// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_MULX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_MULX_HPP_

#include <string>
#include <poplar/Tensor.hpp>
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

class MulComputex : public EwbComputex {
public:
  explicit MulComputex(EwbComputex::InplacePolicy ip);

  poplar::Tensor outplace(poplar::program::Sequence &,
                          poplar::Graph &,
                          const poplar::Tensor &,
                          const poplar::Tensor &,
                          const poplar::DebugNameAndId &,
                          const std::string &) const final;

  poplar::Tensor maybeInplace(poplar::program::Sequence &,
                              poplar::Graph &,
                              poplar::Tensor &,
                              poplar::Tensor &,
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
