// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_POWX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_POWX_HPP_

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

class PowComputex : public EwbComputex {
public:
  explicit PowComputex(EwbComputex::InplacePolicy ip);

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

class PowOpx : public ElementWiseBinaryOutplaceOpx {
public:
  PowOpx(Op *, Devicex *);
};

class PowLhsInplaceOpx : public ElementWiseBinaryInplaceOpx {
public:
  PowLhsInplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_POWX_HPP_
