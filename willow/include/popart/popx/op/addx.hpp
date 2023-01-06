// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_ADDX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_ADDX_HPP_

#include <string>
#include <poplar/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>
#include <popart/popx/op/reducesumx.hpp>

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

class AddComputex : public EwbComputex {
public:
  explicit AddComputex(EwbComputex::InplacePolicy ip);

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
                              const std::string &) const;
};

class AddOpx : public ElementWiseBinaryOutplaceOpx {
public:
  AddOpx(Op *, Devicex *);
  InputCreatorType getInputCreatorType(InIndex) const override;
};

class AddLhsInplaceOpx : public ElementWiseBinaryInplaceOpx {
public:
  AddLhsInplaceOpx(Op *, Devicex *);
};

class AddRhsInplaceOpx : public ElementWiseBinaryInplaceOpx {
public:
  AddRhsInplaceOpx(Op *, Devicex *);
};

class AddArg0GradOpx : public ReduceSumOpx {
public:
  AddArg0GradOpx(Op *, Devicex *);
};

class AddArg1GradOpx : public ReduceSumOpx {
public:
  AddArg1GradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_ADDX_HPP_
