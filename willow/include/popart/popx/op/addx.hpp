// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ADDX_HPP
#define GUARD_NEURALNET_ADDX_HPP

#include "popart/popx/debugcontextx.hpp"
#include <snap/Tensor.hpp>
#include <string>
#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>
#include <popart/popx/op/reducesumx.hpp>

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

class AddComputex : public EwbComputex {
public:
  explicit AddComputex(EwbComputex::InplacePolicy ip);

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

#endif
