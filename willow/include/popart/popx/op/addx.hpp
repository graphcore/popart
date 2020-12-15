// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ADDX_HPP
#define GUARD_NEURALNET_ADDX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>
#include <popart/popx/op/reducesumx.hpp>

namespace popart {
namespace popx {

class AddComputex : public EwbComputex {
public:
  explicit AddComputex(EwbComputex::InplacePolicy ip);

  poplar::Tensor outplace(poplar::program::Sequence &,
                          poplar::Graph &,
                          const poplar::Tensor &,
                          const poplar::Tensor &,
                          const poplar::DebugNameAndId &,
                          const std::string &) const final;

  void inplace(poplar::program::Sequence &,
               poplar::Graph &,
               const poplar::Tensor &,
               const poplar::Tensor &,
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
