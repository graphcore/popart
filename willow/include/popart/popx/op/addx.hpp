#ifndef GUARD_NEURALNET_ADDX_HPP
#define GUARD_NEURALNET_ADDX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>
#include <popart/popx/op/reducesumx.hpp>

namespace popart {

class AddOp;

namespace popx {

class AddOpx : public ElementWiseBinaryOpx {
public:
  AddOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
  InputCreatorType getInputCreatorType(InIndex) const override;
  poplar::Tensor createInput(InIndex index,
                             const std::string &name) const override;
  std::vector<TensorId> mustExistBeforeCreate(InIndex) const override;
};

class AddLhsInplaceOpx : public AddOpx {
public:
  AddLhsInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class AddRhsInplaceOpx : public AddOpx {
public:
  AddRhsInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
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
