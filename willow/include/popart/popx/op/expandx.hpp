#ifndef GUARD_NEURALNET_CONCATX_HPP
#define GUARD_NEURALNET_CONCATX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class BaseExpandOpx : public PopOpx {
protected:
  void expand_broadcast(const Shape output_shape, poplar::Tensor &expand) const;

public:
  BaseExpandOpx(Op *, Devicex *);
  InputCreatorType getInputCreatorType(InIndex) const final;

  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;

  view::RegMap unwindRegion(InIndex, OutIndex) const final;

protected:
  const ExpandOp *const op;
};

class ExpandOpx : public BaseExpandOpx {
public:
  ExpandOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class ExpandInplaceOpx : public BaseExpandOpx {
public:
  ExpandInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class ExpandGradOpx : public PopOpx {
public:
  ExpandGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  std::vector<size_t> xShape;
};

} // namespace popx
} // namespace popart

#endif
