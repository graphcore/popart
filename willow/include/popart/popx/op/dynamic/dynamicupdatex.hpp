#ifndef GUARD_NEURALNET_DYNAMICUPDATEX_HPP
#define GUARD_NEURALNET_DYNAMICUPDATEX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class DynamicUpdateOpx : public Opx {
public:
  DynamicUpdateOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
  InputCreatorType getInputCreatorType(InIndex index) const override;
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
  poplar::Tensor createInput(InIndex index,
                             const std::string &name) const final;
  std::vector<TensorId> mustExistBeforeCreate(InIndex) const final;
  virtual poplar::Tensor cloneNcopyOpt(poplar::program::Sequence &,
                                       const poplar::Tensor &) const;
};

class DynamicUpdateInplaceOpx : public DynamicUpdateOpx {
public:
  DynamicUpdateInplaceOpx(Op *, Devicex *);
  poplar::Tensor cloneNcopyOpt(poplar::program::Sequence &,
                               const poplar::Tensor &) const override;
};

} // namespace popx
} // namespace popart

#endif
