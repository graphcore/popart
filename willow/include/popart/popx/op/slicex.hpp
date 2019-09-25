#ifndef GUARD_NEURALNET_SLICEX_HPP
#define GUARD_NEURALNET_SLICEX_HPP

#include <popart/popx/opx.hpp>

namespace popart {

class SliceOp;
class SliceInplaceOp;

namespace popx {

class BaseSliceOpx : public Opx {
public:
  BaseSliceOpx(Op *, Devicex *);

  InputCreatorType getInputCreatorType(InIndex) const final;

  poplar::Tensor unwindTensorLayout(std::vector<poplar::Tensor> tensors,
                                    InIndex inIndex,
                                    OutIndex outIndex) const final;

  std::vector<std::pair<Op *, InIndex>>
      getCreatorCandicates(InIndex) const final;
};

class SliceOpx : public BaseSliceOpx {
public:
  SliceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  SliceOp *getSliceOp() const;
};

class SliceInplaceOpx : public BaseSliceOpx {
public:
  SliceInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  SliceInplaceOp *getSliceInplaceOp() const;
};

class SliceGradOpx : public Opx {
public:
  SliceGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  bool canFindPreSlicedTensor() const;
  std::pair<bool, poplar::Tensor> getPreSlicedTensorIfPossible() const;
};

} // namespace popx
} // namespace popart

#endif
