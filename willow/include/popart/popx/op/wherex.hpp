// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_WHEREX_HPP
#define GUARD_NEURALNET_WHEREX_HPP

#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class BaseWhereOpx : public PopOpx {
public:
  BaseWhereOpx(Op *, Devicex *);

  void grow(snap::program::Sequence &) const final;

  InputCreatorType getInputCreatorType(InIndex inIndex) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const final;
  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const override;
  snap::Tensor unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;

private:
  virtual void doGrow(snap::program::Sequence &,
                      const snap::Tensor &,
                      const snap::Tensor &,
                      const snap::Tensor &) const = 0;

  // Always unwind on one index. Favour the order x, y, condition
  InIndex unwindIndex() const;
};

class WhereOpx : public BaseWhereOpx {
public:
  WhereOpx(Op *, Devicex *);
  void doGrow(snap::program::Sequence &prog,
              const snap::Tensor &,
              const snap::Tensor &,
              const snap::Tensor &) const final;
};

class WhereLhsInplaceOpx : public BaseWhereOpx {
public:
  WhereLhsInplaceOpx(Op *, Devicex *);
  void doGrow(snap::program::Sequence &,
              const snap::Tensor &,
              const snap::Tensor &,
              const snap::Tensor &) const final;
};

class WhereRhsInplaceOpx : public BaseWhereOpx {
public:
  WhereRhsInplaceOpx(Op *, Devicex *);
  void doGrow(snap::program::Sequence &,
              const snap::Tensor &,
              const snap::Tensor &,
              const snap::Tensor &) const final;
};

class WhereXGradOpx : public PopOpx {
public:
  WhereXGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class WhereYGradOpx : public PopOpx {
public:
  WhereYGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
