// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_WHEREX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_WHEREX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <set>
#include <snap/Tensor.hpp>
#include <popart/popx/popopx.hpp>

#include "popart/names.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

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

protected:
  // Use for both the outplace variant and as a fallback for the inplace
  // variants
  void outplace(snap::program::Sequence &prog,
                const snap::Tensor &x,
                const snap::Tensor &y,
                const snap::Tensor &condition) const;

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

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_WHEREX_HPP_
