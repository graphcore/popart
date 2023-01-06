// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_WHEREX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_WHEREX_HPP_

#include <set>
#include <poplar/Tensor.hpp>
#include <popart/popx/opx.hpp>

#include "popart/names.hpp"
#include "popart/popx/debugcontextx.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class BaseWhereOpx : public Opx {
public:
  BaseWhereOpx(Op *, Devicex *);

  void grow(poplar::program::Sequence &) const final;

  InputCreatorType getInputCreatorType(InIndex inIndex) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const final;
  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const override;
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;

protected:
  // Use for both the outplace variant and as a fallback for the inplace
  // variants
  void outplace(poplar::program::Sequence &prog,
                const poplar::Tensor &x,
                const poplar::Tensor &y,
                const poplar::Tensor &condition) const;

private:
  virtual void doGrow(poplar::program::Sequence &,
                      const poplar::Tensor &,
                      const poplar::Tensor &,
                      const poplar::Tensor &) const = 0;

  // Always unwind on one index. Favour the order x, y, condition
  InIndex unwindIndex() const;
};

class WhereOpx : public BaseWhereOpx {
public:
  WhereOpx(Op *, Devicex *);
  void doGrow(poplar::program::Sequence &prog,
              const poplar::Tensor &,
              const poplar::Tensor &,
              const poplar::Tensor &) const final;
};

class WhereLhsInplaceOpx : public BaseWhereOpx {
public:
  WhereLhsInplaceOpx(Op *, Devicex *);
  void doGrow(poplar::program::Sequence &,
              const poplar::Tensor &,
              const poplar::Tensor &,
              const poplar::Tensor &) const final;
};

class WhereRhsInplaceOpx : public BaseWhereOpx {
public:
  WhereRhsInplaceOpx(Op *, Devicex *);
  void doGrow(poplar::program::Sequence &,
              const poplar::Tensor &,
              const poplar::Tensor &,
              const poplar::Tensor &) const final;
};

class WhereXGradOpx : public Opx {
public:
  WhereXGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class WhereYGradOpx : public Opx {
public:
  WhereYGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_WHEREX_HPP_
