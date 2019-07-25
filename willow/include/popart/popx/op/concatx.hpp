#ifndef GUARD_NEURALNET_CONCATX_HPP
#define GUARD_NEURALNET_CONCATX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

class ConcatOp;
class ConcatGradOp;

namespace popx {

class ConcatOpx : public Opx {
public:
  ConcatOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  const ConcatOp *const op;
};

class ConcatInplaceOpx : public Opx {
public:
  ConcatInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  const ConcatOp *const op;
};

class ConcatGradOpx : public Opx {
public:
  ConcatGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  const ConcatGradOp *const op;
};

} // namespace popx
} // namespace popart

#endif
