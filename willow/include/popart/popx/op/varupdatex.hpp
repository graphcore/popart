#ifndef GUARD_NEURALNET_VARUPDATEX_HPP
#define GUARD_NEURALNET_VARUPDATEX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

class SGDVarUpdateOp;
class ConstSGDVarUpdateOp;

namespace popx {

class VarUpdateOpx : public Opx {
public:
  VarUpdateOpx(Op *, Devicex *);

};

class SGDVarUpdateOpx : public VarUpdateOpx {
public:
  SGDVarUpdateOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class CopyVarUpdateOpx : public VarUpdateOpx {
public:
  CopyVarUpdateOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
