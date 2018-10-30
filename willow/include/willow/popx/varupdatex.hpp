#ifndef GUARD_NEURALNET_VARUPDATEX_HPP
#define GUARD_NEURALNET_VARUPDATEX_HPP

#include <willow/names.hpp>
#include <willow/popx/opx.hpp>

namespace willow {

class SGDVarUpdateOp;
class ConstSGDVarUpdateOp;

namespace popx {

class SGDVarUpdateOpx : public Opx {
public:
  SGDVarUpdateOpx(Op *, Devicex *);
  SGDVarUpdateOp *getSGDVarUpdateOp() const;
};

class ConstSGDVarUpdateOpx : public Opx {
public:
  ConstSGDVarUpdateOpx(Op *, Devicex *);
  ConstSGDVarUpdateOp *getConstSGDVarUpdateOp() const;
  void grow() const override final;
};

} // namespace popx
} // namespace willow

#endif
