#ifndef GUARD_NEURALNET_VARUPDATEX_HPP
#define GUARD_NEURALNET_VARUPDATEX_HPP

#include <willow/names.hpp>
#include <willow/popx/opx.hpp>

namespace willow {

class VarUpdateOp;

namespace popx {

class VarUpdateOpx : public Opx {
public:
  VarUpdateOpx(Op *, Devicex *);
  VarUpdateOp *getVarUpdateOp() const;
};

} // namespace popx
} // namespace willow

#endif
