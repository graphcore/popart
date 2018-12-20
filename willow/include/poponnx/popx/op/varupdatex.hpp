#ifndef GUARD_NEURALNET_VARUPDATEX_HPP
#define GUARD_NEURALNET_VARUPDATEX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class SGDVarUpdateOp;
class ConstSGDVarUpdateOp;

namespace popx {

class SGDVarUpdateOpx : public Opx {
public:
  SGDVarUpdateOpx(Op *, Devicex *);
};

class ConstSGDVarUpdateOpx : public Opx {
public:
  ConstSGDVarUpdateOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
