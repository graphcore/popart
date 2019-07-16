#ifndef GUARD_NEURALNET_VARUPDATEX_HPP
#define GUARD_NEURALNET_VARUPDATEX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class SGDVarUpdateOp;
class ConstSGDVarUpdateOp;

namespace popx {

class VarUpdateOpx : public Opx {
public:
  VarUpdateOpx(Op *, Devicex *);

  // In the case of gradient accumulation these functions
  // allow input at index1 (varGrad) to be created
  // with the same mapping as the variable it's updating.
  poplar::Tensor createInput(int index, const std::string &name) const override;
  InputCreatorType getInputCreatorType(int index1) const override;
  std::vector<TensorId> mustExistBeforeCreate(int index1) const override;
};

class SGDVarUpdateOpx : public VarUpdateOpx {
public:
  SGDVarUpdateOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class ConstSGDVarUpdateOpx : public VarUpdateOpx {
public:
  ConstSGDVarUpdateOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class CopyVarUpdateOpx : public VarUpdateOpx {
public:
  CopyVarUpdateOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
