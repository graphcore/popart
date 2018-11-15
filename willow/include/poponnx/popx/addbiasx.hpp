#ifndef GUARD_NEURALNET_ADD_BIASX_HPP
#define GUARD_NEURALNET_ADD_BIASX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/identityx.hpp>
#include <poponnx/popx/opx.hpp>
#include <poponnx/popx/reducesumx.hpp>

namespace willow {

class AddBiasOp;
class AddBiasGradOp;

namespace popx {

class AddBiasOpx : public Opx {
public:
  AddBiasOpx(Op *, Devicex *);
  AddBiasOp *getAddBiasOp() const;
  void grow(poplar::program::Sequence &) const final;

  std::vector<TensorId> mustExistBeforeCreate(int index0) const override;
  bool canCreateInput(int index0) const final;
  poplar::Tensor createInput(int index) const final;
  bool createsEquiv(int index0, Opx *opx1, int index1) const final;
};

class AddBiasDataGradOpx : public IdentityOpx {
public:
  AddBiasDataGradOpx(Op *, Devicex *);
};

class AddBiasBiasGradOpx : public ReduceSumOpx {
public:
  AddBiasBiasGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace willow

#endif
