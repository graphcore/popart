#ifndef GUARD_NEURALNET_ADD_BIASX_HPP
#define GUARD_NEURALNET_ADD_BIASX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/identityx.hpp>
#include <poponnx/popx/op/reducesumx.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class AddBiasOp;
class AddBiasGradOp;

namespace popx {

class AddBiasOpx : public Opx {
public:
  AddBiasOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  std::vector<TensorId> mustExistBeforeCreate(int index0) const override;
  InputCreatorType getInputCreatorType(InIndex) const final;
  poplar::Tensor createInput(InIndex index,
                             const std::string &name) const final;
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
} // namespace poponnx

#endif
