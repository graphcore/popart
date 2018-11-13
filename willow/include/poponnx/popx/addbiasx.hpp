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
  void grow() const override final;
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
