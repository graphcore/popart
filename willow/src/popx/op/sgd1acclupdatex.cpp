#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/Zero.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/sgd1acclupdate.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/sgd1acclupdatex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

SGD1AcclUpdateOpx::SGD1AcclUpdateOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<SGD1AcclUpdateOp>(op, {Onnx::CustomOperators::SGD1AcclUpdate});
}

void SGD1AcclUpdateOpx::grow(poplar::program::Sequence &prog) const {

  auto vu_op = getOp<SGD1AcclUpdateOp>();

  auto mmConst = vu_op.initMm1.isConst();
  auto wdConst = vu_op.initWdsf1.isConst();
  auto wdVal   = vu_op.initWdsf1.val();

  auto toUpdate = getInTensor(VarUpdateOp::getVarToUpdateInIndex());

  if (mmConst) {
    auto mmVal = vu_op.initMm1.val();
    if (mmVal == 0.0f) {
      popops::zero(graph(), toUpdate, prog, debugPrefix("reset"));
    } else {
      popops::mapInPlace(graph(),
                         pe::Mul(pe::_1, pe::Const(mmVal)),
                         {toUpdate},
                         prog,
                         debugPrefix("constMomentumScaling"));
    }
  } else {
    popops::mapInPlace(
        graph(),
        pe::Mul(pe::_1, pe::_2),
        {toUpdate, getInTensor(SGD1AcclUpdateOp::getMm1InIndex())},
        prog,
        debugPrefix("constMomentumScaling"));
  }

  if (wdConst) {
    if (wdVal != 0.0f) {
      popops::scaledAddTo(graph(),
                          toUpdate,
                          getInTensor(VarUpdateOp::getUpdaterInIndex()),
                          wdVal,
                          prog,
                          debugPrefix("constScaledAddWdsf1"));
    }
  } else {
    popops::scaledAddTo(graph(),
                        toUpdate,
                        getInTensor(VarUpdateOp::getUpdaterInIndex()),
                        getInTensor(SGD1AcclUpdateOp::getWdsf1InIndex()),
                        prog,
                        debugPrefix("nonConstScaledAddWdsf1"));
  }

  // return a reference to the input
  setOutTensor(VarUpdateOp::getUpdatedVarOutIndex(), toUpdate);
}

namespace {
OpxCreator<SGD1AcclUpdateOpx>
    ResetAcclOpxCreator({Onnx::CustomOperators::SGD1AcclUpdate});
} // namespace

} // namespace popx
} // namespace popart
