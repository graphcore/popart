#include <popops/Collectives.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/sgd1varupdate.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/sgd1varupdatex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

SGD1VarUpdateOpx::SGD1VarUpdateOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<SGD1VarUpdateOp>(op, Onnx::CustomOperators::SGD1VarUpdate);
}

// in this step, w <- w - { lr / vs } * v.
// { lr / vs } is the scaled learning rate.
//
void SGD1VarUpdateOpx::grow(poplar::program::Sequence &prog) const {

  auto sgd1varUpdateOp = getOp<SGD1VarUpdateOp>();
  auto scaledLR        = sgd1varUpdateOp.initSlr1.val();
  auto scaledLRisConst = sgd1varUpdateOp.initSlr1.isConst();

  poplar::Tensor velocity = getInTensor(SGD1VarUpdateOp::getUpdaterInIndex());
  if (dv_p->getReplicationFactor() > 1) {
    velocity =
        popops::replicatedAllReduce(graph(),
                                    velocity,
                                    popops::Operation::ADD,
                                    prog,
                                    debugPrefix("allReduceVelocitySGD1"),
                                    {{"useReplicatedImplementation", "true"}});

    // T12001 : when there is replication and momentum, we'll need an
    // additional scaling here probably.

    //   popops::mapInPlace(
    //       graph(),
    //       popops::expr::BinaryOpType::MULTIPLY,
    //       velocity,
    //       getConst(velocity.elementType(),
    //                {},
    //                1.0 / static_cast<double>(dv_p->getReplicationFactor()),
    //                debugPrefix("oneOverReplFactor")),
    //       prog,
    //       "momentumReplCorrectionVarUpdate");
    // }
  }

  // non-const scaled learning rate case
  if (!scaledLRisConst) {
    popops::scaledSubtractFrom(
        graph(),
        getInTensor(VarUpdateOp::getVarToUpdateInIndex()), // weights
        velocity,
        getInTensor(SGD1VarUpdateOp::getSlr1InIndex()),
        prog,
        debugPrefix("nonConstScaledSubtractSGD1"));
  }

  // const scaled learning rate case
  else {
    popops::scaledSubtractFrom(
        graph(),
        getInTensor(sgd1varUpdateOp.getVarToUpdateInIndex()), // weights
        velocity,
        scaledLR,
        prog,
        debugPrefix("constScaledSubtractSGD1"));
  }
}

namespace {
OpxCreator<SGD1VarUpdateOpx>
    sgd1VarUpdateOpxCreator(Onnx::CustomOperators::SGD1VarUpdate);
}
} // namespace popx
} // namespace popart
