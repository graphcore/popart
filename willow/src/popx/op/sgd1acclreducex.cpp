#include <popops/Collectives.hpp>
#include <popart/error.hpp>
#include <popart/op/sgd1acclreduce.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/sgd1acclreducex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

SGD1AcclReduceOpx::SGD1AcclReduceOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<SGD1AcclReduceOp>(op, {Onnx::CustomOperators::SGD1AcclReduce});
}

void SGD1AcclReduceOpx::grow(poplar::program::Sequence &prog) const {

  const poplar::Tensor &velocity =
      getInTensor(VarUpdateOp::getVarToUpdateInIndex());

  if (dv_p->getReplicationFactor() > 1) {
    poplar::Tensor velocityReduced =
        popops::replicatedAllReduce(graph(),
                                    velocity,
                                    popops::Operation::ADD,
                                    prog,
                                    debugPrefix("allReduceVelocitySGD1"),
                                    {{"useReplicatedImplementation", "true"}});

    poplar::program::Copy copy(velocityReduced, velocity);
    prog.add(copy);
  }

  else {
    throw internal_error(
        "SGD1AcclReduceOp should not be present as replication "
        "factor is 1");
  }

  // reference accl/velocity returned
  setOutTensor(VarUpdateOp::getUpdatedVarOutIndex(), velocity);
}

namespace {
OpxCreator<SGD1AcclReduceOpx>
    SGD1AcclReduceOpxCreator({Onnx::CustomOperators::SGD1AcclReduce});
}

} // namespace popx
} // namespace popart
