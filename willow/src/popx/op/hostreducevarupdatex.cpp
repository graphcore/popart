#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/hostreducevarupdate.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/hostreducevarupdatex.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/Collectives.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

GradCopyToHostOpx::GradCopyToHostOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<GradCopyToHostOp>(op, Onnx::CustomOperators::GradCopyToHost);
}

void GradCopyToHostOpx::grow(poplar::program::Sequence &prog) const {
  const auto updater_index    = GradCopyToHostOp::getInIndex();
  poplar::Tensor weightDeltas = getInTensor(updater_index);

  const auto grad_id = inId(updater_index);
  auto deviceToHostStream =
      dv_p->insertGradientStoreStream(grad_id, inInfo(updater_index), graph());

  dv_p->getHostReduceStreamIds().emplace_back(deviceToHostStream.handle());

  // TODO(T12685): Once replicatedReduceScatter is part of the Poplar
  // public API we can replace the replicatedAllReduce with it and
  // then do the AllGather on the host.
  // If accumulation is enabled then the replicatedAllReduce is run from
  // SGD1AcclReduceOp
  if (dv_p->getReplicationFactor() > 1 && dv_p->getAccumulationFactor() == 1) {
    weightDeltas =
        popops::replicatedAllReduce(graph(),
                                    weightDeltas,
                                    popops::Operation::ADD,
                                    prog,
                                    debugPrefix("allReduce_Add"),
                                    {{"useReplicatedImplementation", "true"}});
  }

  poplar::program::Copy gradientsToHostProg(weightDeltas, deviceToHostStream);
  prog.add(gradientsToHostProg);
}

GradCopyFromHostOpx::GradCopyFromHostOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<GradCopyFromHostOp>(op, Onnx::CustomOperators::GradCopyFromHost);
}

void GradCopyFromHostOpx::grow(poplar::program::Sequence &prog) const {
  const auto updater_index    = GradCopyFromHostOp::getInIndex();
  poplar::Tensor weightDeltas = getInTensor(updater_index);

  const auto grad_id = inId(updater_index);
  auto hostToDeviceStream =
      dv_p->insertGradientLoadStream(grad_id, inInfo(updater_index), graph());

  dv_p->getHostReduceStreamIds().emplace_back(hostToDeviceStream.handle());

  poplar::program::Copy gradientsFromHostProg(hostToDeviceStream, weightDeltas);
  prog.add(gradientsFromHostProg);

  // output is a reference to the updated input
  setOutTensor(GradCopyFromHostOp::getOutIndex(), weightDeltas);
}

HostReduceVarCopyOpx::HostReduceVarCopyOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<HostSGD0VarUpdate>(op, Onnx::CustomOperators::HostSGD0VarUpdate);
}

void HostReduceVarCopyOpx::grow(poplar::program::Sequence &prog) const {
  const auto var_update_index = HostSGD0VarUpdate::getVarToUpdateInIndex();
  const auto updater_index    = HostSGD0VarUpdate::getUpdaterInIndex();
  const auto grad_id          = inId(updater_index);
  poplar::Tensor weights      = getInTensor(var_update_index);

  const auto weight_id    = inId(var_update_index);
  auto hostToDeviceStream = dv_p->insertWeightLoadStream(
      weight_id, inInfo(var_update_index), graph());

  dv_p->getHostReduceStreamIds().emplace_back(hostToDeviceStream.handle());

  poplar::program::Copy hostWeightsToDeviceProg(hostToDeviceStream, weights);
  prog.add(hostWeightsToDeviceProg);

  // output is a reference to the updated input
  setOutTensor(HostSGD0VarUpdate::getUpdatedVarOutIndex(),
               getInTensor(var_update_index));
}

namespace {
OpxCreator<GradCopyToHostOpx>
    GradCopyToHostOpxCreator(Onnx::CustomOperators::GradCopyToHost);

OpxCreator<GradCopyFromHostOpx>
    GradCopyFromHostOpxCreator(Onnx::CustomOperators::GradCopyFromHost);

OpxCreator<HostReduceVarCopyOpx>
    HostReduceVarCopyOpxCreator(Onnx::CustomOperators::HostSGD0VarUpdate);

} // namespace

} // namespace popx
} // namespace popart
