// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/collectives.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/collectivesx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/Collectives.hpp>

namespace popart {
namespace popx {

ReplicatedAllReduceInplaceOpx::ReplicatedAllReduceInplaceOpx(Op *op,
                                                             Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<ReplicatedAllReduceInplaceOp>(
      op, Onnx::CustomOperators::ReplicatedAllReduceInplace);
}

void ReplicatedAllReduceInplaceOpx::grow(
    poplar::program::Sequence &prog) const {
  const auto inIndex      = ReplicatedAllReduceInplaceOp::getInIndex();
  poplar::Tensor toReduce = getInTensor(inIndex);
  popops::replicatedAllReduceWithOutput(
      graph(),
      toReduce,
      toReduce,
      popops::Operation::ADD,
      prog,
      "",
      {{"useReplicatedImplementation", "true"}});
  setOutTensor(ReplicatedAllReduceInplaceOp::getOutIndex(), toReduce);
}

ReplicatedAllReduceOpx::ReplicatedAllReduceOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<ReplicatedAllReduceOp>(op,
                                  Onnx::CustomOperators::ReplicatedAllReduce);
}

void ReplicatedAllReduceOpx::grow(poplar::program::Sequence &prog) const {
  const auto inIndex      = ReplicatedAllReduceOp::getInIndex();
  poplar::Tensor toReduce = getInTensor(inIndex);
  poplar::Tensor output =
      popops::replicatedAllReduce(graph(),
                                  toReduce,
                                  popops::Operation::ADD,
                                  prog,
                                  "",
                                  {{"useReplicatedImplementation", "true"}});
  setOutTensor(ReplicatedAllReduceOp::getOutIndex(), output);
}

ReplicatedReduceScatterOpx::ReplicatedReduceScatterOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<ReplicatedReduceScatterOp>(
      op, Onnx::CustomOperators::ReplicatedReduceScatter);
}

void ReplicatedReduceScatterOpx::grow(poplar::program::Sequence &prog) const {
  const auto inIndex             = ReplicatedReduceScatterOp::getInIndex();
  poplar::Tensor toReduceScatter = getInTensor(inIndex);

  poplar::Tensor reducedScattered =
      popops::replicatedReduceScatter(graph(),
                                      toReduceScatter,
                                      popops::Operation::ADD,
                                      prog,
                                      "",
                                      poplar::OptionFlags{});

  setOutTensor(ReplicatedReduceScatterOp::getOutIndex(), reducedScattered);
}

namespace {
OpxCreator<ReplicatedAllReduceInplaceOpx> ReplicatedAllReduceInplaceOpxCreator(
    Onnx::CustomOperators::ReplicatedAllReduceInplace);

OpxCreator<ReplicatedAllReduceOpx>
    ReplicatedAllReduceOpxCreator(Onnx::CustomOperators::ReplicatedAllReduce);

OpxCreator<ReplicatedReduceScatterOpx> ReplicatedReduceScatterOpxCreator(
    Onnx::CustomOperators::ReplicatedReduceScatter);

} // namespace

} // namespace popx
} // namespace popart
