// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/collectives/replicatedallreducex.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/Collectives.hpp>

namespace popart {
namespace popx {

ReplicatedAllReduceOpx::ReplicatedAllReduceOpx(Op *op, Devicex *devicex)
    : CollectivesBaseOpx(op, devicex) {
  verifyOp<CollectivesBaseOp>(op);
}

void ReplicatedAllReduceOpx::grow(poplar::program::Sequence &prog) const {
  const auto inIndex                   = ReplicatedAllReduceOp::getInIndex();
  poplar::Tensor toReduce              = getInTensor(inIndex);
  poplar::OptionFlags allReduceOptions = dv_p->gclOptions;
  allReduceOptions.set("useReplicatedImplementation", "true");
  poplar::Tensor output =
      popops::replicatedAllReduce(graph(),
                                  toReduce,
                                  popops::Operation::ADD,
                                  prog,
                                  debugPrefix("replicatedAllReduce"),
                                  allReduceOptions);
  setOutTensor(ReplicatedAllReduceOp::getOutIndex(), output);
}

InputCreatorType ReplicatedAllReduceOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CanUnwind;
}

poplar::Tensor ReplicatedAllReduceOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                          InIndex,
                                                          OutIndex) const {
  return tensor;
}

view::RegMap ReplicatedAllReduceOpx::unwindRegion(InIndex, OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

ReplicatedAllReduceInplaceOpx::ReplicatedAllReduceInplaceOpx(Op *op,
                                                             Devicex *devicex)
    : ReplicatedAllReduceOpx(op, devicex) {
  verifyOp<ReplicatedAllReduceInplaceOp>(
      op, Onnx::CustomOperators::ReplicatedAllReduceInplace);
}

void ReplicatedAllReduceInplaceOpx::grow(
    poplar::program::Sequence &prog) const {
  const auto inIndex      = ReplicatedAllReduceInplaceOp::getInIndex();
  poplar::Tensor toReduce = getInTensor(inIndex);
  poplar::OptionFlags allReduceOptions = dv_p->gclOptions;
  allReduceOptions.set("useReplicatedImplementation", "true");
  popops::replicatedAllReduceInPlace(graph(),
                                     toReduce,
                                     popops::Operation::ADD,
                                     prog,
                                     debugPrefix("replicatedAllReduce"),
                                     allReduceOptions);
  setOutTensor(ReplicatedAllReduceInplaceOp::getOutIndex(), toReduce);
}

namespace {
OpxCreator<ReplicatedAllReduceInplaceOpx> ReplicatedAllReduceInplaceOpxCreator(
    Onnx::CustomOperators::ReplicatedAllReduceInplace);

OpxCreator<ReplicatedAllReduceOpx>
    ReplicatedAllReduceOpxCreator(Onnx::CustomOperators::ReplicatedAllReduce);

} // namespace

} // namespace popx
} // namespace popart
