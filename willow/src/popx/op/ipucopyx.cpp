// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <poprithms/logging/timepartitionlogger.hpp>
#include <popart/error.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/ipucopyx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

#include <poputil/TileMapping.hpp>

namespace popart {
namespace popx {

IpuCopyOpx::IpuCopyOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<IpuCopyOp>(op, Onnx::CustomOperators::IpuCopy);
}

void IpuCopyOpx::grow(poplar::program::Sequence &prog) const {

  IpuCopyOp &op = getOp<IpuCopyOp>();

  logging::devicex::trace(
      "Adding copyToIpu for {}, {}", op.str(), op.getFromToStr());

  for (auto &idx_tensor : op.input->tensorMap()) {
    auto idx = idx_tensor.first;
    // Need to get the non virtual graph, so cannot use PopOpx::graph()
    auto t = poputil::copyToIpu(dv_p->lowering().graph().getPoplarGraph(),
                                getInTensor(idx),
                                prog,
                                static_cast<int>(op.getDestIpu()),
                                debugContext("ipuCopy"));
    setOutTensor(idx, t);
  }
}

void IpuCopyOpx::createPipelinedOutput() const {

  const auto growTimeTracker =
      op_p->getIr().timePartitionLogger().scopedStopwatch(
          "Creating IpuCopy pipeline output (Ir Lowering)");

  IpuCopyOp &op = getOp<IpuCopyOp>();

  logging::devicex::trace(
      "Creating destination tensors for {}, {}", op.str(), op.getFromToStr());

  for (auto &idx_tensor : op.input->tensorMap()) {
    auto idx = idx_tensor.first;

    // When pipelining, create the copy destination, but dont add the copy
    // program.
    poplar::Tensor tLocalForCopy, tForCopy;
    auto t = poputil::createIpuCopy(dv_p->lowering().graph().getPoplarGraph(),
                                    getInTensor(idx),
                                    static_cast<int>(op.getDestIpu()),
                                    tForCopy,
                                    tLocalForCopy,
                                    debugContext("createOutput"));
    setOutTensor(idx, t);
  }
}

void IpuCopyOpx::growPipelined(poplar::program::Sequence &prog) const {
  IpuCopyOp &op = getOp<IpuCopyOp>();

  for (auto &idx_tensor : op.input->tensorMap()) {
    auto idx   = idx_tensor.first;
    auto outId = op_p->outId(idx);

    auto &source      = getInTensor(idx);
    auto &destination = dv_p->lowering().tensors().get(outId);

    // Using dontOutline=false will ensure the copies (buffers & code) are
    // reused.
    prog.add(poplar::program::Copy(source, destination, false, debugContext()));
  }
}

poplar::Tensor IpuCopyOpx::unwindTensorLayout(poplar::Tensor tensor,
                                              InIndex in,
                                              OutIndex out) const {
  IpuCopyOp &op = getOp<IpuCopyOp>();
  auto srcIpu   = op.getSourceIpu(op.input->tensor(in)->id);

  poplar::Tensor tLocalForCopy, tForCopy;
  auto t = poputil::createIpuCopy(dv_p->lowering().graph().getPoplarGraph(),
                                  tensor,
                                  static_cast<int>(srcIpu),
                                  tForCopy,
                                  tLocalForCopy,
                                  debugContext("unwoundInput"));
  return t;
}

view::RegMap IpuCopyOpx::unwindRegion(InIndex, OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

snap::Graph &IpuCopyOpx::srcVirtualGraph(InIndex in) const {
  if (op_p->getIr().virtualGraphsEnabled()) {
    IpuCopyOp &op = getOp<IpuCopyOp>();
    auto srcIpu   = op.getSourceIpu(op.input->tensor(in)->id);
    return dv_p->lowering().getVirtualGraph(srcIpu, op_p->settings.tileSet);
  } else {
    throw error("IpuCopyOpx unexpected on model without virtual graphs");
  }
}

snap::Graph &IpuCopyOpx::dstVirtualGraph(OutIndex out) const {
  if (op_p->getIr().virtualGraphsEnabled()) {
    IpuCopyOp &op = getOp<IpuCopyOp>();
    auto dstIpu   = op.getDestIpu();
    return dv_p->lowering().getVirtualGraph(dstIpu, op_p->settings.tileSet);
  } else {
    throw error("IpuCopyOpx unexpected on model without virtual graphs");
  }
}

namespace {
OpxCreator<IpuCopyOpx> ipuCopyOpxCreator(Onnx::CustomOperators::IpuCopy);
} // namespace

} // namespace popx
} // namespace popart
