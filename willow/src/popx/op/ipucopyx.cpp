#include <popart/error.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/ipucopyx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

#include <poputil/TileMapping.hpp>

namespace popart {
namespace popx {

IpuCopyOpx::IpuCopyOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<IpuCopyOp>(op, Onnx::CustomOperators::IpuCopy);
}

void IpuCopyOpx::grow(poplar::program::Sequence &prog) const {

  IpuCopyOp &op = getOp<IpuCopyOp>();

  logging::devicex::trace(
      "Adding copyToIpu for {}, {}", op.str(), op.getFromToStr());

  for (auto &idx_tensor : op.input->tensorMap()) {
    auto idx = idx_tensor.first;
    // Need to get the non virtual graph, so cannot use Opx::graph()
    auto t = poputil::copyToIpu(dv_p->graph(),
                                getInTensor(idx),
                                prog,
                                static_cast<int>(op.getDestIpu()),
                                debugPrefix());
    setOutTensor(idx, t);
  }
}

void IpuCopyOpx::createPipelinedOutput() const {
  IpuCopyOp &op = getOp<IpuCopyOp>();

  logging::devicex::trace(
      "Creating destination tensors for {}, {}", op.str(), op.getFromToStr());

  for (auto &idx_tensor : op.input->tensorMap()) {
    auto idx = idx_tensor.first;

    // When pipelining, create the copy destination, but dont add the copy
    // program.
    poplar::Tensor tLocalForCopy, tForCopy;
    auto t = poputil::createIpuCopy(dv_p->graph(),
                                    getInTensor(idx),
                                    static_cast<int>(op.getDestIpu()),
                                    tForCopy,
                                    tLocalForCopy,
                                    debugPrefix("createOutput"));
    setOutTensor(idx, t);
  }
}

void IpuCopyOpx::growPipelined(poplar::program::Sequence &prog) const {
  IpuCopyOp &op = getOp<IpuCopyOp>();

  for (auto &idx_tensor : op.input->tensorMap()) {
    auto idx   = idx_tensor.first;
    auto outId = op_p->outId(idx);

    auto &source      = getInTensor(idx);
    auto &destination = dv_p->tensors.get(outId);

    // Using dontOutline=true will make each copy to be unique
    // which will then allow them to be merged into a single copy/sync.
    prog.add(poplar::program::Copy(source, destination, true));
  }
}

namespace {
OpxCreator<IpuCopyOpx> ipuCopyOpxCreator(Onnx::CustomOperators::IpuCopy);
} // namespace

} // namespace popx
} // namespace popart
