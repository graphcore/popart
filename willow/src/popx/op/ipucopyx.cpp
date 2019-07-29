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

  for (auto &idx_tensor : op.input->tensorMap()) {
    auto idx = idx_tensor.first;
    // Need to get the non virtual graph, so cannot use Opx::graph()
    auto t = poputil::copyToIpu(dv_p->graph(),
                                getInTensor(idx),
                                prog,
                                static_cast<int>(op.getDestIpu()));
    setOutTensor(idx, t);
  }
}

namespace {
OpxCreator<IpuCopyOpx> ipuCopyOpxCreator(Onnx::CustomOperators::IpuCopy);
} // namespace

} // namespace popx
} // namespace popart
