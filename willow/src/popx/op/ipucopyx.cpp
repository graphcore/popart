#include <poponnx/error.hpp>
#include <poponnx/op/ipucopy.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/ipucopyx.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/tensorindex.hpp>

#include <poputil/TileMapping.hpp>

namespace poponnx {
namespace popx {

IpuCopyOpx::IpuCopyOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<IpuCopyOp>(op, Onnx::CustomOperators::IpuCopy);
}

void IpuCopyOpx::grow(poplar::program::Sequence &prog) const {

  IpuCopyOp &op = getOp<IpuCopyOp>();

  for (auto &idx_tensor : op.input->tensorMap()) {
    auto idx = idx_tensor.first;
    // Need to get the non virtual graph, so can not use Opx::graph()
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
} // namespace poponnx
