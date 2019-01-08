#include <poponnx/error.hpp>
#include <poponnx/op/ipucopy.hpp>
#include <poponnx/popx/op/ipucopyx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <poputil/TileMapping.hpp>

namespace poponnx {
namespace popx {

IpuCopyOpx::IpuCopyOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<IpuCopyOp>(op, Onnx::CustomOperators::IpuCopy);
}

void IpuCopyOpx::grow(poplar::program::Sequence &prog) const {

  IpuCopyOp &op = getOp<IpuCopyOp>();

  insert(
      outId(0),
      poputil::copyToIpu(masterGraph(), get(inId(0)), prog, op.getDestIpu()));
}

namespace {
OpxCreator<IpuCopyOpx> ipuCopyOpxCreator(Onnx::CustomOperators::IpuCopy);
} // namespace

} // namespace popx
} // namespace poponnx
