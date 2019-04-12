#include <poponnx/error.hpp>
#include <poponnx/op/argextrema.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/argextremax.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

ArgExtremaOpx::ArgExtremaOpx(Op *op, Devicex *devicex)
    : BaseSortOpx(op, devicex) {
  verifyOp<ArgExtremaOp>(op);
  keepdims = dynamic_cast<ArgExtremaOp *>(op)->getKeepDims() != 0;
}

void ArgExtremaOpx::grow(poplar::program::Sequence &prog) const {

  // Use the specialised slice
  poplar::Tensor values = selectSlice(growIndicesSort(prog), axis);

  // Squeeze out the axis dimension?
  if (!keepdims) {
    values = values.squeeze({axis});
  }
  setOutTensor(ArgExtremaOp::getOutIndex(), values);
}

} // namespace popx
} // namespace poponnx
