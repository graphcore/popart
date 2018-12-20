#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/sum.hpp>
#include <poponnx/popx/op/sumx.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {
namespace popx {

SumOpx::SumOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SumOp>(op, Onnx::Operators::Sum);
}

void SumOpx::grow(poplar::program::Sequence &prog) const {

  SumOp &sumOp = getOp<SumOp>();

  if (sumOp.input->n() == 1) {
    throw error(
        "SumOpx with one input should be removed by pattern 'PreUniRepl'");
  }
  // if the total number of tensors is less than
  // "5", then perform a series of adds.
  else if (sumOp.input->n() < 5) {
    poplar::Tensor sum = popops::map(graph(),
                                     popops::expr::BinaryOpType::ADD,
                                     get(inId(0)),
                                     get(inId(1)),
                                     prog,
                                     idStr());

    for (InIndex i = 2; i < sumOp.input->n(); ++i) {
      popops::mapInPlace(graph(),
                         popops::expr::BinaryOpType::ADD,
                         sum,
                         get(inId(i)),
                         prog,
                         idStr());
    }
    insert(outId(SumOp::getOutIndex()), sum);
  }

  else {
    throw error("Must implemented SumOpx::grow() for greater than 4 inputs");
  }
}

namespace {
OpxCreator<SumOpx> sumOpxCreator(Onnx::Operators::Sum);
}

} // namespace popx
} // namespace poponnx
