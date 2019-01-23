#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/sum.hpp>
#include <poponnx/popx/op/sumx.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {
namespace popx {

SumOpx::SumOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SumOp>(op, {Onnx::Operators::Sum_6, Onnx::Operators::Sum_8});
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

InputCreatorType SumOpx::getInputCreatorType(InIndex index) const {
  SumOp &sumOp = getOp<SumOp>();
  // if the total number of tensors is less than
  // "5", then perform a series of adds.
  if (sumOp.input->n() < 5) {
    // Check shape doesn't change due to numpy-style broadcasting.
    // Design choice: even without broadcasting, it is possible for the
    // two inputs (of same shape) have different layout.
    // The poplar binary op can choose the layout of the output to take
    // the layout of either input.
    // However, let's layout both inputs in the same way. That way we can
    // definitely unwind through this opx, and it will also be efficient
    // when performing the op.
    if (sumOp.inInfo(index) == sumOp.outInfo(SumOp::getOutIndex())) {
      return InputCreatorType::AGNOSTICTOLAYOUT;
    } else {
      return InputCreatorType::DEADEND;
    }
  } else {
    throw error("Must implemented SumOpx::getInputCreatorType() for greater "
                "than 4 inputs");
  }
}

namespace {
OpxCreator<SumOpx> sumOpxCreator({Onnx::Operators::Sum_6,
                                  Onnx::Operators::Sum_8});
} // namespace

} // namespace popx
} // namespace poponnx
