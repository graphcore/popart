#include <poponnx/op/printtensor.hpp>
#include <poponnx/popx/op/printtensorx.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {
namespace popx {

PrintTensorOpx::PrintTensorOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<PrintTensorOp>(op, Onnx::CustomOperators::PrintTensor_1);
}

void PrintTensorOpx::grow(poplar::program::Sequence &prog) const {
  auto input = getInTensor(PrintTensorOp::getInIndex());

  if (getOp<PrintTensorOp>().shouldPrint()) {
    auto printProg = poplar::program::PrintTensor(getTitle(), input);
    prog.add(printProg);
  }

  auto output = cloneNcopy(prog, input);
  setOutTensor(PrintTensorOp::getOutIndex(), output);
}

std::string PrintTensorOpx::getTitle() const {
  if (op_p->getPhase() == Phase::FWD) {
    return op_p->inTensor(PrintTensorOp::getInIndex())->id;
  } else {
    return op_p->outTensor(PrintTensorOp::getOutIndex())->id;
  }
}

namespace {
OpxCreator<PrintTensorOpx>
    printtensorOpxCreator(Onnx::CustomOperators::PrintTensor_1);
} // namespace

} // namespace popx
} // namespace poponnx
