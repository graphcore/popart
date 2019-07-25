#include <popart/op/printtensor.hpp>
#include <popart/popx/op/printtensorx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {
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
  if (op_p->scheduledPreLoss == ScheduledPreLoss::Yes) {
    return op_p->inTensor(PrintTensorOp::getInIndex())->id;
  } else if (op_p->scheduledPreLoss == ScheduledPreLoss::No) {
    return op_p->outTensor(PrintTensorOp::getOutIndex())->id;
  } else {
    throw error("ScheduledPreLoss Unknown not allowed in getTitle");
  }
}

namespace {
OpxCreator<PrintTensorOpx>
    printtensorOpxCreator(Onnx::CustomOperators::PrintTensor_1);
} // namespace

} // namespace popx
} // namespace popart
