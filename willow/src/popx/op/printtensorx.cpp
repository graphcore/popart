// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/op/printtensor.hpp>
#include <popart/popx/op/printtensorx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {
namespace popx {

PrintTensorOpx::PrintTensorOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<PrintTensorOp>(op, Onnx::CustomOperators::PrintTensor_1);
}

void PrintTensorOpx::grow(poplar::program::Sequence &prog) const {
  auto input = getInTensor(PrintTensorOp::getInIndex());

  if (getOp<PrintTensorOp>().shouldPrint()) {
    auto printProg =
        poplar::program::PrintTensor(getTitle(), input.getPoplarTensor());
    prog.add(printProg);
  }

  auto output = cloneNcopy(prog, input);
  setOutTensor(PrintTensorOp::getOutIndex(), output);
}

std::string PrintTensorOpx::getTitle() const {
  const auto &op = getOp<PrintTensorOp>();
  auto title     = op.getTitle();

  if (op_p->scheduledPreLoss == ScheduledPreLoss::Yes) {
    if (title.size() > 0) {
      return title;
    } else {
      return op_p->inTensor(PrintTensorOp::getInIndex())->id;
    }
  } else if (op_p->scheduledPreLoss == ScheduledPreLoss::No) {
    if (title.size() > 0) {
      return logging::format("{}_gradient", title);
    } else {
      return op_p->outTensor(PrintTensorOp::getOutIndex())->id;
    }
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
