// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <string>
#include <poplar/PrintTensor.hpp>
#include <poplar/Program.hpp>
#include <poplar/StringRef.hpp>
#include <popart/op/printtensor.hpp>
#include <popart/popx/op/printtensorx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/op.hpp"
#include "popart/popx/opx.hpp"
#include "popart/printtensorfmt.hpp"

namespace popart {
namespace popx {
class Devicex;

const poplar::PrintTensorFmt toPoplarPrintTensorFmt(const PrintTensorFmt &fmt) {
  return poplar::PrintTensorFmt(
      fmt.summariseThreshold,
      fmt.edgeItems,
      fmt.maxLineWidth,
      fmt.digits,
      static_cast<poplar::PrintTensorFmt::FloatFormat>(
          static_cast<int>(fmt.floatFormat)),
      fmt.separator,
      fmt.openBracket,
      fmt.closeBracket);
}

PrintTensorOpx::PrintTensorOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<PrintTensorOp>(op, Onnx::CustomOperators::PrintTensor_1);
}

void PrintTensorOpx::grow(poplar::program::Sequence &prog) const {
  const auto &op = getOp<PrintTensorOp>();
  auto input     = getInTensor(PrintTensorOp::getInIndex());

  if (op.shouldPrint()) {
    auto printProg = poplar::program::PrintTensor(
        getTitle(), input, toPoplarPrintTensorFmt(op.getFmt()));
    prog.add(printProg);
  }

  auto output = cloneNcopy(prog, input);
  setOutTensor(PrintTensorOp::getOutIndex(), output);
}

std::string PrintTensorOpx::getTitle() const {
  const auto &op = getOp<PrintTensorOp>();
  auto title     = op.getTitle();

  if (title.size() > 0) {
    return title;
  } else {
    return op_p->inTensor(PrintTensorOp::getInIndex())->id;
  }
}

namespace {
OpxCreator<PrintTensorOpx>
    printtensorOpxCreator(Onnx::CustomOperators::PrintTensor_1);
} // namespace

} // namespace popx
} // namespace popart
