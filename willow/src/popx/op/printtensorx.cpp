// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <snap/Program.hpp>
#include <string>
#include <poplar/PrintTensor.hpp>
#include <poplar/StringRef.hpp>
#include <popart/op/printtensor.hpp>
#include <popart/popx/op/printtensorx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/vertex.hpp"

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

PrintTensorOpx::PrintTensorOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<PrintTensorOp>(op, Onnx::CustomOperators::PrintTensor_1);
}

void PrintTensorOpx::grow(snap::program::Sequence &prog) const {
  const auto &op = getOp<PrintTensorOp>();
  auto input     = getInTensor(PrintTensorOp::getInIndex());

  if (op.shouldPrint()) {
    auto printProg = snap::program::PrintTensor(
        getTitle(), input, toPoplarPrintTensorFmt(op.getFmt()));
    prog.getPoplarSequence().add(printProg);
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
