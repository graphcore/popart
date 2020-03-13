// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popart/error.hpp>
#include <popart/op/reciprocal.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/reciprocalx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {
namespace popx {

ReciprocalOpx::ReciprocalOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<ReciprocalOp>(op, Onnx::Operators::Reciprocal_6);
}

void ReciprocalOpx::grow(poplar::program::Sequence &prog) const {
  auto ones = getConst(popType(op_p->inInfo(0)), {1}, 1.0, debugPrefix("ones"));

  setOutTensor(0,
               popops::map(graph(),
                           popops::expr::BinaryOpType::DIVIDE,
                           ones,
                           getInTensor(0),
                           prog,
                           debugPrefix("divide")));
}

namespace {
OpxCreator<ReciprocalOpx> reciprocalOpxCreator(Onnx::Operators::Reciprocal_6);
} // namespace

} // namespace popx
} // namespace popart
