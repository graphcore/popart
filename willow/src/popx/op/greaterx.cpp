// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popops/Cast.hpp>
#include <popart/error.hpp>
#include <popart/op/greater.hpp>
#include <popart/popx/devicex.hpp>

#include <popart/popx/op/greaterx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <snap/popops/ElementWise.hpp>

namespace popart {
namespace popx {

GreaterOpx::GreaterOpx(Op *op, Devicex *devicex)
    : BinaryComparisonOpx(op, devicex) {
  verifyOp<GreaterOp>(op,
                      {Onnx::Operators::Greater_7, Onnx::Operators::Greater_9});
}

void GreaterOpx::grow(snap::program::Sequence &prog) const {

  insert(outId(GreaterOp::getOutIndex()),
         snap::popops::map(graph(),
                           popops::expr::BinaryOpType::GREATER_THAN,
                           get(inId(GreaterOp::getArg0InIndex())),
                           get(inId(GreaterOp::getArg1InIndex())),
                           prog,
                           debugContext()));
}

namespace {

OpxCreator<GreaterOpx> greaterOpxCreator_7(Onnx::Operators::Greater_7);
OpxCreator<GreaterOpx> greaterOpxCreator_9(Onnx::Operators::Greater_9);

} // namespace

} // namespace popx
} // namespace popart
