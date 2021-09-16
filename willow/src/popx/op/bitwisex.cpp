// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/bitwise.hpp>
#include <popart/popx/devicex.hpp>

#include <popart/popx/op/bitwisex.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace popart {
namespace popx {

BitwiseNotOpx::BitwiseNotOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<BitwiseNotOpx>(op, {Onnx::AiGraphcore::OpSet1::BitwiseNot});
}

void BitwiseNotOpx::grow(snap::program::Sequence &prog) const {
  insert(
      outId(BitwiseNotOp::getOutIndex()),
      snap::Tensor{
          popops::map(graph().getPoplarGraph(),
                      popops::expr::UnaryOpType::BITWISE_NOT,
                      get(inId(BitwiseNotOp::getInIndex())).getPoplarTensor(),
                      prog.getPoplarSequence(),
                      debugContext()),
          graph()});
}

BitwiseBinaryOpx::BitwiseBinaryOpx(Op *op, Devicex *devicex)
    : ElementWiseBinaryOpx(op, devicex) {
  verifyOp<BitwiseBinaryOpx>(op,
                             {Onnx::AiGraphcore::OpSet1::BitwiseAnd,
                              Onnx::AiGraphcore::OpSet1::BitwiseOr,
                              Onnx::AiGraphcore::OpSet1::BitwiseXor,
                              Onnx::AiGraphcore::OpSet1::BitwiseXnor});
}

void BitwiseBinaryOpx::grow(snap::program::Sequence &prog) const {
  insert(
      outId(BitwiseBinaryOp::getOutIndex()),
      snap::Tensor{
          popops::map(
              graph().getPoplarGraph(),
              determineOpType(),
              getInTensor(BitwiseBinaryOp::getArg0InIndex()).getPoplarTensor(),
              getInTensor(BitwiseBinaryOp::getArg1InIndex()).getPoplarTensor(),
              prog.getPoplarSequence(),
              debugContext()),
          graph()});
}

popops::expr::BinaryOpType BitwiseBinaryOpx::determineOpType() const {
  if (op_p->opid == Onnx::AiGraphcore::OpSet1::BitwiseAnd) {
    return popops::expr::BinaryOpType::BITWISE_AND;
  }
  if (op_p->opid == Onnx::AiGraphcore::OpSet1::BitwiseOr) {
    return popops::expr::BinaryOpType::BITWISE_OR;
  }
  if (op_p->opid == Onnx::AiGraphcore::OpSet1::BitwiseXor) {
    return popops::expr::BinaryOpType::BITWISE_XOR;
  }
  if (op_p->opid == Onnx::AiGraphcore::OpSet1::BitwiseXnor) {
    return popops::expr::BinaryOpType::BITWISE_XNOR;
  }
  throw error("Unknown opx type {}", op_p->opid);
}

namespace {

OpxCreator<BitwiseNotOpx>
    bitwiseNotOpxCreator(Onnx::AiGraphcore::OpSet1::BitwiseNot);

OpxCreator<BitwiseBinaryOpx>
    bitwiseAndOpxCreator(Onnx::AiGraphcore::OpSet1::BitwiseAnd);

OpxCreator<BitwiseBinaryOpx>
    bitwiseOrOpxCreator(Onnx::AiGraphcore::OpSet1::BitwiseOr);
OpxCreator<BitwiseBinaryOpx>
    bitwiseXorOpxCreator(Onnx::AiGraphcore::OpSet1::BitwiseXor);
OpxCreator<BitwiseBinaryOpx>
    bitwiseXnorOpxCreator(Onnx::AiGraphcore::OpSet1::BitwiseXnor);

} // namespace

} // namespace popx
} // namespace popart
