// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/mul.hpp>
#include <popart/popx/op/mulx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace popart {
namespace popx {

MulComputex::MulComputex(EwbComputex::InplacePolicy ip) : EwbComputex(ip) {}

poplar::Tensor MulComputex::outplace(poplar::program::Sequence &prog,
                                     poplar::Graph &graph,
                                     const poplar::Tensor &a,
                                     const poplar::Tensor &b,
                                     const poplar::DebugNameAndId &dnai,
                                     const std::string &debugStr) const {
  return popops::mul(graph, a, b, prog, {dnai, debugStr});
}

void MulComputex::inplace(poplar::program::Sequence &prog,
                          poplar::Graph &graph,
                          const poplar::Tensor &tInOut,
                          const poplar::Tensor &tIn,
                          const poplar::DebugNameAndId &dnai,
                          const std::string &debugStr) const {
  popops::mulInPlace(graph, tInOut, tIn, prog, {dnai, debugStr});
}

MulOpx::MulOpx(Op *op, Devicex *devicex)
    : ElementWiseBinaryOutplaceOpx(
          op,
          devicex,
          std::make_unique<MulComputex>(EwbComputex::InplacePolicy::NEVER)) {
  verifyOp<MulOp>(op, {Onnx::Operators::Mul_6, Onnx::Operators::Mul_7});
}

MulLhsInplaceOpx::MulLhsInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseBinaryInplaceOpx(
          op,
          devicex,
          std::make_unique<MulComputex>(EwbComputex::InplacePolicy::LHS)) {
  verifyOp<MulLhsInplaceOp>(op);
}

MulRhsInplaceOpx::MulRhsInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseBinaryInplaceOpx(
          op,
          devicex,
          std::make_unique<MulComputex>(EwbComputex::InplacePolicy::RHS)) {
  verifyOp<MulRhsInplaceOp>(op);
}

namespace {
static OpxCreator<MulOpx> mulOpxCreator({Onnx::Operators::Mul_6,
                                         Onnx::Operators::Mul_7});
OpxCreator<MulLhsInplaceOpx>
    mulLhsInplaceOpxCreator(Onnx::CustomOperators::MulLhsInplace);
OpxCreator<MulRhsInplaceOpx>
    mulRhsInplaceOpxCreator(Onnx::CustomOperators::MulRhsInplace);

static OpxCreator<Opx>
    mulArg0GradOpxCreator(Onnx::GradOperators::MulArg0Grad,
                          "MulArg0GradOp should be optimised out, "
                          "\"MulArgGradOp\" pattern is required");
static OpxCreator<Opx>
    mulArg1GradOpxCreator(Onnx::GradOperators::MulArg1Grad,
                          "MulArg1GradOp should be optimised out, "
                          "\"MulArgGradOp\" pattern is required");
} // namespace

} // namespace popx
} // namespace popart
