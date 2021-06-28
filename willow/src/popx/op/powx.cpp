// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/error.hpp>
#include <popart/op/pow.hpp>
#include <popart/popx/op/powx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace popart {
namespace popx {

PowComputex::PowComputex(EwbComputex::InplacePolicy ip) : EwbComputex(ip) {}

snap::Tensor PowComputex::outplace(poplar::program::Sequence &prog,
                                   snap::Graph &graph,
                                   const snap::Tensor &a,
                                   const snap::Tensor &b,
                                   const poplar::DebugNameAndId &dnai,
                                   const std::string &debugStr) const {
  return snap::Tensor{popops::pow(graph.getPoplarGraph(),
                                  a.getPoplarTensor(),
                                  b.getPoplarTensor(),
                                  prog,
                                  {dnai, debugStr}),
                      graph};
}

void PowComputex::inplace(poplar::program::Sequence &prog,
                          snap::Graph &graph,
                          const snap::Tensor &tInOut,
                          const snap::Tensor &tIn,
                          const poplar::DebugNameAndId &dnai,
                          const std::string &debugStr) const {
  popops::powInPlace(graph.getPoplarGraph(),
                     tInOut.getPoplarTensor(),
                     tIn.getPoplarTensor(),
                     prog,
                     {dnai, debugStr});
}

PowOpx::PowOpx(Op *op, Devicex *devicex)
    : ElementWiseBinaryOutplaceOpx(
          op,
          devicex,
          std::make_unique<PowComputex>(EwbComputex::InplacePolicy::NEVER)) {
  verifyOp<PowOp>(op, {Onnx::Operators::Pow_1, Onnx::Operators::Pow_7});
}

PowLhsInplaceOpx::PowLhsInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseBinaryInplaceOpx(
          op,
          devicex,
          std::make_unique<PowComputex>(EwbComputex::InplacePolicy::LHS)) {
  verifyOp<PowLhsInplaceOp>(op);
}

namespace {
OpxCreator<PowOpx> powOpxCreator({Onnx::Operators::Pow_1,
                                  Onnx::Operators::Pow_7});
OpxCreator<PowLhsInplaceOpx>
    powLhsInplaceOpxCreator(Onnx::CustomOperators::PowLhsInplace);
OpxCreator<PopOpx> powArg0OpxCreator(Onnx::GradOperators::PowArg0Grad,
                                     "PowArg0Grad should be optimised out, "
                                     "\"PowArg0GradOp\" pattern is required");
OpxCreator<PopOpx> powArg1OpxCreator(Onnx::GradOperators::PowArg1Grad,
                                     "PowArg1Grad should be optimised out, "
                                     "\"PowArg1GradOp\" pattern is required");
} // namespace

} // namespace popx
} // namespace popart
