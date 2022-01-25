// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/add.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/addx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

#include <popops/ElementWise.hpp>
#include <poputil/TileMapping.hpp>

namespace popart {
namespace popx {

AddComputex::AddComputex(EwbComputex::InplacePolicy ip) : EwbComputex(ip) {}

snap::Tensor AddComputex::outplace(snap::program::Sequence &prog,
                                   snap::Graph &graph,
                                   const snap::Tensor &a,
                                   const snap::Tensor &b,
                                   const poplar::DebugNameAndId &dnai,
                                   const std::string &name) const {
  return snap::Tensor{popops::add(graph.getPoplarGraph(),
                                  a.getPoplarTensor(),
                                  b.getPoplarTensor(),
                                  prog.getPoplarSequence(),
                                  {dnai, name}),
                      graph};
}

void AddComputex::inplace(snap::program::Sequence &prog,
                          snap::Graph &graph,
                          const snap::Tensor &tInOut,
                          const snap::Tensor &tIn,
                          const poplar::DebugNameAndId &dnai,
                          const std::string &name) const {
  popops::addInPlace(graph.getPoplarGraph(),
                     tInOut.getPoplarTensor(),
                     tIn.getPoplarTensor(),
                     prog.getPoplarSequence(),
                     {dnai, name});
}

AddOpx::AddOpx(Op *op, Devicex *devicex)
    : ElementWiseBinaryOutplaceOpx(
          op,
          devicex,
          std::make_unique<AddComputex>(EwbComputex::InplacePolicy::NEVER)) {
  verifyOp<AddOp>(op,
                  {Onnx::Operators::Add_6,
                   Onnx::Operators::Add_7,
                   Onnx::CustomOperators::AddLhsInplace,
                   Onnx::CustomOperators::AddRhsInplace});
}

AddLhsInplaceOpx::AddLhsInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseBinaryInplaceOpx(
          op,
          devicex,
          std::make_unique<AddComputex>(EwbComputex::InplacePolicy::LHS)) {
  verifyOp<AddLhsInplaceOp>(op);
}

AddRhsInplaceOpx::AddRhsInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseBinaryInplaceOpx(
          op,
          devicex,
          std::make_unique<AddComputex>(EwbComputex::InplacePolicy::RHS)) {
  verifyOp<AddRhsInplaceOp>(op);
}

AddArg0GradOpx::AddArg0GradOpx(Op *op, Devicex *devicex)
    : ReduceSumOpx(op, devicex) {
  verifyOp<AddArg0GradOp>(op, Onnx::GradOperators::AddArg0Grad);
}

AddArg1GradOpx::AddArg1GradOpx(Op *op, Devicex *devicex)
    : ReduceSumOpx(op, devicex) {
  verifyOp<AddArg1GradOp>(op, Onnx::GradOperators::AddArg1Grad);
}

namespace {
// OpxCreator<AddOpx> addOpxCreator({Onnx::Operators::Add_6,
// Onnx::Operators::Add_7});

OpxCreator<AddOpx> addOpxCreator_6(Onnx::Operators::Add_6);
OpxCreator<AddOpx> addOpxCreator_7(Onnx::Operators::Add_7);
OpxCreator<AddLhsInplaceOpx>
    addLhsInplaceOpxCreator(Onnx::CustomOperators::AddLhsInplace);
OpxCreator<AddRhsInplaceOpx>
    addRhsInplaceOpxCreator(Onnx::CustomOperators::AddRhsInplace);

OpxCreator<AddArg0GradOpx>
    addArg0GradOpxCreator(Onnx::GradOperators::AddArg0Grad);
OpxCreator<AddArg1GradOpx>
    addArg1GradOpxCreator(Onnx::GradOperators::AddArg1Grad);
} // namespace

} // namespace popx
} // namespace popart
