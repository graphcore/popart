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

poplar::Tensor AddComputex::outplace(poplar::program::Sequence &prog,
                                     poplar::Graph &graph,
                                     const poplar::Tensor &a,
                                     const poplar::Tensor &b,
                                     const std::string &debugStr) const {
  return popops::add(graph, a, b, prog, debugStr);
}

void AddComputex::inplace(poplar::program::Sequence &prog,
                          poplar::Graph &graph,
                          const poplar::Tensor &tInOut,
                          const poplar::Tensor &tIn,
                          const std::string &debugStr) const {
  popops::addInPlace(graph, tInOut, tIn, prog, debugStr);
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

InputCreatorType AddOpx::getInputCreatorType(InIndex index) const {
  // TODO: T17972 Allowing add (in particular lhs, rhs inplace adds) leads to
  // inefficient sub graph copying. Investigate why, then remove the below logic
  // once fixed.
  if (!(op_p->getIr().getSessionOptions().decomposeGradSum ||
        op_p->getIr().getSessionOptions().batchSerializationSettings.factor >
            0)) {
    return InputCreatorType::Deadend;
  }

  return ElementWiseBinaryOpx::getInputCreatorType(index);
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
