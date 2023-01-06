// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <string>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/ir.hpp>
#include <popart/op/add.hpp>
#include <popart/popx/op/addx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/elementwisex.hpp"
#include "popart/popx/op/reducesumx.hpp"
#include "popart/popx/opx.hpp"
#include "popart/sessionoptions.hpp"

namespace poplar {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
namespace popx {
class Devicex;

AddComputex::AddComputex(EwbComputex::InplacePolicy ip) : EwbComputex(ip) {}

poplar::Tensor AddComputex::outplace(poplar::program::Sequence &prog,
                                     poplar::Graph &graph,
                                     const poplar::Tensor &a,
                                     const poplar::Tensor &b,
                                     const poplar::DebugNameAndId &dnai,
                                     const std::string &name) const {
  return popops::add(graph, a, b, prog, {dnai, name});
}

poplar::Tensor AddComputex::maybeInplace(poplar::program::Sequence &prog,
                                         poplar::Graph &graph,
                                         poplar::Tensor &tInOut,
                                         poplar::Tensor &tIn,
                                         const poplar::DebugNameAndId &dnai,
                                         const std::string &name) const {

  return mapMaybeInPlace(graph,
                         popops::expr::BinaryOpType::ADD,
                         tInOut,
                         tIn,
                         prog,
                         {dnai, name},
                         {},
                         name);
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
