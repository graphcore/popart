#include <poponnx/error.hpp>
#include <poponnx/op/add.hpp>
#include <poponnx/popx/op/addx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace poponnx {
namespace popx {

AddOpx::AddOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<AddOp>(op, Onnx::Operators::Add);
}

void AddOpx::grow(poplar::program::Sequence &prog) const {
  insert(outId(AddOp::getOutIndex()),
         popops::map(graph(),
                     popops::expr::BinaryOpType::ADD,
                     get(inId(AddOp::getArg0InIndex())),
                     get(inId(AddOp::getArg1InIndex())),
                     prog,
                     idStr()));
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
OpxCreator<AddOpx> addOpxCreator(Onnx::Operators::Add);
OpxCreator<AddArg0GradOpx>
    addArg0GradOpxCreator(Onnx::GradOperators::AddArg0Grad);
OpxCreator<AddArg1GradOpx>
    addArg1GradOpxCreator(Onnx::GradOperators::AddArg1Grad);
} // namespace

} // namespace popx
} // namespace poponnx
