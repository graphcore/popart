#include <poponnx/error.hpp>
#include <poponnx/op/add.hpp>
#include <poponnx/popx/op/addx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace poponnx {
namespace popx {

static poplar::Tensor addInplace(poplar::Graph &graph,
                                 const poplar::Tensor &t_inout,
                                 const poplar::Tensor &t_in,
                                 poplar::program::Sequence &prog,
                                 const std::string debug_id) {
  if (!t_inout.isParallelWriteable()) {
    logging::debug(
        "Unable to inplace add operation {}, tensor is not parallel writeable",
        debug_id);
    return popops::map(
        graph, popops::expr::BinaryOpType::ADD, t_inout, t_in, prog, debug_id);
  } else {
    popops::mapInPlace(
        graph, popops::expr::BinaryOpType::ADD, t_inout, t_in, prog, debug_id);
    return t_inout;
  }
}

AddOpx::AddOpx(Op *op, Devicex *devicex) : ElementWiseBinaryOpx(op, devicex) {
  verifyOp<AddOp>(op, {Onnx::Operators::Add_6, Onnx::Operators::Add_7});
}

void AddOpx::grow(poplar::program::Sequence &prog) const {

  setOutTensor(AddOp::getOutIndex(),
               popops::map(graph(),
                           popops::expr::BinaryOpType::ADD,
                           getInTensor(AddOp::getArg0InIndex()),
                           getInTensor(AddOp::getArg1InIndex()),
                           prog,
                           idStr()));
}

AddLhsInplaceOpx::AddLhsInplaceOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<AddLhsInplaceOp>(op);
}

void AddLhsInplaceOpx::grow(poplar::program::Sequence &prog) const {
  auto out = addInplace(graph(),
                        getInTensor(AddLhsInplaceOp::getArg0InIndex()),
                        getInTensor(AddLhsInplaceOp::getArg1InIndex()),
                        prog,
                        idStr());
  setOutTensor(AddLhsInplaceOp::getOutIndex(), out);
}

AddRhsInplaceOpx::AddRhsInplaceOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<AddRhsInplaceOp>(op);
}

void AddRhsInplaceOpx::grow(poplar::program::Sequence &prog) const {
  auto out = addInplace(graph(),
                        getInTensor(AddRhsInplaceOp::getArg1InIndex()),
                        getInTensor(AddRhsInplaceOp::getArg0InIndex()),
                        prog,
                        idStr());

  setOutTensor(AddRhsInplaceOp::getOutIndex(), out);
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
} // namespace poponnx
