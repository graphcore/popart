#include <poponnx/op/if.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/ifx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

IfOpx::IfOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<IfOp>(op, Onnx::Operators::If_1);
}

void IfOpx::grow(poplar::program::Sequence &prog) const {
  auto &ifop = getOp<IfOp>();

  auto condition = getInTensor(IfOp::getConditionInIndex());

  auto &then_prog = dv_p->programFragment(ifop.getThenScope());
  auto &else_prog = dv_p->programFragment(ifop.getElseScope());

  std::vector<poplar::Tensor> outputs;

  for (int i = 0; i < ifop.inputsPerBranch(); i++) {
    auto input = getInTensor(ifop.getThenBranchInIndex(i));
    outputs.push_back(graph().clone(input));
    poplar::program::Copy copy_prog(input, outputs[i]);
    then_prog.add(copy_prog);
  }

  for (int i = 0; i < ifop.inputsPerBranch(); i++) {
    auto input = getInTensor(ifop.getElseBranchInIndex(i));
    poplar::program::Copy copy_prog(input, outputs[i]);
    else_prog.add(copy_prog);
  }

  prog.add(poplar::program::If(condition, then_prog, else_prog));

  for (int i = 0; i < ifop.inputsPerBranch(); i++) {
    setOutTensor(i, outputs[i]);
  }
}

namespace {
OpxCreator<IfOpx> ifOpxCreator(Onnx::Operators::If_1);
}

} // namespace popx
} // namespace poponnx
