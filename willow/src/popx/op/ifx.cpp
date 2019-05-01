#include <poponnx/op/if.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/ifx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

IfOpx::IfOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<IfOp>(op, Onnx::Operators::If_1);
}

void IfOpx::copyInputs(poplar::program::Sequence &prog,
                       const Scope &scope,
                       const std::vector<TensorId> &input_ids) const {
  auto graph_id = scope.str();
  for (auto &id : input_ids) {
    auto ifop_input   = get(id);
    auto branch_input = get((scope / id).str());
    poplar::program::Copy copy_prog(ifop_input, branch_input);
    prog.add(copy_prog);
  }
}

std::vector<poplar::Tensor> IfOpx::prepareOutputs() const {
  std::vector<poplar::Tensor> outputs;
  auto &ifop = getOp<IfOp>();

  for (auto &id : ifop.getThenOutputIds()) {
    auto then_output = get((ifop.getThenScope() / id).str());
    outputs.push_back(graph().clone(then_output));
  }

  return outputs;
}

poplar::program::Sequence
IfOpx::prepareBranch(const Scope &scope,
                     const std::vector<TensorId> &input_ids,
                     const std::vector<TensorId> &output_ids,
                     const std::vector<poplar::Tensor> &outputs) const {
  auto &branch_prog = dv_p->programFragment(scope);
  poplar::program::Sequence container_prog;
  copyInputs(container_prog, scope, input_ids);
  container_prog.add(branch_prog);
  copyOutputs(container_prog, scope, output_ids, outputs);
  return container_prog;
}

void IfOpx::copyOutputs(poplar::program::Sequence &prog,
                        const Scope &scope,
                        const std::vector<TensorId> &output_ids,
                        const std::vector<poplar::Tensor> &outputs) const {
  for (int i = 0; i < output_ids.size(); i++) {
    auto &id           = output_ids[i];
    auto branch_output = get((scope / id).str());
    auto if_out        = outputs[i];
    poplar::program::Copy copy_prog(branch_output, if_out);
    prog.add(copy_prog);
  }
}

void IfOpx::grow(poplar::program::Sequence &prog) const {
  auto &ifop = getOp<IfOp>();

  auto outputs = prepareOutputs();

  auto then_prog = prepareBranch(ifop.getThenScope(),
                                 ifop.getThenInputIds(),
                                 ifop.getThenOutputIds(),
                                 outputs);
  auto else_prog = prepareBranch(ifop.getElseScope(),
                                 ifop.getElseInputIds(),
                                 ifop.getElseOutputIds(),
                                 outputs);

  auto condition = getInTensor(IfOp::getConditionInIndex());
  prog.add(poplar::program::If(condition, then_prog, else_prog));

  for (int i = 0; i < outputs.size(); i++) {
    setOutTensor(i, outputs[i]);
  }
}

namespace {
OpxCreator<IfOpx> ifOpxCreator(Onnx::Operators::If_1);
}

} // namespace popx
} // namespace poponnx
