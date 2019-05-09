#include <poponnx/graph.hpp>
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
                       const Graph &branch_graph,
                       const std::vector<TensorId> &input_ids) const {
  auto graph_id = branch_graph.id.str();
  for (auto &id : input_ids) {
    auto ifop_input      = get(id);
    auto branch_input_id = (Scope() / graph_id / id).str();

    if (!dv_p->tensors.contains(branch_input_id)) {
      dv_p->tensors.insert(branch_input_id, graph().clone(ifop_input));
    }

    auto branch_input = get(branch_input_id);
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

void IfOpx::copyOutputs(poplar::program::Sequence &prog,
                        const Graph &graph,
                        const std::vector<TensorId> &output_ids,
                        const std::vector<poplar::Tensor> &outputs) const {
  for (int i = 0; i < output_ids.size(); i++) {
    auto &id           = output_ids[i];
    auto branch_output = get((Scope() / graph.id.str() / id).str());
    auto if_out        = outputs[i];
    poplar::program::Copy copy_prog(branch_output, if_out);
    prog.add(copy_prog);
  }
}

void IfOpx::callBranch(poplar::program::Sequence &prog,
                       const Graph &graph) const {
  if (!dv_p->containsFragment(graph)) {
    dv_p->createFragmentAndGrow(graph);
  }

  auto &branch_prog = dv_p->programFragment(graph);
  prog.add(branch_prog);
}

void IfOpx::grow(poplar::program::Sequence &prog) const {
  auto &ifop = getOp<IfOp>();

  poplar::program::Sequence then_prog;
  poplar::program::Sequence else_prog;

  copyInputs(then_prog, ifop.getThenGraph(), ifop.getThenInputIds());
  copyInputs(else_prog, ifop.getElseGraph(), ifop.getElseInputIds());

  callBranch(then_prog, ifop.getThenGraph());
  callBranch(else_prog, ifop.getElseGraph());

  auto outputs = prepareOutputs();

  copyOutputs(then_prog, ifop.getThenGraph(), ifop.getThenOutputIds(), outputs);
  copyOutputs(else_prog, ifop.getElseGraph(), ifop.getElseOutputIds(), outputs);

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
