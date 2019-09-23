#include <popart/graph.hpp>
#include <popart/op/call.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/callx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

namespace popart {
namespace popx {

CallOpx::CallOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<CallOp>(op, Onnx::CustomOperators::Call);
}

ICreatorCandidatePtr CallOpx::getCreator(InIndex index) const {
  auto &callop    = getOp<CallOp>();
  auto &callgraph = callop.getCalledGraph();
  auto tensor_id  = callgraph.getInputId(index);
  auto tensor     = callgraph.getTensors().get(tensor_id);

  return dv_p->getTensorCreator(tensor);
}

poplar::Tensor CallOpx::createInput(InIndex index,
                                    const std::string &name) const {
  auto creator = getCreator(index);
  return creator->createInput(name);
}

InputCreatorType CallOpx::getInputCreatorType(int index) const {
  auto creator = getCreator(index);
  if (creator) {
    return InputCreatorType::CANCREATE;
  } else {
    return InputCreatorType::DEADEND;
  }
}

bool CallOpx::createsEquiv(int index0, const Opx *opx1, int index1) const {
  // If opx1 is a CallOpx, delegate to opx1->getCreator getCreator
  // while loop handles +1 depths of CallOpxs

  ICreatorCandidatePtr c2;
  if (opx1->op_p->opid != Onnx::CustomOperators::Call) {
    c2 = dv_p->getTensorCreator(opx1->inTensor(index1));
  } else {
    while (opx1->op_p->opid == Onnx::CustomOperators::Call) {
      c2     = dynamic_cast<const CallOpx *>(opx1)->getCreator(index1);
      opx1   = c2->getOpx();
      index1 = c2->getIndex();
    }
  }

  // pass responsibility to creator
  auto creator = getCreator(index0);
  return creator->createsEquivalent(c2);
}

std::vector<TensorId> CallOpx::mustExistBeforeCreate(int index) const {
  auto creator = getCreator(index);
  return creator->mustExistBeforeCreate();
}

std::vector<poplar::Tensor> CallOpx::prepareOutputs() const {
  std::vector<poplar::Tensor> outputs;
  auto &callop = getOp<CallOp>();

  for (auto out_id : callop.getCalledGraph().getOutputIds()) {
    auto output = get(out_id);
    outputs.push_back(graph().clone(output));
  }

  return outputs;
}

void CallOpx::copyModified(poplar::program::Sequence &prog) const {
  auto &callop = getOp<CallOp>();

  for (int i = 0; i < callop.input->n(); i++) {
    if (callop.isInputModified(i)) {
      auto call_input  = get(callop.inId(i));
      auto graph_input = get(callop.getCalledGraph().getInputId(i));

      poplar::program::Copy copy_prog(graph_input, call_input);
      prog.add(copy_prog);
    }
  }
}

void CallOpx::copyInputs(poplar::program::Sequence &prog) const {
  auto &callop = getOp<CallOp>();

  for (int i = 0; i < callop.input->n(); i++) {
    auto call_input     = get(callop.inId(i));
    auto graph_input_id = callop.getCalledGraph().getInputId(i);
    auto graph_input    = get(graph_input_id);
    poplar::program::Copy copy_prog(call_input, graph_input);
    prog.add(copy_prog);
  }
}

void CallOpx::copyOutputs(poplar::program::Sequence &prog,
                          const std::vector<poplar::Tensor> &outputs) const {
  auto &callop = getOp<CallOp>();

  for (int i = 0; i < outputs.size(); i++) {
    auto &call_output = outputs.at(i);
    auto graph_output = get(callop.getCalledGraph().getOutputId(i));

    poplar::program::Copy copy_prog(graph_output, call_output);
    prog.add(copy_prog);
  }
}

void CallOpx::doCall(poplar::program::Sequence &prog) const {
  auto &callop       = getOp<CallOp>();
  auto &called_graph = callop.getCalledGraph();
  auto &graph_prog   = dv_p->progs.scopeFragment(called_graph);
  prog.add(graph_prog);
}

void CallOpx::grow(poplar::program::Sequence &prog) const {
  copyInputs(prog);
  doCall(prog);
  auto outputs = prepareOutputs();
  copyOutputs(prog, outputs);
  copyModified(prog);
  for (int i = 0; i < outputs.size(); i++) {
    setOutTensor(i, outputs.at(i));
  }
}

namespace {
OpxCreator<CallOpx> callOpxCreator(Onnx::CustomOperators::Call);
}

} // namespace popx
} // namespace popart
