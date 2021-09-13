// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popops/Zero.hpp>

#include <popart/graph.hpp>
#include <popart/op/if.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/ifx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

namespace popart {
namespace popx {

void IfOpx::copyInputs(snap::program::Sequence &thenProg,
                       snap::program::Sequence &elseProg) const {
  auto &ifop      = getOp<IfOp>();
  auto &thenGraph = ifop.getThenGraph();
  auto &elseGraph = ifop.getElseGraph();

  auto copyInput = [&](snap::program::Sequence &prog,
                       const Graph &graph,
                       InIndex ifopInputIndex,
                       InIndex branchInputIndex) {
    auto ifInputId     = inId(ifopInputIndex);
    auto branchInputId = graph.getInputId(branchInputIndex);

    auto ifInput     = get(ifInputId).getPoplarTensor();
    auto branchInput = get(branchInputId).getPoplarTensor();

    poplar::program::Copy copyProg(ifInput, branchInput, false, debugContext());
    prog.add(copyProg);
  };

  auto copyBranchInputs = [&](snap::program::Sequence &prog,
                              const Graph &graph) {
    auto &idxMap = ifop.getBranchInIndicesMap(graph);
    for (auto &opIdx_branchIdx : idxMap) {
      auto opIdx     = opIdx_branchIdx.first;
      auto branchIdx = opIdx_branchIdx.second;
      copyInput(prog, graph, opIdx, branchIdx);
    }
  };

  copyBranchInputs(thenProg, thenGraph);
  copyBranchInputs(elseProg, elseGraph);
}

void IfOpx::callBranch(snap::program::Sequence &prog,
                       const Graph &graph) const {
  auto &branch_progs = dv_p->lowering().progs.scopeFragments(graph);
  for (auto branch_prog : branch_progs) {
    prog.add(branch_prog);
  }
}

void IfOpx::copyOutputs(snap::program::Sequence &thenProg,
                        snap::program::Sequence &elseProg,
                        const std::vector<snap::Tensor> &outputs) const {
  auto &ifop      = getOp<IfOp>();
  auto &thenGraph = ifop.getThenGraph();
  auto &elseGraph = ifop.getElseGraph();

  auto copyOutput = [&](snap::program::Sequence &prog,
                        const Graph &graph,
                        OutIndex opIndex,
                        OutIndex branchIndex) {
    auto opId     = outId(opIndex);
    auto branchId = graph.getOutputId(branchIndex);

    auto opOutput     = outputs.at(opIndex);
    auto branchOutput = get(branchId).getPoplarTensor();
    poplar::program::Copy copyProg(
        branchOutput, opOutput.getPoplarTensor(), false, debugContext());
    prog.add(copyProg);
  };

  auto zeroOutput = [&](snap::program::Sequence &prog, OutIndex opIndex) {
    auto opId     = outId(opIndex);
    auto opOutput = outputs.at(opIndex);
    popops::zero(graph().getPoplarGraph(),
                 opOutput.getPoplarTensor(),
                 prog.getPoplarSequence(),
                 debugContext("zero"));
  };

  auto copyOrZeroBranchOutput =
      [&](snap::program::Sequence &prog, const Graph &graph, int outIndex) {
        auto &idxMap = ifop.getBranchOutIndicesMap(graph);
        auto found   = idxMap.find(outIndex);
        if (found != idxMap.end()) {
          copyOutput(prog, graph, outIndex, found->second);
        } else {
          zeroOutput(prog, outIndex);
        }
      };

  for (int i = 0; i < ifop.output->n(); i++) {
    copyOrZeroBranchOutput(thenProg, thenGraph, i);
    copyOrZeroBranchOutput(elseProg, elseGraph, i);
  }
}

std::vector<snap::Tensor> IfOpx::prepareOutputs() const {
  std::vector<snap::Tensor> outputs;
  auto &ifop = getOp<IfOp>();

  auto cloneOutput = [&](const Graph &branch, OutIndex branchIndex) {
    auto branchId     = branch.getOutputId(branchIndex);
    auto branchOutput = get(branchId).getPoplarTensor();
    outputs.push_back(
        snap::Tensor{graph().getPoplarGraph().clone(branchOutput), graph()});
  };

  auto tryCloneOutputFromBranch = [&](const Graph &graph, int outIndex) {
    auto &idxMap = ifop.getBranchOutIndicesMap(graph);
    auto found   = idxMap.find(outIndex);
    if (found != idxMap.end()) {
      cloneOutput(graph, found->second);
      return true;
    } else {
      return false;
    }
  };

  for (int i = 0; i < ifop.output->n(); i++) {
    if (!tryCloneOutputFromBranch(ifop.getThenGraph(), i) &&
        !tryCloneOutputFromBranch(ifop.getElseGraph(), i)) {
      throw error("Could not find suitable branch output to clone {} from",
                  outId(i));
    }
  }

  return outputs;
}

IfOpx::IfOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<IfOp>(op);
}

void IfOpx::grow(snap::program::Sequence &prog) const {
  auto &ifop = getOp<IfOp>();

  auto thenDbgStr = logging::format("{}/then", ifop.getThenGraph().id);
  auto elseDbgStr = logging::format("{}/else", ifop.getElseGraph().id);
  snap::program::Sequence then_prog(debugContext(thenDbgStr), graph());
  snap::program::Sequence else_prog(debugContext(elseDbgStr), graph());

  copyInputs(then_prog, else_prog);

  callBranch(then_prog, ifop.getThenGraph());
  callBranch(else_prog, ifop.getElseGraph());

  auto outputs = prepareOutputs();

  copyOutputs(then_prog, else_prog, outputs);

  poplar::Tensor condition =
      getInTensor(IfGradOp::getConditionInIndex()).getPoplarTensor();

  // Reshape to scalar in case the user passed in tensor of shape [1]
  condition = condition.reshape({});
  prog.add(poplar::program::If(condition,
                               then_prog.getPoplarSequence(),
                               else_prog.getPoplarSequence(),
                               debugContext("condition")));

  for (int i = 0; i < outputs.size(); i++) {
    setOutTensor(i, outputs.at(i));
  }
}

IfGradOpx::IfGradOpx(Op *op, Devicex *devicex) : IfOpx(op, devicex) {
  verifyOp<IfGradOp>(op, Onnx::CustomGradOperators::IfGrad);
}

std::vector<std::tuple<TensorId, TensorId, bool>>
IfOpx::getInputsToPrepare() const {
  auto &ifop = getOp<IfOp>();

  std::vector<std::tuple<TensorId, TensorId, bool>> inputs;

  for (auto graph : ifop.getCalledGraphs()) {
    auto &idxMap = ifop.getBranchInIndicesMap(*graph);
    for (auto &kv : idxMap) {
      inputs.emplace_back(ifop.input->tensor(kv.first)->id,
                          graph->getInputId(kv.second),
                          false);
    }
  }

  return inputs;
}

namespace {
OpxCreator<IfOpx> ifOpxCreator({Onnx::Operators::If_1, Onnx::Operators::If_11});
OpxCreator<IfGradOpx> ifGradOpxCreator(Onnx::CustomGradOperators::IfGrad);
} // namespace

} // namespace popx
} // namespace popart
