// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <popops/Zero.hpp>
#include <popart/graph.hpp>
#include <popart/op/if.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/ifx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

#include "popart/error.hpp"
#include "popart/graphid.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/opx.hpp"
#include "popart/popx/popprograms.hpp"
#include "popart/popx/preparedtensor.hpp"
#include "popart/tensor.hpp"

namespace popart {
class Op;

namespace popx {

void IfOpx::copyInputs(poplar::program::Sequence &thenProg,
                       poplar::program::Sequence &elseProg) const {
  auto &ifop      = getOp<IfOp>();
  auto &thenGraph = ifop.getThenGraph();
  auto &elseGraph = ifop.getElseGraph();

  auto copyInput = [&](poplar::program::Sequence &prog,
                       const Graph &graph,
                       InIndex ifopInputIndex,
                       InIndex branchInputIndex) {
    auto ifInputId     = inId(ifopInputIndex);
    auto branchInputId = graph.getInputId(branchInputIndex);

    auto ifInput     = get(ifInputId);
    auto branchInput = get(branchInputId);

    poplar::program::Copy copyProg(ifInput, branchInput, false, debugContext());
    prog.add(copyProg);
  };

  auto copyBranchInputs = [&](poplar::program::Sequence &prog,
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

void IfOpx::callBranch(poplar::program::Sequence &prog,
                       const Graph &graph) const {
  auto &branch_progs = dv_p->lowering().progs().scopeFragments(graph);
  for (auto branch_prog : branch_progs) {
    prog.add(branch_prog);
  }
}

void IfOpx::copyOutputs(poplar::program::Sequence &thenProg,
                        poplar::program::Sequence &elseProg,
                        const std::vector<poplar::Tensor> &outputs) const {
  auto &ifop      = getOp<IfOp>();
  auto &thenGraph = ifop.getThenGraph();
  auto &elseGraph = ifop.getElseGraph();

  auto copyOutput = [&](poplar::program::Sequence &prog,
                        const Graph &graph,
                        OutIndex opIndex,
                        OutIndex branchIndex) {
    auto opId     = outId(opIndex);
    auto branchId = graph.getOutputId(branchIndex);

    auto opOutput     = outputs.at(opIndex);
    auto branchOutput = get(branchId);
    poplar::program::Copy copyProg(
        branchOutput, opOutput, false, debugContext());
    prog.add(copyProg);
  };

  auto zeroOutput = [&](poplar::program::Sequence &prog, OutIndex opIndex) {
    auto opId     = outId(opIndex);
    auto opOutput = outputs.at(opIndex);
    popops::zero(graph(), opOutput, prog, debugContext("zero"));
  };

  auto copyOrZeroBranchOutput =
      [&](poplar::program::Sequence &prog, const Graph &graph, int outIndex) {
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

std::vector<poplar::Tensor> IfOpx::prepareOutputs() const {
  std::vector<poplar::Tensor> outputs;
  auto &ifop = getOp<IfOp>();

  auto cloneOutput = [&](const Graph &branch, OutIndex branchIndex) {
    auto branchId     = branch.getOutputId(branchIndex);
    auto branchOutput = get(branchId);
    outputs.push_back(graph().clone(branchOutput));
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

IfOpx::IfOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<IfOp>(op);
}

void IfOpx::grow(poplar::program::Sequence &prog) const {
  auto &ifop = getOp<IfOp>();

  auto thenDbgStr = logging::format("{}/then", ifop.getThenGraph().id);
  auto elseDbgStr = logging::format("{}/else", ifop.getElseGraph().id);
  poplar::program::Sequence then_prog(debugContext(thenDbgStr));
  poplar::program::Sequence else_prog(debugContext(elseDbgStr));

  copyInputs(then_prog, else_prog);

  callBranch(then_prog, ifop.getThenGraph());
  callBranch(else_prog, ifop.getElseGraph());

  auto outputs = prepareOutputs();

  copyOutputs(then_prog, else_prog, outputs);

  auto condition = getInTensor(IfGradOp::getConditionInIndex());

  // Reshape to scalar in case the user passed in tensor of shape [1]
  condition = condition.reshape({});
  prog.add(poplar::program::If(
      condition, then_prog, else_prog, debugContext("condition")));

  for (int i = 0; i < outputs.size(); i++) {
    setOutTensor(i, outputs.at(i));
  }
}

IfGradOpx::IfGradOpx(Op *op, Devicex *devicex) : IfOpx(op, devicex) {
  verifyOp<IfGradOp>(op, Onnx::CustomGradOperators::IfGrad);
}

PreparedTensorInfos IfOpx::getInputsToPrepare() const {
  auto &ifop = getOp<IfOp>();

  PreparedTensorInfos inputs;

  for (auto graph : ifop.getCalledGraphs()) {
    auto &idxMap = ifop.getBranchInIndicesMap(*graph);
    for (auto &kv : idxMap) {
      inputs.emplace_back(ifop.input->tensor(kv.first)->id,
                          graph->getInputId(kv.second),
                          CanAlias::No,
                          RequireParallelWritable::Yes);
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
