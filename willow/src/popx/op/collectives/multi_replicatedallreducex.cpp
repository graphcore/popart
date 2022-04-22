// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <gcl/Collectives.hpp>
#include <memory>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/collectives/multi_replicatedallreducex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/collectives/multi_replicatedallreduce.hpp"
#include "popart/popx/op/collectives/collectivesx.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/region.hpp" // IWYU pragma: keep
#include "popart/tensorindex.hpp"

namespace popart {

namespace popx {

MultiReplicatedAllReduceOpx::MultiReplicatedAllReduceOpx(popart::Op *op,
                                                         Devicex *devicex)
    : MultiCollectiveBaseOpx(op, devicex) {
  verifyOp<MultiReplicatedAllReduceOp>(
      op, Onnx::CustomOperators::MultiReplicatedAllReduce);
}

InputCreatorType
MultiReplicatedAllReduceOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CanUnwind;
}

snap::Tensor
MultiReplicatedAllReduceOpx::unwindTensorLayout(snap::Tensor tensor,
                                                InIndex in,
                                                OutIndex out) const {
  if (in == out) {
    return tensor;
  } else {
    throw error("[MultiReplicatedAllReduceOpx::unwindTensorLayout] Unexpected "
                "input output pair {}->{}.",
                in,
                out);
  }
}

view::RegMap MultiReplicatedAllReduceOpx::unwindRegion(InIndex,
                                                       OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

// Prepare the output tensors
void MultiReplicatedAllReduceOpx::growPart(OpxGrowPartId id) const {
  logging::opx::debug("[MultiReplicatedAllReduceOpx::growPart] part {}/{}",
                      id,
                      op_p->output->n());
  snap::Tensor inputTensor = getInTensor(id);
  // The input tensor can be a mix of inplace and outplace tensors
  // 1. for inplace tensors, the input tensor is also the output tensor
  // 2. for outplace tensors we constuct a copy of the input tensor
  //    (the copy is added inside of grow)
  poplar::Tensor outputTensor;
  if (op_p->modifiesIndex(id)) {
    if (inputTensor.isParallelWriteable()) {
      outputTensor = inputTensor.getPoplarTensor();
    } else {
      throw error("[MultiReplicatedAllReduceOpx::growPart] Tensor {} was marked"
                  " for inplacing but is not writeable.",
                  inputTensor.getPoplarTensor().getDebugStr());
    }
  } else {
    outputTensor =
        inGraph(id).getPoplarGraph().clone(inputTensor.getPoplarTensor());
  }
  if (hasInViewChangers(id)) {
    setOutViewChangers(id, getInViewChangers(id));
  }
  setOutTensor(id, snap::Tensor(outputTensor, outGraph(id)));
}

void MultiReplicatedAllReduceOpx::grow(snap::program::Sequence &prog) const {
  logging::opx::debug("[MultiReplicatedAllReduceOpx::grow] Growing  "
                      "MultiReplicatedAllReduceOpx");
  MultiReplicatedAllReduceOp &op = getOp<MultiReplicatedAllReduceOp>();

  // Fill vector of inputs.
  // some of these
  std::vector<poplar::Tensor> inputs;
  std::vector<poplar::Tensor> src;
  std::vector<poplar::Tensor> dst;
  for (InIndex i = 0; i < op.input->n(); ++i) {
    auto t = getOutTensor(i).flatten().getPoplarTensor();
    inputs.emplace_back(getOutTensor(i).flatten().getPoplarTensor());

    if (!op_p->modifiesIndex(i)) {
      src.emplace_back(getInTensor(i).flatten().getPoplarTensor());
      dst.emplace_back(getOutTensor(i).flatten().getPoplarTensor());
    }
  }

  if (src.size() > 0) {
    // Copy all outplace tensors in one program
    prog.getPoplarSequence().add(
        poplar::program::Copy(poplar::concat(src), poplar::concat(dst), false));
  }

  // Call gcl
  poplar::Tensor data = poplar::concat(inputs);
  if (!data.isParallelWriteable()) {
    throw error("[MultiReplicatedAllReduceOpx::grow] The data tensor for the "
                " in place allreduce collective is not writeable");
  }
  gcl::allReduceInPlaceCrossReplica(
      dv_p->lowering().graph().getPoplarGraph(),
      data,
      getPoplarCollectiveOperator(op.getCollectiveOp()),
      prog.getPoplarSequence(),
      toGCLCommGroup(op.getGCLCommGroup()),
      "MultiAllReduce",
      dv_p->lowering().gclOptions);
}

namespace {
OpxCreator<MultiReplicatedAllReduceOpx> MultiReplicatedAllReduceOpxCreator(
    Onnx::CustomOperators::MultiReplicatedAllReduce);
}
} // namespace popx
} // namespace popart
