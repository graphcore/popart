#include <poponnx/graph.hpp>
#include <poponnx/op.hpp>
#include <poponnx/patterns/fuser.hpp>
#include <poponnx/pbwrap.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>
#include <poponnx/tensors.hpp>

namespace poponnx {

bool Fuser::apply(Op *op) const {

  Graph &graph = op->getGraph();

  Op *op0      = op;
  Tensor *out0 = op0->output->tensor(0);
  // we have checked that out0 only has 1 consumer
  Op *op1 = out0->consumers.getOps()[0];

  // create the replacement op01, connect it to
  // - the inputs of op0
  // - the outputs of op1
  OpId id01 = moveMergedIntoIr(op);
  Op *op01  = graph.getOp(id01);

  // wire-up the inputs.
  // 1) connect the inputs of op0 to op01
  graph.connectInputsFromInputMapWrapper(
      InputMapWrapper(op0->input->tensorIdMap()), id01);
  // 2) disconnect the inputs of op0 from op0
  for (auto index_tensor : op0->input->tensorMap()) {
    Tensor *in0 = index_tensor.second;
    in0->consumers.decrement(op0);
  }

  // we can't use connectOutputs, as that expects
  // that the output Tensor doesn't exist and must
  // be created. We rewire outputs manually:
  for (auto index_tensor : op1->output->tensorMap()) {
    OutIndex index = index_tensor.first;
    Tensor *tensor = index_tensor.second;
    op01->output->insert(index, tensor);
    tensor->resetProducer(op01);
  }

  // remove the tensor and nodes
  for (auto index_tensor : op1->input->tensorMap()) {
    // InIndex index = index_tensor.first;
    Tensor *tensor = index_tensor.second;
    tensor->consumers.decrement(op1);
  }
  graph.eraseOp(op1->id);

  graph.getTensors().remove(out0->id);
  graph.eraseOp(op0->id);

  return true;
}

bool Fuser::matches(Op *op0) const {
  if (op0->opid == get0()) {
    const Tensor *out0 = op0->output->tensor(0);
    // out0 must be consumed just once
    if (out0->consumers.getTotal() == 1) {
      Op *op1 = out0->consumers.getOps()[0];
      if (op1->opid == get1()) {

        // The fused ops must be on the same IPU
        if (op0->getOptionalVirtualGraphId() ==
            op1->getOptionalVirtualGraphId()) {
          return true;
        }
      }
    }
  }
  return false;
}

std::vector<const Tensor *> Fuser::touches(Op *op) const {
  return {op->output->tensor(0)};
}

} // namespace poponnx
