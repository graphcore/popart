// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <poprithms/logging/timepartitionlogger.hpp>
#include <popart/chains.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/patterns/inplace.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/pbwrap.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/util.hpp>

namespace popart {

ExternOpTensorBundle::ExternOpTensorBundle(Op *opCopy,
                                           std::unique_ptr<Op> opNew)
    : up_op(std::move(opNew)) {

  // dummy inputs
  for (auto &index_tensor : opCopy->input->tensorMap()) {
    std::unique_ptr<Tensor> up_t_clone =
        index_tensor.second->clone(index_tensor.second->getGraph());
    Tensor *t_clone = up_t_clone.get();
    t_clone->id =
        opCopy->getGraph().getIr().createIntermediateTensorId(t_clone->id);
    if (tensors.find(t_clone->id) != tensors.end()) {
      throw internal_error(
          "Trying to add an input tensor that is already in tensors {}",
          t_clone->id);
    }
    tensors[t_clone->id] = std::move(up_t_clone);
    up_op->input->insert(index_tensor.first, t_clone);
    t_clone->consumers.increment(up_op.get());
  }

  // dummy outputs
  for (auto &index_tensor : opCopy->output->tensorMap()) {
    std::unique_ptr<Tensor> up_t_clone =
        index_tensor.second->clone(index_tensor.second->getGraph());
    Tensor *t_clone = up_t_clone.get();
    if (tensors.find(t_clone->id) != tensors.end()) {
      throw internal_error(
          "Trying to add an output tensor that is already in tensors {}",
          t_clone->id);
    }
    tensors[t_clone->id] = std::move(up_t_clone);
    up_op->output->insert(index_tensor.first, t_clone);
    t_clone->setProducer(up_op.get());
  }

  up_op->setup();
}

ExternOpTensorBundle::~ExternOpTensorBundle() = default;

Op *ExternOpTensorBundle::getOp() { return up_op.get(); }

Inplace::Inplace() : Pattern() {}

// what is touched? The output, and all the inputs at the target indices
std::vector<const Tensor *> Inplace::touches(Op *op, OperatorIdentifier) const {

  // Should reconsider this. If we ensure that all returns
  // to host will be done after all inplace consumers of a tensor have
  // run, we can set this up such that the output tensor is not
  // touched (where the defn of touched would then be slightly different)

  std::vector<const Tensor *> touched;
  touched.reserve(op->input->n() + 1);
  touched.push_back(op->output->tensor(0));
  // TODO : it is actually a sub-set of the inputs, only those aliased (T7108)
  for (auto &x : op->input->indicesMap()) {
    touched.push_back(x.first);
  }
  return touched;
}

bool Inplace::apply(Op *op,
                    OperatorIdentifier identifier,
                    const OpsBeforeKey &newConsIn) const {

  auto scopedStopwatch =
      op->getIr().timePartitionLogger().scopedStopwatch("Inplace::apply");
  auto output_tensor = op->output->tensor(0);
  auto &graph        = op->getGraph();

  auto newCons = newConsIn;

  // it would be nice to use "makeReplacementOpInIr" but Inplace
  // Op constructors don't have the required signature for that
  std::unique_ptr<Op> up_inplaceOp = op->getInplaceVariant(identifier);
  Op *inplaceOp                    = up_inplaceOp.get();

  transferBaseProperties(op, inplaceOp);
  inplaceOp->setName(getReplacementOpName(op, ""));
  graph.moveIntoGraph(std::move(up_inplaceOp));

  // replace op with inplaceOp everywhere in newCons
  if (newCons.find(op) != newCons.end()) {
    newCons[inplaceOp] = newCons[op];
    newCons.erase(op);
  }
  for (auto &key_vals : newCons) {
    auto &vals = key_vals.second;
    for (auto &v : vals) {
      if (v == op) {
        v = inplaceOp;
      }
    }
  }

  // Remap the tensors from `op` to `inplaceOp`
  for (auto index_tensor : op->input->tensorMap()) {
    Tensor *in_tensor = index_tensor.second;
    InIndex in_index  = index_tensor.first;
    in_tensor->consumers.increment(inplaceOp);
    graph.topoCons->transfer(op, inplaceOp);
    in_tensor->consumers.decrement(op);
    inplaceOp->input->insert(in_index, in_tensor);
  }
  output_tensor->resetProducer(inplaceOp);
  inplaceOp->output->insert(0, output_tensor);
  inplaceOp->setup();

  graph.topoCons->insert(newCons);

  logging::pattern::debug("InplaceAll::apply : replaced {}({}) with {}({})",
                          op->id,
                          op->opid,
                          inplaceOp->id,
                          inplaceOp->opid);

  inplaceOp->getGraph().eraseOp(op->id);

  logging::pattern::trace(
      "Call to InplaceAll::apply is complete, returning true.");
  return true;
}

namespace {
static AddPatternName<Inplace> registerName("InPlace");
} // namespace
} // namespace popart
