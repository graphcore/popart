#include <poponnx/ir.hpp>
#include <poponnx/op.hpp>
#include <poponnx/patterns/inplace.hpp>
#include <poponnx/pbwrap.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/topocons.hpp>
#include <poponnx/util.hpp>

namespace poponnx {

// Example 1:
// Consider the following SERIES of non-linearity ops:
// (0) -> [relu] -> (1) -> [sigmoid] -> (2) -> [exp] -> (3) -> [other]
// 1.1: In-placing relu:
//      (0) -> [relu-inplace]
//      (0) -> [sigmoid] -> (2) -> [exp] -> (3) -> [other]
//      with dependencies:
//      [relu-inplace] before [sigmoid]
// 1.2: In-placing sigmoid:
//      (0) -> [relu-inplace]
//      (0) -> [sigmoid-inplace]
//      (0) -> [exp] -> (3) -> [other]
//      with dependencies:
//      [relu-inplace] before [sigmoid-inplace]
//      [sigmoid-inplace] before [exp]
// 1.3: In-placing exp:
//      (0) -> [relu-inplace]
//      (0) -> [sigmoid-inplace]
//      (0) -> [exp-inplace]
//      (0)->  [other]
//      with dependencies:
//      [relu-inplace] before [sigmoid-inplace]
//      [sigmoid-inplace] before [exp-inplace]
//      [exp-inplace] before [other].
// All in-placed :)
//
// Example 2:
// Consider the following non-linearity ops in PARALLEL:
// (0) -> [relu] -> (1) -> [other]
// (0) -> [sigmoid] -> (2)
// (0) -> [exp] -> (3)
// 2.1: In-placing relu:
//      (0) -> [relu-inplace]
//      (0) -> [other]
//      (0) -> [sigmoid] -> (2)
//      (0) -> [exp] -> (3)
//      with dependencies:
//      other after relu-inplace
//      relu-inplace after sigmoid
//      relu-inplace after exp
// Good. Now can we make sigmoid inplace? No, because then what
// would come first, relu-inplace or sigmoid-inplace. RULE:
// An Op can only be in-placed if it can run after all existing
// consumers of the tensor it in-places.
// Partial success in-placing :|

std::vector<const Tensor *> Inplace0::touches(Op *op) const {
  // Should reconsider this. If we ensure that all returns
  // to host will be done after all inplace consumers of a tensor have
  // run, we can set this up such that the output tensor is not
  // touched (where the defn of touched would then be slightly different)
  return {op->input->tensor(0), op->output->tensor(0)};
}

bool Inplace0::apply(Op *op) const {
  auto input_tensor  = op->input->tensor(0);
  auto output_tensor = op->output->tensor(0);
  auto ir            = op->pir;

  // Create the inplace op variant
  std::unique_ptr<Op> up_inplaceOp = op->getInplaceVariant(0);
  Op *inplaceOp                    = up_inplaceOp.get();
  ir->moveIntoIr(std::move(up_inplaceOp));

  // Remap the tensors from `op` to `inplaceOp`
  for (auto index_tensor : op->input->tensorMap()) {
    Tensor *in_tensor = index_tensor.second;

    in_tensor->consumers.increment(inplaceOp);
    ir->topoCons->transfer(op, inplaceOp);
    in_tensor->consumers.decrement(op);
  }

  ir->topoCons->setFinalConsumer(input_tensor, inplaceOp);
  output_tensor->resetProducer(inplaceOp);

  inplaceOp->input->insert(0, input_tensor);
  inplaceOp->output->insert(0, output_tensor);

  logging::info("Inplace0::apply : replace {}({}) with {}({})",
                op->id,
                op->opid,
                inplaceOp->id,
                inplaceOp->opid);

  ir->eraseOp(op->id);
  return true;
}

bool Inplace0::matches(Op *op) const {

  if (!op->input->hasIndex(0)) {
    return false;
  }

  if (!op->output->hasIndex(0)) {
    return false;
  }

  if (!op->hasInplaceVariant(0)) {
    return false;
  }

  // the tensor which we're proposing
  // to perform an in-place modification on
  const Tensor *t_inplace = op->input->tensor(0);

  // Consider an Op which does
  // C <- gamma*A + B
  // and inplace version,
  // C *= gamma and then C += B.
  // if A = B, then the inplace is not valid, as A <- 2*gamma*A
  // For certain ops it is fine if the input is repeated (Add),
  // but for now we will just say that this is not in-placeable.
  if (t_inplace->consumers.n(op) > 1) {
    return false;
  }

  // if it's not topologically possible to perform the proposed in-place
  // op after all current ops consuming the tensor, we cannot proceed.
  // see Example 2 above.

  // if the tensor to be inplaced only has 1 consumer, we're good
  std::vector<Op *> allConsumers = t_inplace->consumers.getOps();
  if (allConsumers.size() == 1) {
    return true;
  }

  // if we are going to in-place the tensor, these are
  // the additional topological constraints we need:
  OpsBeforeKey gCons;
  gCons[op] = {};
  for (auto before : t_inplace->consumers.getOps()) {
    if (before != op) {
      gCons[op].push_back(before);
    }
  }

  if (!op->pir->isSchedulable(gCons)) {
    return false;
  }

  return true;
}

bool InplaceAll::matches(Op *op) const {
  std::vector<InIndex> inIndices;

  for (auto pair : op->input->tensorMap()) {
    inIndices.push_back(pair.first);
  }

  if (!op->hasInplaceVariant(inIndices)) {
    return false;
  }

  for (auto pair : op->input->tensorMap()) {
    const Tensor *t = op->input->tensor(pair.first);

    if (t->consumers.n(op) > 1) {
      logging::info(
          "InplaceAll::matches : inplace candidate {} rejected due to "
          "aliasing input {}",
          op->name(),
          pair.first);
      return false;
    }

    if (t->consumers.getOps().size() != 1) {
      OpsBeforeKey gCons;
      gCons[op] = {};

      for (auto before : t->consumers.getOps()) {
        if (before != op) {
          gCons[op].push_back(before);
        }
      }

      if (!op->pir->isSchedulable(gCons)) {
        logging::pattern::debug(
            "InplaceAll::matches : inplace candidate {} rejected due to "
            "schedulable conflict",
            op->name());
        return false;
      }
    }
  }

  return true;
}

std::vector<const Tensor *> InplaceAll::touches(Op *op) const {
  std::vector<const Tensor *> result = {op->output->tensor(0)};
  result.reserve(op->input->n() + 1);

  for (auto pair : op->input->tensorMap()) {
    const Tensor *t = op->input->tensor(pair.first);

    result.push_back(t);
  }

  return result;
}

bool InplaceAll::apply(Op *op) const {
  auto output_tensor = op->output->tensor(0);
  auto ir            = op->pir;

  std::vector<InIndex> inIndices;
  for (auto pair : op->input->tensorMap()) {
    inIndices.push_back(pair.first);
  }

  // Create the inplace op variant
  std::unique_ptr<Op> up_inplaceOp = op->getInplaceVariant(inIndices);
  Op *inplaceOp                    = up_inplaceOp.get();
  ir->moveIntoIr(std::move(up_inplaceOp));

  // Remap the tensors from `op` to `inplaceOp`
  for (auto index : inIndices) {
    Tensor *in_tensor = op->input->tensor(index);
    in_tensor->consumers.increment(inplaceOp);
    ir->topoCons->transfer(op, inplaceOp);
    in_tensor->consumers.decrement(op);
  }

  output_tensor->resetProducer(inplaceOp);
  inplaceOp->output->insert(0, output_tensor);

  for (auto index : inIndices) {
    Tensor *input_tensor = op->input->tensor(index);
    ir->topoCons->setFinalConsumer(input_tensor, inplaceOp);
    inplaceOp->input->insert(index, input_tensor);
  }

  logging::pattern::debug("InplaceAll::apply : replace {}({}) with {}({})",
                          op->id,
                          op->opid,
                          inplaceOp->id,
                          inplaceOp->opid);

  inplaceOp->pir->eraseOp(op->id);

  return true;
}

namespace {
static PatternCreator<Inplace0> inplace0Pattern(PatternType::INPLACE0,
                                                "InPlace0");
static PatternCreator<InplaceAll> inplaceAllPattern(PatternType::INPLACEALL,
                                                    "InPlaceAll");
} // namespace

} // namespace poponnx
