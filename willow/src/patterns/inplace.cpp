#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/patterns/inplace.hpp>
#include <poponnx/pbwrap.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/util.hpp>

namespace willow {

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
  return {op->input.tensor(0), op->output.tensor(0)};
}

bool Inplace0::apply(Op *op) const {

  // (1) create the inplace replacement for op
  std::unique_ptr<Op> up_inplaceOp = op->getInplaceVariant(0);
  Op *inplaceOp                    = up_inplaceOp.get();
  OpId inplaceId = op->pir->moveIntoIr(std::move(up_inplaceOp));

  // (2) connect the inputs of op to inplaceOp
  op->pir->connectInputsFromInputMapWrapper(
      InputMapWrapper(op->input.tensorIdMap()), inplaceId);

  for (auto index_tensor : op->input.tensorMap()) {
    Tensor *in_tensor = index_tensor.second;

    // (3) all tensors which have op as a consumer,
    //     transfer topo cons involving op to inplaceOp
    in_tensor->consumers.takeTopoCons(op, inplaceOp);

    // (4) all tensors which have op as a consumer,
    //     disconnect op as a consumer (replaced with inplaceOp)
    in_tensor->consumers.decrement(op);
  }

  // The tensor which will be updated in-place:
  Tensor *in0 = inplaceOp->input.tensor(0);

  // The tensor which will be removed:
  Tensor *out = op->output.tensor(0);

  // (5) Make sure that all consumers of in0 run before inplaceOp
  in0->consumers.setTopoLast(inplaceOp);

  // (6,7) connect the consumers of out to in0,
  //       both consumers_m and topoCons.
  //       Note that we are guaranteed at this point
  //       to not have an Op which consumes both Tensors,
  //       as if that were the case we could not perform the
  //       inplacing (not schedulable). This function also resets
  //       the inputs of the ops to in0.
  in0->consumers.takeFrom(out->consumers);

  // (8) add the constraint that all ops which were
  //     consumers of out run after inplaceOp
  for (auto &after : out->consumers.getOps()) {
    in0->consumers.insertTopoCon(inplaceOp, after);
  }

  inplaceOp->pir->eraseOp(op->id);
  inplaceOp->pir->getTensors().remove(out->id);

  return true;
}

bool Inplace0::matches(Op *op) const {

  if (!op->input.hasIndex(0)) {
    return false;
  }

  if (!op->output.hasIndex(0)) {
    return false;
  }

  if (!op->hasInplaceVariant(0)) {
    return false;
  }

  // the tensor which we're proposing
  // to perform an in-place modification on
  const Tensor *t_inplace = op->input.tensor(0);

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

} // namespace willow
