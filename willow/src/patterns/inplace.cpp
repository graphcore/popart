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

// what is touched? The output, and all the inputs at the target indices
std::vector<const Tensor *> Inplace::touches(Op *op) const {

  // Should reconsider this. If we ensure that all returns
  // to host will be done after all inplace consumers of a tensor have
  // run, we can set this up such that the output tensor is not
  // touched (where the defn of touched would then be slightly different)

  std::vector<const Tensor *> touched = {op->output->tensor(0)};
  auto inIndices                      = targetInIndices(op);
  touched.reserve(inIndices.size() + 1);
  for (auto index : inIndices) {
    touched.push_back(op->input->tensor(index));
  }
  return touched;
}

bool Inplace::apply(Op *op) const {
  auto output_tensor             = op->output->tensor(0);
  auto ir                        = op->pir;
  std::vector<InIndex> inIndices = targetInIndices(op);

  if (op->inplaceVariants(inIndices).size() == 0) {
    throw error("Cannot call Inplace::apply for {} as no good variants",
                op->str());
  }

  // Create the inplace op variant, using the first one for now TODO
  auto identifier = op->inplaceVariants(inIndices)[0];
  std::unique_ptr<Op> up_inplaceOp =
      op->getInplaceVariant(identifier, inIndices);

  Op *inplaceOp = up_inplaceOp.get();
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
    auto newCons = ir->topoCons->finalConsumerCons(input_tensor, inplaceOp);
    ir->topoCons->insert(newCons);
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

bool Inplace::matches(Op *op) const {
  std::vector<InIndex> inIndices = targetInIndices(op);

  if (op->inplaceVariants(inIndices).size() == 0) {
    return false;
  }

  if (!op->output->hasIndex(0)) {
    return false;
  }

  // if it's not topologically possible to perform the proposed in-place
  // op after all current ops consuming the tensor, we cannot proceed.
  // see Example 2 above.
  for (InIndex index : inIndices) {

    const Tensor *t = op->input->tensor(index);

    // Consider an Op which does
    // C <- gamma*A + B
    // and inplace version,
    // C *= gamma and then C += B.
    // if A = B, then the inplace is not valid, as A <- 2*gamma*A
    // For certain ops it is fine if the input is repeated (Add),
    // but for now we will just say that this is not in-placeable.
    if (t->consumers.n(op) > 1) {
      logging::info(
          "InplaceAll::matches : inplace candidate {} rejected due to "
          "aliasing input {}",
          op->name(),
          t->str());
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

      // we here are using the fact that if
      // 1) is a sorting with A->C
      // 2) is a sorting with B->C
      // then there is either a sorting with A->B->C or one with B->A->C

      if (!op->pir->isSchedulable(gCons)) {
        logging::pattern::debug(
            "InplaceAll::matches : inplace candidate {} rejected due to "
            "scheduling conflict",
            op->name());
        return false;
      }
    }
  }

  return true;
}

namespace {
static PatternCreator<Inplace0> inplace0Pattern(PatternType::INPLACE0,
                                                "InPlace0");
static PatternCreator<InplaceAll> inplaceAllPattern(PatternType::INPLACEALL,
                                                    "InPlaceAll");
} // namespace

} // namespace poponnx
