#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/op/identity.hpp>
#include <poponnx/op/pad.hpp>
#include <poponnx/patterns/postnrepl.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

// (see .hpp for ascii picture definitions)
bool PostNRepl::apply(Op *op) const {

  Ir *pir = op->pir;

  // op is [*]
  Tensor *ori = op->input->tensor(0);

  std::vector<Tensor *> replicates;
  // setting replicates (rep1), (rep2), (rep3)
  for (auto &ind_t : op->output->tensorMap()) {
    replicates.push_back(ind_t.second);
  }

  for (auto t_repl : replicates) {
    // for rep1 : {[op0], [op2]}
    for (Op *op_z : t_repl->consumers.getOps()) {
      // at what indices is (rep1) consumed?
      for (int index : op_z->input->indices(t_repl)) {
        // must rather consume ori
        op_z->input->reset(index, ori);
      }
    }
    // ori is consumed by all consumers of t_repl
    // (this is the same wiring as above, always needs
    // to be done for tensor and op)
    ori->consumers.extend(t_repl->consumers.getMap());
  }
  ori->consumers.decrement(op);
  // delete replicates
  for (auto repl : replicates) {
    pir->getTensors().remove(repl->id);
  }

  // delete [*]
  pir->eraseOp(op->id);

  return true;
}

bool PostNRepl::matches(Op *op) const {

  // The Identity op fits the criteria of PostNRepl: An Op which replicates its
  // input N times (where N = 1 for identity)
  if (op->isConvertibleTo<IdentityOp>()) {
    // good so far
  }
  // A sum with only one input
  else if (op->opType == OpType::SUM && op->input->n() == 1) {
    // good so far
  }
  // A pad with zero-padding
  else if (op->opType == OpType::PAD &&
           dynamic_cast<const PadOp *>(op)->padSizeZero()) {
    // good so far
  } else {
    // doesn't match a known case of PostNRepl
    return false;
  }

  // we have a viable match
  return true;
}

// touches all the outputs of the root op [*] from the Ir,
// and maybe the input to the root op.
std::vector<const Tensor *> PostNRepl::touches(Op *op) const {
  std::vector<const Tensor *> outs;
  for (auto &t_inds : op->output->indicesMap()) {
    outs.push_back(t_inds.first);
  }
  return outs;
}

namespace {
static PatternCreator<PostNRepl> PostNReplPattern(PatternType::POSTNREPL,
                                                  "PostNRepl");
}

} // namespace poponnx
