#include <poponnx/error.hpp>
#include <poponnx/identity.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/nll.hpp>
#include <poponnx/pad.hpp>
#include <poponnx/patterns.hpp>
#include <poponnx/pbwrap.hpp>
#include <poponnx/softmax.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/util.hpp>

namespace willow {

class SoftmaxGradDirectOp;

PatternTypes initPatternTypes() { return PatternTypes(); }

const PatternTypes &getPatternTypes() {
  const static PatternTypes X = initPatternTypes();
  return X;
}

bool Pattern::touchesAnchored(Op *op) const {
  for (auto &tensor : touches(op)) {
    if (op->pir->isAnchored(tensor->id)) {
      return true;
    }
  }
  return false;
};

PatternTypes::PatternTypes() {

  opTypes_ = {{"PostNRepl", PatternType::POSTNREPL},
              {"PreUniRepl", PatternType::PREUNIREPL},
              {"SoftmaxGradDirect", PatternType::SOFTMAXGRADDIRECT},
              {"SplitConvBias", PatternType::SPLITCONVBIAS},
              {"ReduceSumToIdentity", PatternType::REDUCESUMTOIDENTITY}};

  std::vector<std::string> opTypeKeys;
  opTypeKeys.reserve(opTypes_.size());
  for (auto &x : opTypes_) {
    strings_[x.second] = x.first;
  }
}

const PatternType &PatternTypes::get(std::string op_type) const {
  auto found = opTypes_.find(op_type);
  if (found == opTypes_.end()) {
    std::vector<std::string> opTypeNames;
    opTypeNames.reserve(opTypes_.size());
    for (auto &name_type : opTypes_) {
      opTypeNames.push_back(name_type.first);
    }
    std::stringstream errm;
    errm << "No PatternType found for " << op_type << ". Options are ";
    appendSequence(errm, opTypeNames);
    throw error(errm.str());
  }

  return found->second;
}

const std::string &PatternTypes::get(PatternType opType) const {
  return strings_.at(opType);
}

bool PreUniRepl::matches(Op *op) const {
  // op must have 1 input, and that input
  // must be consumed by only op (and only once)
  if (op->input.n() != 1) {
    return false;
  } else if (op->input.tensor(0)->consumers.getTotal() != 1) {
    return false;
  }

  // A sum with only one input
  else if (op->opType == OpType::SUM) {
    return true;
    // A pad with zero-padding
  } else if (op->opType == OpType::PAD &&
             dynamic_cast<const PadOp *>(op)->padSizeZero()) {
    return true;
  } else {
    return false;
  }
}

// NLLGRAD (0) -> x -> SOFTMAXGRAD.
OpType SoftmaxGradDirect::get0() const { return OpType::NLLGRAD; }

// NLLGRAD -> x -> SOFTMAXGRAD (1).
OpType SoftmaxGradDirect::get1() const { return OpType::SOFTMAXGRAD; }

bool FuserPattern::matches(Op *op0) const {
  if (op0->opType == get0()) {
    const Tensor *ten_d = op0->output.tensor(0);
    // Consumed just once? Should be the case
    if (ten_d->consumers.getTotal() == 1) {
      Op *op1 = ten_d->consumers.getOps()[0];
      if (op1->opType == get1()) {
        return true;
      }
    }
  }
  return false;
}

std::vector<const Tensor *> PreUniRepl::touches(Op *op) const {
  return {op->input.tensor(0)};
}

std::vector<const Tensor *> FuserPattern::touches(Op *op) const {
  return {op->output.tensor(0)};
}

// (see .hpp for ascii picture definitions)
void PreUniRepl::apply(Op *op) const {
  // op is []
  // ()
  Tensor *tensorIn = op->input.tensor(0);
  // (.)
  Tensor *tensorOut = op->output.tensor(0);
  // [.]
  auto op0 = tensorIn->getProducer();
  // (.) gets all consumers of () other than []
  tensorOut->consumers.extend(tensorIn->consumers.getMap());
  tensorOut->consumers.decrement(op);
  // [.] produces (.) directly
  int index = op0->output.indices(tensorIn)[0];
  op0->output.reset(index, tensorOut);
  tensorOut->resetProducer(op0);
  Ir *pir = op->pir;
  // delete ()
  pir->getTensors().remove(tensorIn->id); // name);
  // delete [.]
  pir->eraseOp(op->id);
}

bool PostNRepl::matches(Op *op) const {

  // The Identity op fits the criteria of PostNRepl: An Op which replicates its
  // input N times (where N = 1 for identity)
  if (op->isConvertibleTo<IdentityOp>()) {
    // good so far
  }
  // A sum with only one input
  else if (op->opType == OpType::SUM && op->input.n() == 1) {
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
  // we check that the consumer topological constraints
  // of (ori), and (rep1, rep2, rep3) can be be resolved
  // if [*] is removed.
  TopoBundle tcInf = getTopoConInfo(op);

  // merging weak constraints is impossible
  if (tcInf.nWeakTopoCons > 0) {
    return false;
  }

  // at most 1 consumer can be last
  else if (tcInf.nTopoLasts > 1) {
    return false;
  }

  // if the last consumer is op ([*] in the schematic),
  // that won't work as op is going to be removed.
  // Also, this should not be possible if this
  // is really a replicating op
  else if (tcInf.lastCon == op) {
    return false;
  }

  // we have a viable match
  return true;
}

PostNRepl::TopoBundle PostNRepl::getTopoConInfo(Op *op) const {
  TopoBundle tcInf;
  std::vector<const Tensor *> wouldMerge;
  // The unique input to op:
  wouldMerge.push_back(op->input.tensor(0));
  // And the N output tensors of op:
  for (auto &t_inds : op->output.indicesMap()) {
    wouldMerge.push_back(t_inds.first);
  }

  for (auto &tensor : wouldMerge) {
    if (tensor->consumers.hasTopoLast()) {
      ++tcInf.nTopoLasts;
      tcInf.lastCon = tensor->consumers.getTopoLast();
    }
    if (tensor->consumers.hasWeakTopoCons()) {
      ++tcInf.nWeakTopoCons;
    }
  }
  return tcInf;
}

// touches all the outputs of the root op [*] from the Ir,
// and maybe the input to the root op.
std::vector<const Tensor *> PostNRepl::touches(Op *op) const {
  std::vector<const Tensor *> outs;
  for (auto &t_inds : op->output.indicesMap()) {
    outs.push_back(t_inds.first);
  }
  // if one of new consumers of (ori) is an inplace-consumer, then (ori)
  // is touched. We take a superset of this case: if there
  // are any topological constraints on any of (ori, rep1, rep2, rep3),
  // we assume (ori) is touched.
  TopoBundle topoBundle = getTopoConInfo(op);
  if (topoBundle.nTopoLasts > 0 || topoBundle.nWeakTopoCons > 0) {
    outs.push_back(op->input.tensor(0));
  }
  return outs;
}

// (see .hpp for ascii picture definitions)
void PostNRepl::apply(Op *op) const {

  Ir *pir = op->pir;

  // op is [*]
  Tensor *ori = op->input.tensor(0);

  // get the info on which will be the last op
  // to consume ori, if there is one
  TopoBundle tcInf = getTopoConInfo(op);

  std::vector<Tensor *> replicates;
  // setting replicates (rep1), (rep2), (rep3)
  for (auto &ind_t : op->output.tensorMap()) {
    replicates.push_back(ind_t.second);
  }

  for (auto t_repl : replicates) {
    // for rep1 : {[op0], [op2]}
    for (Op *op_z : t_repl->consumers.getOps()) {
      // at what indices is (rep1) consumed?
      for (int index : op_z->input.indices(t_repl)) {
        // must rather consume ori
        op_z->input.reset(index, ori);
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

  // finally, clear up topo last if necessary
  if (ori->consumers.hasTopoLast()) {
    ori->consumers.removeTopoLast();
  }
  if (tcInf.nTopoLasts == 1) {
    ori->consumers.setTopoLast(tcInf.lastCon);
  }
}

OpId SoftmaxGradDirect::moveMergedIntoIr(Op *opRoot) const {
  // The root of the pattern is an NLLGrad,
  // we need to move from it th the SoftmaxOp
  Ir *pir     = opRoot->pir;
  Op *nllgrad = opRoot;

  return pir->moveIntoIr(std::unique_ptr<Op>(new SoftmaxGradDirectOp(
      pir, dynamic_cast<NllGradOp *>(nllgrad)->nlll())));
}

void FuserPattern::apply(Op *op) const {
  Ir *pir = op->pir;

  Op *op0      = op;
  Tensor *out0 = op0->output.tensor(0);
  Op *op1      = out0->consumers.getOps()[0];
  Tensor *out1 = op1->output.tensor(0);

  // create the replacement op01, connect it to
  // - the inputs if op0
  // - the output of op1
  OpId id01 = moveMergedIntoIr(op);
  Op *op01  = pir->getOp(id01);

  // wire-up the inputs
  pir->connectInputsFromInputMapWrapper(
      InputMapWrapper(op0->input.tensorIdMap()), id01);
  for (auto index_tensor : op0->input.tensorMap()) {
    Tensor *in0 = index_tensor.second;
    in0->consumers.decrement(op0);
    // Send any topological constraints from op0 to op01
    if (in0->consumers.hasTopoLast()) {
      if (in0->consumers.getTopoLast() == op0) {
        in0->consumers.removeTopoLast();
        in0->consumers.setTopoLast(op01);
      }
    }
    if (in0->consumers.hasWeakTopoCons()) {
      throw error("WeakTopoCons needs handling in this Fuser Pattern.");
    }
  }

  // we can't use connectOutputs, as that expects
  // that the output Tensor doesn't exist and must
  // be created. We rewire outputs manually:
  op01->output.insert(0, out1);
  out1->resetProducer(op01);

  // remove the tensor and nodes
  pir->getTensors().remove(out0->id);
  pir->eraseOp(op0->id);
  pir->eraseOp(op1->id);
}

} // namespace willow
