#include <poponnx/chains.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/op.hpp>
#include <poponnx/patterns/inplace.hpp>
#include <poponnx/pbwrap.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensors.hpp>
#include <poponnx/topocons.hpp>
#include <poponnx/util.hpp>

namespace poponnx {

ExternOpTensorBundle::ExternOpTensorBundle(Op *opCopy,
                                           std::unique_ptr<Op> opNew)
    : up_op(std::move(opNew)) {

  // dummy inputs
  for (auto &index_tensor : opCopy->input->tensorMap()) {
    std::unique_ptr<Tensor> up_t_clone = index_tensor.second->clone();
    Tensor *t_clone                    = up_t_clone.get();
    tensors[t_clone->id]               = std::move(up_t_clone);
    up_op->input->insert(index_tensor.first, t_clone);
    t_clone->consumers.increment(up_op.get());
  }

  // dummy outputs
  for (auto &index_tensor : opCopy->output->tensorMap()) {
    std::unique_ptr<Tensor> up_t_clone = index_tensor.second->clone();
    Tensor *t_clone                    = up_t_clone.get();
    tensors[t_clone->id]               = std::move(up_t_clone);
    up_op->output->insert(index_tensor.first, t_clone);
    t_clone->setProducer(up_op.get());
  }

  up_op->setup();
}

Op *ExternOpTensorBundle::getOp() { return up_op.get(); }

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
std::vector<const Tensor *> Inplace::touches(Op *op, OperatorIdentifier) const {

  // Should reconsider this. If we ensure that all returns
  // to host will be done after all inplace consumers of a tensor have
  // run, we can set this up such that the output tensor is not
  // touched (where the defn of touched would then be slightly different)

  std::vector<const Tensor *> touched;
  touched.reserve(op->input->n() + 1);
  touched.push_back(op->output->tensor(0));
  // TODO : it is actually  sub-set of the inputs, only those aliased (T6707)
  for (auto &x : op->input->indicesMap()) {
    touched.push_back(x.first);
  }
  return touched;
}

bool Inplace::apply(Op *op,
                    OperatorIdentifier identifier,
                    const OpsBeforeKey &newConsIn) const {
  auto output_tensor = op->output->tensor(0);
  auto &ir           = op->getIr();

  auto newCons = newConsIn;

  std::unique_ptr<Op> up_inplaceOp = op->getInplaceVariant(identifier);
  Op *inplaceOp                    = up_inplaceOp.get();
  ir.moveIntoIr(std::move(up_inplaceOp));
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
    ir.topoCons->transfer(op, inplaceOp);
    in_tensor->consumers.decrement(op);
    inplaceOp->input->insert(in_index, in_tensor);
  }
  output_tensor->resetProducer(inplaceOp);
  inplaceOp->output->insert(0, output_tensor);

  ir.getTensors().updateAliases(op);
  ir.topoCons->insert(newCons);

  logging::pattern::debug("InplaceAll::apply : replace {}({}) with {}({})",
                          op->id,
                          op->opid,
                          inplaceOp->id,
                          inplaceOp->opid);

  inplaceOp->getIr().eraseOp(op->id);
  return true;
}

OpsBeforeKey Inplace::getNewTopoCons(Op *op, OperatorIdentifier inpid) const {

  ExternOpTensorBundle eot_bun(op, op->getInplaceVariant(inpid));
  Op *inOp = eot_bun.getOp();

  logging::pattern::debug(
      "Getting new topological constraints if {} replacing {}",
      inOp->str(),
      op->str());

  auto populate = [](std::map<Op *, view::Regions> &M,
                     Op *key,
                     const view::Regions &newRegs) {
    if (newRegs.size() != 0) {
      auto found = M.find(key);
      if (found == M.end()) {
        M[key] = {};
      }
      view::Regions &regions = M[key];
      for (auto &region : newRegs) {
        // TODO : check that region is not
        // a sub-region of one already in (T6707)
        regions.push_back(region);
      }
    }
  };

  // this is what we populate here
  OpsBeforeKey gCons;

  // tensor naming plan,
  // t0 ------------> t1 ----> op ----> t2 -------------> t3
  //
  // can we instead have,
  // t0 ------------> t1 ---> inOp ---> t2 -------------> t3
  //        |             |          |           |
  //        |             ------------           |
  //        |                  |                 |
  //       chsI            bottleLink           chsO
  //
  // ?
  // Roughly speaking,
  //  we will look for all t0 which are an alias of t1,
  //  and all t3 which are modified AND an alias of t2,
  //  and add the constraint that consumers of t0
  //  run before consumers of t3, if the modified aliased region
  //  of t3 overlaps with the used (consumed) aliased region of t0

  auto &tensors = op->getIr().getTensors();
  Tensor *t2    = op->output->tensor(0);

  // for each consumer of t0, pass the region consumed (used)
  // through chainsTo, then through the bottleLink, to a region
  // in t2, creating a map;
  // of (op):(regions in t2)
  std::map<Op *, view::Regions> consumer_regions;
  for (auto index_tensor : op->input->tensorMap()) {
    InIndex index = index_tensor.first;
    Tensor *t1    = index_tensor.second;
    view::Link bottleLink(inOp->aliases(index), inOp->fwdRegMap(index));
    for (auto t0_chsI : tensors.aliasChainsTo(t1)) {
      auto t0   = t0_chsI.first;
      auto chsI = t0_chsI.second;
      for (auto c : t0->consumers.getOps()) {
        // the indices at which the tensor is consumed:
        for (InIndex t0in : c->input->indices(t0)) {
          auto serialChains = chsI.series(bottleLink);
          auto r2           = serialChains.apply(c->uses(t0in));
          populate(consumer_regions, c, r2);
        }
      }
    }
  }

  std::map<Op *, view::Regions> modifier_regions;
  // first, all ops downstream of op which modify t2, and the modified regions
  for (auto t3_chsO : tensors.aliasChainsTo(t2)) {
    auto t3   = t3_chsO.first;
    auto chsO = t3_chsO.second;
    for (auto consumer : t3->consumers.getOps()) {
      for (InIndex t3_in : consumer->input->indices(t3)) {
        view::Regions r2 = chsO.apply(consumer->modifies(t3_in));
        populate(modifier_regions, consumer, r2);
      }
    }
  }
  // second, what in t2 would inOp modify?
  view::Regions opModRegs;
  for (auto index_tensor : op->input->tensorMap()) {
    InIndex index = index_tensor.first;
    opModRegs.push_back(inOp->fwdRegMap(index)(inOp->modifies(index)));
  }
  populate(modifier_regions, op, opModRegs);

  // modifer_regions X consumer_regions, the match-up
  for (const auto &op_regs0 : consumer_regions) {
    for (const auto &op_regs1 : modifier_regions) {
      Op *before = op_regs0.first;
      Op *after  = op_regs1.first;
      for (const auto &reg0 : op_regs0.second) {
        for (const auto &reg1 : op_regs1.second) {
          // TODO : more efficient way of testing whether Regions
          // (set of Regions) intersect (T6707)
          if (!reg0.intersect(reg1).isEmpty()) {
            if (gCons.find(after) == gCons.end()) {
              gCons[after] = {before};
            } else {
              gCons[after].push_back(before);
            }
          }
        }
      }
    }
  }

  // handle the prickly case of gCons[op] containing op.
  // op must run before itself? Needs special attention.

  // Consider an Op which does
  // C <- gamma*A + B
  // and inplace version,
  // C *= gamma and then C += B.
  // if A = B, then the inplace is not valid, as A <- 2*gamma*A
  // For certain ops it is fine if the input is repeated (Add),
  // but for others this could be very bad

  bool pricklyCase = false;
  if (gCons.find(op) != gCons.end()) {
    for (auto &before : gCons.at(op)) {
      if (before == op) {
        // the prickly A -> A constraint case, where A = op
        pricklyCase = true;
      }
    }
  }

  bool schedFail = false;
  if (pricklyCase) {
    for (auto index_tensor : op->input->tensorMap()) {
      if (!inOp->modifies(index_tensor.first).isEmpty()) {
        InIndex modifyingIndex = index_tensor.first;
        Tensor *modifiedTensor = index_tensor.second;
        // we check if any of the inputs are aliased to modifiedTensor
        auto aliasedTensorMap = tensors.aliasChainsTo(modifiedTensor);
        for (auto &aliasedTensor_chain : aliasedTensorMap) {
          Tensor *aliasedTensor = aliasedTensor_chain.first;
          for (auto &index2_tensor2 : op->input->tensorMap()) {
            if (index2_tensor2.first != modifyingIndex &&
                aliasedTensor == index2_tensor2.second) {
              schedFail = true;
            }
          }
        }
      }
    }
  }

  if (pricklyCase && !schedFail) {
    std::vector<Op *> newAfters;
    for (auto after : gCons.at(op)) {
      if (after != op) {
        newAfters.push_back(after);
      }
    }
    if (newAfters.size() == 0) {
      gCons.erase(op);
    } else {
      gCons[op] = newAfters;
    }
  }

  return gCons;
} // namespace poponnx

} // namespace poponnx
