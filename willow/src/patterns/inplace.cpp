#include <popart/chains.hpp>
#include <popart/graph.hpp>
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
    // t_clone->id += "/" + std::to_string(index_tensor.first);
    t_clone->id += std::to_string(index_tensor.first);
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

    if (!in_tensor->hasProducer()) {
      graph.markAsInputConsumedInplaceForOptimization(in_tensor->id);
    }
  }
  output_tensor->resetProducer(inplaceOp);
  inplaceOp->output->insert(0, output_tensor);
  inplaceOp->setup();

  graph.getTensors().updateAliases(inplaceOp);
  graph.topoCons->insert(newCons);

  logging::pattern::debug("InplaceAll::apply : replaced {}({}) with {}({})",
                          op->id,
                          op->opid,
                          inplaceOp->id,
                          inplaceOp->opid);

  inplaceOp->getGraph().eraseOp(op->id);
  return true;
}

// if "op" is replaced with an inplace variant of type "inpid",
// what additional topological constraints are needed?
OpsBeforeKey Inplace::getNewTopoCons(Op *op, OperatorIdentifier inpid) const {

  ExternOpTensorBundle eot_bun(op, op->getInplaceVariant(inpid));
  Op *inOp = eot_bun.getOp();

  logging::pattern::debug(
      "Getting new topological constraints if {} replaces {}",
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
      view::Regions regions = M[key];
      for (auto &region : newRegs) {
        regions.push_back(region);
      }
      M[key] = mergeRegions(regions);
    }
  };

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
  //  ? Very briefly, we do the following
  //
  //  (1)
  //  look for all t0 which are an alias of t1,
  //  and all t3 which are modified AND an alias of t2,
  //  and add the constraint that consumers of t0
  //  run BEFORE consumers of t3, but only if the modified aliased
  //  region of t3 overlaps with the used (consumed) aliased region of t0.
  //
  //  (2) if op (above) is constrained to run before otherOp,
  //  we check if we need to add the constraint that consumer of t3
  //  must run before otherOp too. Example: see slice_1_ip_test

  auto &tensors = op->getGraph().getTensors();
  Tensor *t2    = op->output->tensor(0);

  // (1.1) getting all consumers of t0-like tensors (see above diagram)
  // for each consumer of t0, pass the region consumed (used)
  // through chsI, then through the bottleLink, to a region
  // in t2, creating a map;
  // of (op):(regions in t2)
  std::map<Op *, view::Regions> consumer_regions;
  for (auto index_tensor : op->input->tensorMap()) {
    InIndex index = index_tensor.first;
    Tensor *t1    = index_tensor.second;
    for (auto aliasRegion : inOp->aliases(index, 0)) {
      view::Link bottleLink(
          aliasRegion, inOp->fwdRegMap(index, 0), "from_" + inOp->str());
      for (auto t0_chsI : tensors.aliasChainsTo(t1)) {
        auto t0   = t0_chsI.first;
        auto chsI = t0_chsI.second;
        for (auto c : t0->consumers.getOps()) {
          // the indices at which the tensor is consumed:
          for (InIndex t0in : c->input->indices(t0)) {
            auto serialChains = chsI.series(bottleLink);
            for (auto r : c->uses(t0in)) {
              auto r2 = serialChains.apply(r);
              populate(consumer_regions, c, r2);
            }
          }
        }
      }
    }
  }

  // return a map (Op* : Regions) of all Ops which use/modify an alias
  // of t2. The argument getRegion is either uses(.) or modifies(.)
  auto getPostRegions =
      [op, &populate, &tensors, t2, inOp](
          // where getRegion might be op->uses(.) or op->modifies(.)
          std::function<view::Regions(Op *, InIndex)> getRegions) {
        // to be set and returned in this function
        std::map<Op *, view::Regions> regions;

        // first, all ops downstream of op which modify/use
        // t2, and the modified/used regions
        for (auto t3_chsO : tensors.aliasChainsTo(t2)) {
          auto t3   = t3_chsO.first;
          auto chsO = t3_chsO.second;
          for (auto consumer : t3->consumers.getOps()) {
            for (InIndex t3_in : consumer->input->indices(t3)) {
              // where getRegion is the modified or used region
              for (auto r : getRegions(consumer, t3_in)) {
                view::Regions r2 = chsO.apply(r);
                populate(regions, consumer, r2);
              }
            }
          }
        }
        // second, what in t2 would inOp modify/use?
        view::Regions opModRegs;
        for (auto index_tensor : op->input->tensorMap()) {
          InIndex index = index_tensor.first;
          for (auto r0 : getRegions(inOp, index)) {
            for (auto r1 : inOp->fwdRegMap(index, 0)(r0)) {
              opModRegs.push_back(r1);
            }
          }
        }
        populate(regions, op, opModRegs);
        return regions;
      };

  // (1.2) getting all modifiers of t3-like tensors above
  std::map<Op *, view::Regions> modifier_regions =
      getPostRegions([](Op *o, InIndex i) { return o->modifies(i); });

  // this is what we populate in this function
  OpsBeforeKey gCons;

  // modifer_regions X consumer_regions, the match-up
  auto match_up = [&gCons](const std::map<Op *, view::Regions> &before_regions,
                           const std::map<Op *, view::Regions> &after_regions) {
    for (const auto &op_regs0 : before_regions) {
      for (const auto &op_regs1 : after_regions) {
        Op *before = op_regs0.first;
        Op *after  = op_regs1.first;
        for (const auto &reg0 : op_regs0.second) {
          for (const auto &reg1 : op_regs1.second) {
            // TODO : more efficient way of testing whether Regions
            // (set of Regions) intersect (T7104)
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
  };
  match_up(consumer_regions, modifier_regions);

  // (2)
  auto &graph   = op->getGraph();
  auto afterOps = graph.topoCons->getAfters(op);
  std::map<Op *, view::Regions> after_op_regions;
  for (auto after : afterOps) {
    auto found = consumer_regions.find(after);
    if (found != consumer_regions.end()) {
      after_op_regions[after] = found->second;
    }
  }
  std::map<Op *, view::Regions> post_uses_regions =
      getPostRegions([](Op *o, InIndex i) { return o->uses(i); });
  match_up(post_uses_regions, after_op_regions);

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
      auto modified = inOp->modifies(index_tensor.first);
      if (!std::all_of(modified.begin(),
                       modified.end(),
                       [](const view::Region &r) { return r.isEmpty(); })) {
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

  // we're done, now collect a logging string
  if (modifier_regions.size() != 0) {
    std::stringstream ss;
    ss << "Modifier regions for " << op->str() << ": [ ";
    for (auto &x : modifier_regions) {
      ss << x.first->str() << ' ';
    }
    ss << "]";
    logging::pattern::debug(ss.str());
  }

  if (consumer_regions.size() != 0) {
    std::stringstream ss;
    ss << "Consumer regions for " << op->str() << ": [ ";
    for (auto &x : consumer_regions) {
      ss << x.first->str() << ' ';
    }
    ss << "]";
    logging::pattern::debug(ss.str());
  }

  if (gCons.size() != 0) {
    std::stringstream ss;
    for (auto key_befores : gCons) {
      Op *key      = key_befores.first;
      auto befores = key_befores.second;
      for (Op *before : befores) {
        ss << "\n           " << before->debugName() << "-->"
           << key->debugName();
      }
    }
    logging::pattern::debug("New constraints:" + ss.str());
  }

  return gCons;
}

namespace {
static AddPatternName<Inplace> registerName("InPlace");
} // namespace
} // namespace popart
