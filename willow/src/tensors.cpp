// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/chains.hpp>
#include <popart/graph.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensors.hpp>

namespace popart {

view::Chains Tensors::getChainsFromTo(Tensor *from, Tensor *to) const {
  if (from == to) {
    return view::Chains::getIdentity(from->info.shape());
  }

  if (aliasChainsFromKey.find(from) == aliasChainsFromKey.end()) {
    throw error("No chains out of {} found (in particular, none to {})",
                from->str(),
                to->str());
  }
  auto &allChainsFrom = aliasChainsFromKey.at(from);
  if (allChainsFrom.find(to) == allChainsFrom.end()) {
    std::ostringstream oss;
    oss << "There are chains from " << from->str() << " but none to "
        << to->str() << ". ";
    auto foundRev0 = aliasChainsToKey.find(to);
    if (foundRev0 == aliasChainsToKey.end()) {
      oss << "There are NO chains from " << to->str();
    } else {
      auto revMap = foundRev0->second;
      if (revMap.find(from) == revMap.end()) {
        oss << "There are chains from " << to->str() << " but none to "
            << from->str();
      } else {
        oss << " There is a chain from " << to->str() << " to " << from->str();
      }
    }
    throw error(oss.str());
  }
  return allChainsFrom.at(to);
}

TensorId Tensors::moveIntoTensors(std::unique_ptr<Tensor> tensor) {
  auto id = tensor->id;
  insert(id, std::move(tensor));
  return id;
}

std::unordered_map<Tensor *, view::Chains> Tensors::getAliasChains(
    const std::unordered_map<Tensor *,
                             std::unordered_map<Tensor *, view::Chains>> &fullM,
    Tensor *t) const {
  std::unordered_map<Tensor *, view::Chains> retM{};
  auto found = fullM.find(t);
  if (found != fullM.end()) {
    retM = found->second;
  }
  retM[t] = view::Chains::getIdentity(t->info.shape());
  return retM;
}

std::unordered_map<Tensor *, view::Chains>
Tensors::aliasChainsTo(Tensor *to) const {
  return getAliasChains(aliasChainsToKey, to);
}

std::unordered_map<Tensor *, view::Chains>
Tensors::aliasChainsFrom(Tensor *from) const {
  return getAliasChains(aliasChainsFromKey, from);
}

// Regions in "from" aliased "to"
view::Regions Tensors::getAliasRegions(Tensor *from, Tensor *to) const {
  auto aliasedTensorMap = graph.getTensors().aliasChainsFrom(from);
  auto it               = aliasedTensorMap.find(to);
  if (it == aliasedTensorMap.end()) {
    return view::Regions({view::Region::getEmpty(to->info.rank())});
  } else {
    return it->second.apply(view::Region::getFull(from->info.shape()));
  }
}

void Tensors::clearAliases() {
  aliasChainsFromKey.clear();
  aliasChainsToKey.clear();
}

// Let the Chains flow through op (called on new inplace ops)
void Tensors::updateAliases(Op *op) {
  logging::trace("[updateAliases] Updating alias for Op {}", op->debugName());

  std::unordered_map<Tensor *, std::unordered_map<Tensor *, view::Chains>>
      newAliases;

  auto registerChains =
      [&newAliases](Tensor *t0, Tensor *t3, const view::Chains &newChains) {
        if (!newChains.isEmpty()) {
          if (newAliases.find(t0) == newAliases.end()) {
            newAliases[t0] = {};
          }
          if (newAliases.at(t0).find(t3) == newAliases.at(t0).end()) {
            newAliases[t0][t3] = {}; // empty Chains
          }
          // add the new Chains
          newAliases[t0][t3] = newAliases[t0][t3].parallel(newChains);
        }
      };

  // for all of the inputs of op, t1 and all output, t2:
  for (auto i1_t1 : op->input->tensorMap()) {
    for (auto o1_t2 : op->output->tensorMap()) {
      InIndex i1 = i1_t1.first;
      Tensor *t1 = i1_t1.second;

      InIndex o1 = o1_t2.first;
      Tensor *t2 = o1_t2.second;

      logging::trace("[updateAliases] In: {}-{} {}, Out: {}-{} {}",
                     i1,
                     t1->id,
                     t1->info.shape(),
                     o1,
                     t2->id,
                     t2->info.shape());

      view::Regions inRegions = op->aliases(i1, o1);

      for (auto inRegion : inRegions) {
        if (inRegion.isEmpty()) {
          continue;
        }

        auto fwdMap = op->fwdRegMap(i1, o1);
        auto bwdMap = op->bwdRegMap(i1, o1);

        view::Regions outRegions = fwdMap(inRegion);

        // if there is an alias between the unique output
        // t2 and the input t1, this opens new Chains
        for (auto outRegion : outRegions) {
          if (outRegion.isEmpty()) {
            continue;
          }

          view::Link fwdLink(inRegion,
                             fwdMap,
                             "Fwd Link of " + op->debugName() + " " +
                                 std::to_string(i1) + "->" +
                                 std::to_string(o1));

          view::Link bwdLink(outRegion,
                             bwdMap,
                             "Bwd Link of " + op->debugName() + " " +
                                 std::to_string(i1) + "->" +
                                 std::to_string(o1));

          // all chains t0 -> t1 for all t0
          auto allInChains = aliasChainsTo(t1);

          // all chains t2 -> t3 for all t3
          auto allOutChains = aliasChainsFrom(t2);

          for (auto &inwards : allInChains) {
            Tensor *t0 = inwards.first;
            // the chains t0 -> t1
            view::Chains inChains      = inwards.second;
            auto inChainsFwdLinkSeries = inChains.series(fwdLink);

            // the chains t1 -> t0. There are such chains,
            // guaranteed by the existence of chains t0 -> t1
            logging::trace("[updateAliases] getChainsFromTo(t1, t0)");
            view::Chains inChainsRev = getChainsFromTo(t1, t0);

            for (auto &outwards : allOutChains) {

              Tensor *t3 = outwards.first;

              logging::trace("[updateAliases] Chain {}->{}->{}->{}",
                             t0->id,
                             t1->id,
                             t2->id,
                             t3->id);

              // the chains t2 -> t3
              view::Chains outChains = outwards.second;

              // the chains t3 -> t2
              // (which must exist by symmetry of aliasing)
              view::Chains outChainsRev = getChainsFromTo(t3, t2);

              // we now have,
              // t0 -----> t1 -> op -> t2 -----> t3
              // and we want to update aliasChainsToKey[t3][t0]
              // with all new chains that pass through op, as
              // well as aliasChainsToKey[t0][t3]

              auto newFwdChains = inChainsFwdLinkSeries.series(outChains);
              auto newBwdChains =
                  outChainsRev.series(bwdLink).series(inChainsRev);

              bool fwdIsEmpty = newFwdChains.isEmpty();
              bool bwdIsEmpty = newBwdChains.isEmpty();

              if (fwdIsEmpty != bwdIsEmpty) {
                std::ostringstream oss;
                oss << "\n\nnewFwdChains : \n" << newFwdChains << '\n';
                oss << "\ninChains : \n" << inChains << '\n';
                oss << "\nfwdLink : \n" << fwdLink << '\n';
                oss << "\noutChains : \n" << outChains << '\n';
                oss << "\nDetermining if newFwdChains is empty" << '\n';
                oss << "\nConclusion, fwdIsEmpty = : " << fwdIsEmpty << '\n';
                oss << "\n\nnewBwdChains : \n" << newBwdChains << '\n';
                oss << "\noutChainsRev : \n" << outChainsRev << '\n';
                oss << "\nbwdLink : \n" << bwdLink << '\n';
                oss << "\ninChainsRev : \n" << inChainsRev << '\n';
                oss << "\nDetermining if newBwdChains is empty" << '\n';
                oss << "\nConclusion, bwdIsEmpty : " << bwdIsEmpty << '\n';
                throw internal_error(oss.str());
              }

              if (!fwdIsEmpty) {
                logging::trace("[updateAliases] Non-empty fwd chains, "
                               "appending to aliasChainsToKey");
                registerChains(t3, t0, newFwdChains);
              }

              // same logic for t3 -> t0
              if (!bwdIsEmpty) {
                logging::trace("[updateAliases] Non-empty bwd chains, "
                               "appending to aliasChainsToKey");
                registerChains(t0, t3, newBwdChains);
              }
            }
          }
        }
      }
    }
  }

  for (auto x : newAliases) {
    auto t0         = x.first;
    auto t3_chain_s = x.second;
    if (aliasChainsToKey.find(t0) == aliasChainsToKey.end()) {
      aliasChainsToKey[t0] = {};
    }
    for (auto t3_chain : t3_chain_s) {
      auto t3    = t3_chain.first;
      auto chain = t3_chain.second;
      if (aliasChainsToKey.at(t0).find(t3) == aliasChainsToKey.at(t0).end()) {
        aliasChainsToKey[t0][t3] = {}; // empty Chains
      }
      // add the new Chains
      aliasChainsToKey[t0][t3] = aliasChainsToKey[t0][t3].parallel(chain);
      // insert the mirror image
      aliasChainsFromKey[t3][t0] = aliasChainsToKey[t0][t3];
    }
  }
}

std::vector<TensorId> Tensors::getAllTensorIds() const {
  std::vector<TensorId> allIds;
  allIds.reserve(M.size());
  for (auto &id_tensor : M) {
    allIds.push_back(id_tensor.first);
  }
  return allIds;
}

// remove all Tensors with no producer and no consumers
void Tensors::removeIsolated(bool retainCached) {
  for (auto &id : getAllTensorIds()) {
    Tensor *tensor = M[id].get();
    if (tensor->hasProducer() == false && tensor->consumers.getTotal() == 0 &&
        !(retainCached && tensor->isCached())) {
      M.erase(id);
      logging::ir::debug("Removing isolated Tensor {}", id);
    }
  }
}

std::vector<Tensor *> Tensors::getOfType(TensorType type) const {
  std::vector<Tensor *> ofType;
  for (auto &id_pt : M) {
    if (id_pt.second->tensorType() == type) {
      ofType.push_back(id_pt.second.get());
    }
  }
  return ofType;
}

std::vector<TensorId> Tensors::getIds(TensorType type) const {
  auto typedTensors = getOfType(type);
  std::vector<TensorId> ids;
  ids.reserve(typedTensors.size());
  for (Tensor *t : typedTensors) {
    ids.push_back(t->id);
  }
  return ids;
}

Tensors::Tensors(Graph &pg) : graph(pg) {}

Tensor *Tensors::get(TensorId tenId) const {
  auto found = M.find(tenId);
  if (found == M.end()) {
    throw error("No Ir::Tensor with TensorId " + tenId +
                " in Tensors::get(..)");
  }
  return found->second.get();
}

bool Tensors::contains(TensorId tenId, const Scope &scope) const {
  Scope s = scope;

  while (!s.empty()) {
    auto id = (s / tenId).str();
    if (M.find(id) != M.end()) {
      return true;
    } else {
      s.pop();
    }
  }

  if (M.find(tenId) != M.end()) {
    return true;
  } else {
    return false;
  }
}

TensorId Tensors::find(TensorId tenId, const Scope &scope) const {
  Scope s = scope;

  while (!s.empty()) {
    auto id = (s / tenId).str();
    if (M.find(id) != M.end()) {
      return id;
    } else {
      s.pop();
    }
  }

  if (M.find(tenId) != M.end()) {
    return tenId;
  } else {
    throw error("Could not find tensor with id {} in scope {}", tenId, scope);
  }
}

void Tensors::append(std::stringstream &ss) const {
  bool frst = true;
  ss << '[';
  for (auto &id_ptr : M) {
    if (!frst) {
      ss << ' ';
    }
    frst = false;
    ss << id_ptr.first;
  }
  ss << ']';
}

std::vector<TensorId> Tensors::getNoProducerIds() const {
  // the tensors which are not generated by an Op
  std::vector<TensorId> t0 = getIds(TensorType::Stream);
  std::vector<TensorId> t1 = getIds(TensorType::Const);
  std::vector<TensorId> t2 = getIds(TensorType::Variable);
  t0.insert(t0.end(), t1.begin(), t1.end());
  t0.insert(t0.end(), t2.begin(), t2.end());
  return t0;
}

void Tensors::insert(TensorId name, std::unique_ptr<Tensor> t) {
  if (M.find(name) != M.end()) {
    throw internal_error("tensor {} already in M", name);
  }
  M[name] = std::move(t);
}

void Tensors::addConstInit(const TensorId &name,
                           const ONNX_NAMESPACE::TensorProto *pt) {
  addInit(name, pt, TensorType::Const);
  insertConstId(name);
}

void Tensors::addVarInit(const TensorId &name,
                         const ONNX_NAMESPACE::TensorProto *pt) {
  addInit(name, pt, TensorType::Variable);

  // A sanity check: if the tensor is fixed point, it is Const
  if (get(name)->info.getDataTypeInfo()->isFixedPoint()) {
    if (!constIds.contains(name)) {
      std::stringstream ss;
      ss << "A fixed-point Variable tensor `" << name
         << "'. Currently only floating-point tensors can be Variable. "
         << " Consider setting fixed-point tensors to be outputs of Constant "
         << "Ops, using (for example) "
         << "convertAllFixedPointInitializersToConstants().";
      throw error(ss.str());
    }
  }
}

void Tensors::addVarInit(const TensorId &name,
                         const TensorInfo &info,
                         const void *src) {
  insert(name,
         std::unique_ptr<VariableTensor>(new VariableTensor(name, graph)));

  Tensor *init = get(name);
  init->info   = info;
  init->setTensorData(info, src);
}

void Tensors::addConstInit(const TensorId &name,
                           const TensorInfo &info,
                           const void *src) {
  insert(name, std::make_unique<Tensor>(name, TensorType::Const, graph));

  insertConstId(name);

  Tensor *init = get(name);
  init->info   = info;
  init->setTensorData(info, src);
}

void Tensors::makeConstInit(const TensorId &name, const void *src) {
  insertConstId(name);

  auto *tensor = get(name);
  if (tensor->hasProducer()) {
    throw error("cannot make an existing tensor const if it has a producer");
  }
  tensor->setTensorType(TensorType::Const);
  tensor->setTensorData(tensor->info, src);
}

void Tensors::addInit(const TensorId &name,
                      const ONNX_NAMESPACE::TensorProto *pt,
                      TensorType tt) {

  if (tt == TensorType::Variable) {
    insert(name, std::make_unique<VariableTensor>(name, graph));
  } else {
    insert(name, std::make_unique<Tensor>(name, tt, graph));
  }

  Tensor *init = get(name);
  init->info   = TensorInfo(*pt);
  init->setTensorData(*pt);
}

void Tensors::addStream(TensorId tenId, const TensorInfo &info) {
  insert(tenId,
         std::unique_ptr<Tensor>(new Tensor(tenId, TensorType::Stream, graph)));
  get(tenId)->info = info;
}

void Tensors::addActGrad(TensorId tenId) {
  logging::debug("Adding ActGrad Tensor {}", tenId);
  insert(
      tenId,
      std::unique_ptr<Tensor>(new Tensor(tenId, TensorType::ActGrad, graph)));
}

void Tensors::remove(TensorId id) { M.erase(id); }

bool Tensors::contains(TensorId id) const { return M.find(id) != M.end(); }

void Tensors::insertConstId(const std::string &id) { constIds.insert(id); }

} // namespace popart
