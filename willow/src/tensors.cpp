#include <poponnx/chains.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/names.hpp>
#include <poponnx/op.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensors.hpp>

namespace poponnx {

view::Chains Tensors::getChainsFromTo(Tensor *from, Tensor *to) const {
  if (from == to) {
    return view::Chains::getIdentity(from->info.shape());
  }

  auto &allChainsFrom = aliasChainsFromKey.at(from);
  if (allChainsFrom.find(to) == allChainsFrom.end()) {
    throw error("No chains {} -> {} found", from->str(), to->str());
  }
  return allChainsFrom.at(to);
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

// Let the Chains flow through op (called on new inplace ops)
void Tensors::updateAliases(Op *op) {

  // there is no aliasing for ops with more than 1 output,
  if (!(op->output->n() == 1 && op->output->hasIndex(0))) {
    throw error("No updateAliases for op which does not "
                " have a unique output "
                "at index 0, {}",
                op->str());
  }

  // for the unique output of op t2,
  Tensor *t2 = op->output->tensor(0);

  // and for all of the inputs of op, t1,
  for (auto i1_t1 : op->input->tensorMap()) {

    InIndex i1 = i1_t1.first;
    Tensor *t1 = i1_t1.second;

    auto fwdMap            = op->fwdRegMap(i1);
    view::Region inRegion  = op->aliases(i1);
    view::Region outRegion = fwdMap(inRegion);

    // if there is an alias between the unique output
    // t2 and the input t1, this opens new Chains
    if (!outRegion.isEmpty()) {

      auto bwdMap = op->bwdRegMap(i1);

      view::Link fwdLink(inRegion, fwdMap);
      view::Link bwdLink(outRegion, bwdMap);

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
        // guaranteed by the existance of chains t0 -> t1
        view::Chains inChainsRev = getChainsFromTo(t1, t0);

        for (auto &outwards : allOutChains) {
          Tensor *t3 = outwards.first;

          // the chains t2 -> t3
          view::Chains outChains = outwards.second;

          // the chains t3 -> t2 (which must exist by symmetry of aliasing)
          view::Chains outChainsRev = getChainsFromTo(t3, t2);

          // we now have,
          // t0 ------> t1 -> op -> t2 -----> t3
          // and we want to update aliasChainsToKey[t3][t0]
          // with all new chains that pass through op, as
          // well as aliasChainsToKey[t0][t3]

          auto newChains = inChainsFwdLinkSeries.series(outChains);
          if (!newChains.isEmpty()) {
            if (aliasChainsToKey.find(t3) == aliasChainsToKey.end()) {
              aliasChainsToKey[t3] = {};
            }
            if (aliasChainsToKey.at(t3).find(t0) ==
                aliasChainsToKey.at(t3).end()) {
              aliasChainsToKey[t3][t0] = {}; // empty Chains
            }
            // add the new Chains
            aliasChainsToKey[t3][t0] =
                aliasChainsToKey[t3][t0].parallel(newChains);

            // insert the mirror image
            aliasChainsFromKey[t0][t3] = aliasChainsToKey[t3][t0];
          }

          // same logic for t3 -> t0
          newChains = outChainsRev.series(bwdLink).series(inChainsRev);
          if (!newChains.isEmpty()) {
            if (aliasChainsToKey.find(t0) == aliasChainsToKey.end()) {
              aliasChainsToKey[t0] = {};
            }
            if (aliasChainsToKey.at(t0).find(t3) ==
                aliasChainsToKey.at(t0).end()) {
              aliasChainsToKey[t0][t3] = {}; // empty Chains
            }
            // add the new Chains
            aliasChainsToKey[t0][t3] =
                aliasChainsToKey[t0][t3].parallel(newChains);

            // insert the mirror image
            aliasChainsFromKey[t3][t0] = aliasChainsToKey[t0][t3];
          }
        }
      }
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
void Tensors::removeIsolated() {
  for (auto &id : getAllTensorIds()) {
    Tensor *tensor = M[id].get();
    if (tensor->hasProducer() == false && tensor->consumers.getTotal() == 0) {
      M.erase(id);
      logging::ir::debug("Removing isolated Tensor {}", id);
    }
  }
}

std::vector<TensorId> Tensors::getIds(TensorType type) const {
  std::vector<TensorId> ids;
  for (auto &id_pt : M) {
    if (id_pt.second->tensorType() == type) {
      ids.push_back(id_pt.first);
    }
  }
  return ids;
}

Tensors::Tensors(Ir &pg) : ir(pg) {}

Tensor *Tensors::get(TensorId tenId) const {
  auto found = M.find(tenId);
  if (found == M.end()) {
    throw error("no tensor with id " + tenId);
  }
  return found->second.get();
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
    throw error("ILE : tensor " + name + " already in M");
  }
  M[name] = std::move(t);
}

void Tensors::addConstInit(const TensorId &name, const onnx::TensorProto *pt) {
  addInit(name, pt, TensorType::Const);
  insertConstId(name);
}

void Tensors::addVarInit(const TensorId &name, const onnx::TensorProto *pt) {
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
void Tensors::addConstInit(const TensorId &name,
                           const TensorInfo &info,
                           const void *src) {
  insert(name,
         std::unique_ptr<Tensor>(new Tensor(name, TensorType::Const, ir)));

  insertConstId(name);

  Tensor *init = get(name);
  init->info   = info;
  init->setTensorData(info, src);
}

void Tensors::makeConstInit(const TensorId &name, const void *src) {
  insertConstId(name);

  auto *tensor = get(name);
  if (tensor->hasProducer()) {
    throw error("can not make an existing tensor const if it has a producer");
  }
  tensor->setTensorType(TensorType::Const);
  tensor->setTensorData(tensor->info, src);
}

void Tensors::addInit(const TensorId &name,
                      const onnx::TensorProto *pt,
                      TensorType tt) {

  if (tt == TensorType::Variable) {
    insert(name, std::unique_ptr<VariableTensor>(new VariableTensor(name, ir)));
  } else {
    insert(name, std::unique_ptr<Tensor>(new Tensor(name, tt, ir)));
  }

  Tensor *init = get(name);
  init->info   = TensorInfo(*pt);
  init->setTensorData(*pt);
}

void Tensors::addStream(TensorId tenId, const TensorInfo &info) {
  insert(tenId,
         std::unique_ptr<Tensor>(new Tensor(tenId, TensorType::Stream, ir)));
  get(tenId)->info = info;
}

void Tensors::addActGrad(TensorId tenId) {
  logging::debug("Adding ActGrad Tensor {}", tenId);
  insert(tenId,
         std::unique_ptr<Tensor>(new Tensor(tenId, TensorType::ActGrad, ir)));
}

void Tensors::remove(TensorId id) { M.erase(id); }

bool Tensors::contains(TensorId id) const { return M.find(id) != M.end(); }

void Tensors::insertConstId(const std::string &id) { constIds.insert(id); }

} // namespace poponnx
