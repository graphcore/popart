// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/chains.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordebuginfo.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensors.hpp>

namespace popart {

view::Chains Tensors::getChainsFromTo(Tensor *from, Tensor *to) const {
  return aliases.getChainsFromTo(from, to);
}

TensorId Tensors::moveIntoTensors(std::unique_ptr<Tensor> tensor) {
  auto id = tensor->id;
  insert(id, std::move(tensor));
  return id;
}

std::unordered_map<Tensor *, view::Chains>
Tensors::aliasChainsTo(Tensor *to) const {
  return aliases.aliasChainsTo(to);
}

std::unordered_map<Tensor *, view::Chains>
Tensors::aliasChainsFrom(Tensor *from) const {
  return aliases.aliasChainsFrom(from);
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

void Tensors::clearAliases() { aliases.clearAliases(); }

// Let the Chains flow through op (called on new inplace ops)
void Tensors::updateAliases(Op *op) {
  logging::trace("[updateAliases] Updating alias for Op {}", op->debugName());

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

      if (std::all_of(inRegions.begin(), inRegions.end(), [](view::Region &r) {
            return r.isEmpty();
          })) {
        continue;
      }

      auto fwdMap = op->fwdRegMap(i1, o1);
      auto bwdMap = op->bwdRegMap(i1, o1);

      aliases.updateAliases(t1,
                            t2,
                            inRegions,
                            fwdMap,
                            bwdMap,
                            "Fwd Link of " + op->debugName() + " " +
                                std::to_string(i1) + "->" + std::to_string(o1),
                            "Bwd Link of " + op->debugName() + " " +
                                std::to_string(i1) + "->" + std::to_string(o1));
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
void Tensors::removeIsolated(bool retainIoTensors) {
  auto hostLoadTensors = graph.getIr().getHostLoadTensors();
  for (auto &id : getAllTensorIds()) {
    Tensor *tensor = M[id].get();
    if (tensor->hasProducer() == false && tensor->consumers.getTotal() == 0 &&
        !(retainIoTensors &&
          (tensor->tensorLocationInfo.isRemote() || tensor->isAnchored() ||
           tensor->isRootAnchor() ||
           (tensor->tensorType() == TensorType::Stream &&
            hostLoadTensors.find(tensor->id) != hostLoadTensors.end())))) {
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

std::vector<Tensor *>
Tensors::getOfType(const std::vector<TensorType> &tTypes) const {
  std::vector<Tensor *> ofType;
  for (auto type : tTypes) {
    auto ofTypesTemp = getOfType(type);
    ofType.insert(ofType.end(), ofTypesTemp.cbegin(), ofTypesTemp.cend());
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
    throw error("No Ir::Tensor with TensorId '" + tenId +
                "' in Tensors::get(..)");
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
                           const ONNX_NAMESPACE::TensorProto *pt,
                           const DebugContext &debugContext) {
  popart::TensorDebugInfo di(debugContext, name, TensorType::Const);
  addInit(name, pt, TensorType::Const, di);
  insertConstId(name);
}

void Tensors::addVarInit(const TensorId &name,
                         const ONNX_NAMESPACE::TensorProto *pt,
                         const DebugContext &debugContext) {
  popart::TensorDebugInfo di(debugContext, name, TensorType::Variable);
  addInit(name, pt, TensorType::Variable, di);

  // A sanity check: if the tensor is fixed point, it is Const
  if (get(name)->info.getDataTypeInfo()->isFixedPoint()) {
    if (!constIds.contains(name)) {
      std::stringstream ss;
      ss << "Variable Tensor `" << name << "' is fixed-point, but "
         << "currently only floating-point Tensors can be variable in PopART. "
         << "If Tensor `" << name
         << "' should be constant instead of variable, "
         << "it can be converted by using "
         << "the GraphTransormer utility method "
         << "`convertAllFixedPointInitializersToConstants()', which converts "
         << "all fixed-point initializers in an ONNX ModelProto"
         << " to be outputs of ONNX Nodes of type Constant. ";
      throw error(ss.str());
    }
  }
}

void Tensors::addVarInit(const TensorId &name,
                         const TensorInfo &info,
                         const void *src,
                         const DebugContext &debugContext) {
  popart::TensorDebugInfo di(debugContext, name, info, TensorType::Variable);
  insert(name,
         std::unique_ptr<VariableTensor>(new VariableTensor(name, graph, di)));

  Tensor *init = get(name);
  init->info   = info;
  init->setTensorData(info, src);
}

void Tensors::addConstInit(const TensorId &name,
                           const TensorInfo &info,
                           const void *src,
                           const DebugContext &debugContext) {
  popart::TensorDebugInfo di(debugContext, name, info, TensorType::Const);
  insert(name, std::make_unique<Tensor>(name, TensorType::Const, graph, di));

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
                      TensorType tt,
                      const DebugInfo &di) {

  if (tt == TensorType::Variable) {
    insert(name, std::make_unique<VariableTensor>(name, graph, di));
  } else {
    insert(name, std::make_unique<Tensor>(name, tt, graph, di));
  }

  Tensor *init = get(name);
  init->info   = TensorInfo(*pt);
  init->setTensorData(*pt);
}

void Tensors::addStream(TensorId tenId,
                        const TensorInfo &info,
                        const DebugContext &debugContext) {
  popart::TensorDebugInfo di(debugContext, tenId, info, TensorType::Stream);
  insert(tenId,
         std::unique_ptr<Tensor>(
             new Tensor(tenId, TensorType::Stream, graph, di)));
  get(tenId)->info = info;
}

void Tensors::addActGrad(TensorId tenId, const DebugContext &debugContext) {
  popart::TensorDebugInfo di(debugContext, tenId, TensorType::ActGrad);
  logging::debug("Adding ActGrad Tensor {}", tenId);
  insert(tenId,
         std::unique_ptr<Tensor>(
             new Tensor(tenId, TensorType::ActGrad, graph, di)));
}

void Tensors::remove(TensorId id) { M.erase(id); }

bool Tensors::contains(TensorId id) const { return M.find(id) != M.end(); }

void Tensors::insertConstId(const std::string &id) { constIds.insert(id); }

} // namespace popart
