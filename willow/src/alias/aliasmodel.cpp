// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <map>
#include <ostream>
#include <string>
#include <utility>
#include <vector>
#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/typedinteger.hpp>
#include <popart/alias/aliasmodel.hpp>
#include <popart/ir.hpp>

#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {

using PoprithmsTensorId = poprithms::memory::inplace::TensorId;
using PoprithmsOpId     = poprithms::memory::inplace::OpId;

void AliasModel::insertTensor(const PoprithmsTensorId &id, const Tensor &t) {
  toTensor_[t.id] = id;
  fromTensor_[id] = t.id;
  if (t.hasProducer()) {
    insertOp(id.opId(), t.getProducer()->id);
  }
}

void AliasModel::update(OpId oldId, OpId newId) {
  auto found = toOp_.find(oldId);
  if (found != toOp_.cend()) {
    auto oldTargets = found->second;
    toOp_.erase(found);
    toOp_[newId] = oldTargets;
    for (auto t : oldTargets) {
      fromOp_[t] = newId;
    }
  }
}

void AliasModel::insertOp(PoprithmsOpId poprithmsId, OpId id) {

  {
    auto iter = fromOp_.find(poprithmsId);
    if (iter != fromOp_.cend() && iter->second != id) {
      throw error("reinserting, with different value");
    }
  }

  fromOp_[poprithmsId] = id;
  auto found           = toOp_.find(id);
  if (found == toOp_.cend()) {
    toOp_[id] = {poprithmsId};
  }

  else {
    if (std::find(found->second.cbegin(), found->second.cend(), poprithmsId) ==
        found->second.cend()) {
      found->second.push_back(poprithmsId);
    }
  }
}

bool AliasModel::contains(const PoprithmsTensorId &id) const {
  return fromTensor_.find(id) != fromTensor_.cend();
}

TensorId AliasModel::getTensorId(const PoprithmsTensorId &id) const {
  const auto found = fromTensor_.find(id);
  if (found == fromTensor_.cend()) {
    std::ostringstream oss;
    oss << "Error in AliasModel::getTensorId(PoprithmsTensorId = " << id
        << "). There is no key " << id << " in fromTensor_. ";
    throw error(oss.str());
  }
  return found->second;
}

bool AliasModel::contains(const TensorId &id) const {
  return toTensor_.find(id) != toTensor_.cend();
}
PoprithmsTensorId AliasModel::getPoprithmsTensorId(const TensorId &id) const {
  const auto found = toTensor_.find(id);
  if (found == toTensor_.cend()) {
    std::ostringstream oss;
    oss << "Error in AliasModel::getPoprithmsTensorId(TensorId = " << id
        << "). There is no key " << id << " in toTensor_. ";
    throw error(oss.str());
  }
  return found->second;
}

bool AliasModel::contains(PoprithmsOpId id) const {
  return fromOp_.find(id) != fromOp_.cend();
}
OpId AliasModel::getOpId(PoprithmsOpId id) const {
  const auto found = fromOp_.find(id);
  if (found == fromOp_.cend()) {
    std::ostringstream oss;
    oss << "Error in AliasModel::getOpId(PoprithmsOpId = " << id
        << "). There is no key " << id << " in fromOp_. ";
    throw error(oss.str());
  }
  return found->second;
}

bool AliasModel::contains(OpId id) const {
  return toOp_.find(id) != toOp_.cend();
}
std::vector<PoprithmsOpId> AliasModel::getAll(OpId id) const {
  const auto found = toOp_.find(id);
  if (found == toOp_.cend()) {
    std::ostringstream oss;
    oss << "Error in AliasModel::getAll(OpId = " << id << "). There is no key "
        << id << " in toOp_. ";
    throw error(oss.str());
  }
  return found->second;
}

void AliasModel::insertUnaryModifier0(const Op &op) {
  insertUnaryModifier(op, 0);
}

void AliasModel::insertUnaryModifier(const Op &op, InIndex inIndex) {

  auto id0      = getPoprithmsTensorId(op.inId(inIndex));
  auto outPlace = op.isOutplace();

  const auto gate = outPlace ? g.aliasGate({id0}) : g.aliasGate({id0}, 0);

  auto modOut = g.modify(gate);

  insertTensor(modOut, *op.outTensor(0));
  insertOp(gate.opId(), op.id);
  insertOp(modOut.opId(), op.id);
}

void AliasModel::insertBinaryModifier(const Op &op) {
  insertNG2aryModifier(op, 2);
}

void AliasModel::insertNG2aryModifier(const Op &op, unsigned int numInputs) {
  auto outPlace = op.isOutplace();

  auto getReshapeIn = [this, &op](InIndex inIndex) {
    auto id_ = getPoprithmsTensorId(op.inId(inIndex));
    if (op.inInfo(inIndex).nelms() == op.outInfo(0).nelms() &&
        op.inShape(inIndex) != op.outShape(0)) {
      id_ = g.reshape({id_}, op.outShape(0));
      insertOp(id_.opId(), op.id);
    }

    return id_;
  };

  std::vector<PoprithmsTensorId> ids;

  for (unsigned int i = 0; i < numInputs; i++) {
    ids.push_back(getReshapeIn(i));
  }

  const int not_set = -1;

  PoprithmsTensorId gate(not_set, not_set);

  if (outPlace) {
    gate = g.aliasGate(ids);
  } else {
    for (unsigned int i = 0; i < numInputs; i++) {
      if (op.doesAlias(i, 0)) {
        if (gate.opId() != not_set) {
          throw error("Gate already set in insertNG2aryModifier. This method "
                      "assumes just one input aliases the output. This for "
                      "Op {}",
                      op.debugName());
        }
        gate = g.aliasGate(ids, i);
      }
    }
  }

  if (gate.opId() == not_set) {
    throw internal_error("Gate not set in insertNG2aryModifier. This for Op {}",
                         op.debugName());
  }

  const auto rGate = (g.shape(gate) == op.outShape(0))
                         ? gate
                         : g.reshape(gate, op.outShape(0));

  auto modOut = g.modify(gate);

  insertTensor(modOut, *op.outTensor(0));

  insertOp(gate.opId(), op.id);
  insertOp(rGate.opId(), op.id);
  insertOp(modOut.opId(), op.id);
}

void AliasModel::insertViewChange(PoprithmsTensorId vc,
                                  const Tensor &t,
                                  bool isOutplace) {

  auto gate = isOutplace ? g.aliasGate({vc}) : g.aliasGate({vc}, 0);
  insertTensor(gate, t);
  insertOp(gate.opId(), t.getProducer()->id);
  insertOp(vc.opId(), t.getProducer()->id);
}

PoprithmsOpId AliasModel::getGate(OpId id) const {

  auto pims = getAll(id);
  for (auto pid : pims) {
    if (g.isAliasGate(pid)) {
      return pid;
    }
  }

  throw error("Failed to find gate for this Op, {}", id);
}

std::vector<Tensor *> AliasModel::allAliases(const Tensor &t) const {

  auto tensorIt = toTensor_.find(t.id);
  if (tensorIt == toTensor_.end()) {
    throw error("[AliasModel::allAliases] Expected tensor '{}' to "
                "be in the AliasModel",
                t.id);
  }

  auto gTensor = getPoprithmsTensorId(t.id);

  std::vector<Tensor *> result;

  // Iterate over all aliases as found by poprithms.
  for (const auto &gAliasId : g.allAliases(gTensor)) {
    // All PopART tensors map to a Poprithms tensor, but not all Poprithms
    // tensors map to a PopART one. It is safe to only look at those Poprithms
    // aliases that have a corresponding PopART tensor.
    if (contains(gAliasId)) {
      // Translate back to PopART IDs.
      auto popartAliasId = getTensorId(gAliasId);
      auto popartAlias   = t.getIr().getTensor(popartAliasId);

      result.push_back(popartAlias);
    }
  }

  return result;
}

bool AliasModel::contains(const Tensor &super, const Tensor &sub) const {

  auto gSuper = getPoprithmsTensorId(super.id);
  auto gSub   = getPoprithmsTensorId(sub.id);

  return g.contains(gSuper, gSub);
}

} // namespace popart
