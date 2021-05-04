// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithmsinplace.hpp>
#include <popart/graph.hpp>
#include <popart/op/elementwise.hpp>
#include <popart/op/pad.hpp>
#include <popart/op/scaledadd.hpp>
#include <popart/opidentifier.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

using PoprithmsTensorId = poprithms::memory::inplace::TensorId;
using PoprithmsOpId     = poprithms::memory::inplace::OpId;

void PoprithmsAliaser::insertTensor(const PoprithmsTensorId &id,
                                    const Tensor &t) {
  toTensor_[t.id] = id;
  fromTensor_[id] = t.id;
  if (t.hasProducer()) {
    insertOp(id.opId(), t.getProducer()->id);
  }
}

void PoprithmsAliaser::update(OpId oldId, OpId newId) {
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

void PoprithmsAliaser::insertOp(PoprithmsOpId poprithmsId, OpId id) {

  {
    auto iter = fromOp_.find(poprithmsId);
    if (iter != fromOp_.cend() && iter->second != id) {
      throw error("reinsterting, with different value");
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

bool PoprithmsAliaser::contains(const TensorId &id) const {
  return toTensor_.find(id) != toTensor_.cend();
}
PoprithmsTensorId
PoprithmsAliaser::getPoprithmsTensorId(const TensorId &id) const {
  const auto found = toTensor_.find(id);
  if (found == toTensor_.cend()) {
    std::ostringstream oss;
    oss << "Error in PoprithmsAliaser::get(TensorId = " << id
        << "). There is no key " << id << "in toTensor_. ";
    throw error(oss.str());
  }
  return found->second;
}

bool PoprithmsAliaser::contains(PoprithmsOpId id) const {
  return fromOp_.find(id) != fromOp_.cend();
}
OpId PoprithmsAliaser::getOpId(PoprithmsOpId id) const {
  const auto found = fromOp_.find(id);
  if (found == fromOp_.cend()) {
    std::ostringstream oss;
    oss << "Error in PoprithmsAliaser::get(PoprithmsOpId = " << id
        << "). There is no key " << id << "in fromOp_. ";
    throw error(oss.str());
  }
  return found->second;
}

bool PoprithmsAliaser::contains(OpId id) const {
  return toOp_.find(id) != toOp_.cend();
}
std::vector<PoprithmsOpId> PoprithmsAliaser::getAll(OpId id) const {
  const auto found = toOp_.find(id);
  if (found == toOp_.cend()) {
    std::ostringstream oss;
    oss << "Error in PoprithmsAliaser::get(OpId = " << id
        << "). There is no key " << id << "in toOp_. ";
    throw error(oss.str());
  }
  return found->second;
}

PoprithmsAliaser getPoprithmsAliaser(const Graph &graph) {

  logging::ir::debug("\nGrowing PoprithmsAliaser");
  PoprithmsAliaser m;

  for (auto op : graph.getOpSchedule({}, RequireOptimalSchedule::No)) {
    logging::ir::trace(
        "Inserting input Tensors for PoprithmsAliaser, Graph \"{}\"", graph.id);
    for (const auto &x : op->input->tensorMap()) {

      // If it's a new input tensor, register it:
      const auto t0 = x.second;
      if (!t0->hasProducer()) {
        if (!m.contains(t0->id)) { // to_.find(t0->id) == m.to.cend()) {
          const PoprithmsTensorId inId = (t0->tensorType() == TensorType::Const)
                                             ? m.g.constant(t0->info.shape())
                                             : m.g.variable(t0->info.shape());
          m.insertTensor(inId, *t0);
        }
      }
    }
    logging::ir::trace("Growing PoprithmsAliaser for op\"{}\"", op->str());
    op->growAliaser(m);

    for (auto x : m.getAll(op->id)) {
      logging::ir::trace(
          "Setting name for op \"{}\", poprithms OpId={}", op->str(), x);
      m.g.setName(x, op->str());
    }
  }

  logging::ir::trace("Inserting constraints for PoprithmsAliaser, Graph \"{}\"",
                     graph.id);

  // Insert constraints:
  for (const auto &from_tos : graph.getEdgeMap()) {
    const auto from       = from_tos.first;
    const auto &tos       = from_tos.second;
    const auto memoryFrom = m.getAll(graph.getOp(from)->id);
    for (auto to : tos) {
      if (from != to) {
        const auto memoryTo = m.getAll(graph.getOp(to)->id);
        for (auto f : memoryFrom) {
          for (auto t : memoryTo) {
            m.g.constraint(f, t);
          }
        }
      }
    }
  }

  return m;
}

void PoprithmsAliaser::insertUnaryModifier0(const Op &op) {
  insertUnaryModifier(op, 0);
}

void PoprithmsAliaser::insertUnaryModifier(const Op &op, InIndex inIndex) {

  auto id0      = getPoprithmsTensorId(op.inId(inIndex));
  auto outPlace = op.isOutplace();

  const auto gate = outPlace ? g.aliasGate({id0}) : g.aliasGate({id0}, 0);

  auto modOut = g.modify(gate);

  insertTensor(modOut, *op.outTensor(0));
  insertOp(gate.opId(), op.id);
  insertOp(modOut.opId(), op.id);
}

void PoprithmsAliaser::insertBinaryModifier(const Op &op) {

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

  const auto id0 = getReshapeIn(0);
  const auto id1 = getReshapeIn(1);

  const auto gate = outPlace ? g.aliasGate({id0, id1}) :

                             (op.doesAlias(0, 0) ? g.aliasGate({id0, id1}, 0)
                                                 : g.aliasGate({id0, id1}, 1));

  const auto rGate = (g.shape(gate) == op.outShape(0))
                         ? gate
                         : g.reshape(gate, op.outShape(0));

  auto modOut = g.modify(gate);

  insertTensor(modOut, *op.outTensor(0));

  insertOp(gate.opId(), op.id);
  insertOp(rGate.opId(), op.id);
  insertOp(modOut.opId(), op.id);
}

void PoprithmsAliaser::insertViewChange(PoprithmsTensorId vc,
                                        const Tensor &t,
                                        bool isOutplace) {

  auto gate = isOutplace ? g.aliasGate({vc}) : g.aliasGate({vc}, 0);
  insertTensor(gate, t);
  insertOp(gate.opId(), t.getProducer()->id);
  insertOp(vc.opId(), t.getProducer()->id);
}

PoprithmsOpId PoprithmsAliaser::getGate(OpId id) const {

  auto pims = getAll(id);
  for (auto pid : pims) {
    if (g.isAliasGate(pid)) {
      return pid;
    }
  }

  throw error("Failed to find gate for this Op, {}", id);
}

} // namespace popart
