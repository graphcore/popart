// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithmsinplace.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/elementwise.hpp>
#include <popart/op/pad.hpp>
#include <popart/op/scaledadd.hpp>
#include <popart/opidentifier.hpp>
#include <popart/tensorindex.hpp>
#include <popart/topocons.hpp>

#include <poprithms/logging/timepartitionlogger.hpp>
#include <poprithms/schedule/vanilla/vanilla.hpp>

#include <list>
#include <unordered_set>

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

bool PoprithmsAliaser::contains(const PoprithmsTensorId &id) const {
  return fromTensor_.find(id) != fromTensor_.cend();
}

TensorId PoprithmsAliaser::getTensorId(const PoprithmsTensorId &id) const {
  const auto found = fromTensor_.find(id);
  if (found == fromTensor_.cend()) {
    std::ostringstream oss;
    oss << "Error in PoprithmsAliaser::getTensorId(PoprithmsTensorId = " << id
        << "). There is no key " << id << " in fromTensor_. ";
    throw error(oss.str());
  }
  return found->second;
}

bool PoprithmsAliaser::contains(const TensorId &id) const {
  return toTensor_.find(id) != toTensor_.cend();
}
PoprithmsTensorId
PoprithmsAliaser::getPoprithmsTensorId(const TensorId &id) const {
  const auto found = toTensor_.find(id);
  if (found == toTensor_.cend()) {
    std::ostringstream oss;
    oss << "Error in PoprithmsAliaser::getPoprithmsTensorId(TensorId = " << id
        << "). There is no key " << id << " in toTensor_. ";
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
    oss << "Error in PoprithmsAliaser::getOpId(PoprithmsOpId = " << id
        << "). There is no key " << id << " in fromOp_. ";
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
    oss << "Error in PoprithmsAliaser::getAll(OpId = " << id
        << "). There is no key " << id << " in toOp_. ";
    throw error(oss.str());
  }
  return found->second;
}

namespace {

/**
 * Data type that dictates whether we check at runtime whether a tensor that is
 * produced by an op may be added via `insertTensor`. When growing the alias
 * model for a full graph you would expect these tensors to be added by growing
 * their producer.
 **/
enum class AllowInsertingProducedTensors {
  // Produced tensors may be added via `insertTensor`.
  Yes = 0,
  // Producer tensors must be added by their producer.
  No
};

void addTensor(const Graph &graph,
               PoprithmsAliaser &m,
               Tensor *t,
               AllowInsertingProducedTensors allow) {
  if (!m.contains(t->id)) { // to_.find(t0->id) == m.to.cend()) {

    if (allow == AllowInsertingProducedTensors::No && t->hasProducer()) {
      // The tensor should have been added by growing it's producer.
      throw error("[PoprithmsAliaser] Tensor '{}' was not added to the alias "
                  "model by it's producer, {}",
                  t->id,
                  t->getProducer()->str());
    }

    const PoprithmsTensorId inId = (t->tensorType() == TensorType::Const)
                                       ? m.g.constant(t->info.shape())
                                       : m.g.variable(t->info.shape());
    m.insertTensor(inId, *t);
  }
}

void addAliaserOp(const Graph &graph,
                  PoprithmsAliaser &m,
                  Op *op,
                  AllowInsertingProducedTensors allow) {

  logging::ir::trace(
      "Inserting input Tensors for PoprithmsAliaser, Graph \"{}\"", graph.id);
  for (const auto &x : op->input->tensorMap()) {

    // If it's a new input tensor, register it:
    const auto t0 = x.second;
    addTensor(graph, m, t0, allow);
  }

  logging::ir::trace("Growing PoprithmsAliaser for op\"{}\"", op->str());
  op->growAliaser(m);

  // Check every output has been mapped.
  for (const auto &x : op->output->tensorMap()) {
    const auto t1 = x.second;
    if (!m.contains(t1->id)) {
      logging::ir::trace("Op {} failed to add mapping for output #{} ('') "
                         "in PoprithmsAliaser",
                         op->str(),
                         x.first,
                         t1->id);
    }
  }

  for (auto x : m.getAll(op->id)) {
    logging::ir::trace(
        "Setting name for op \"{}\", poprithms OpId={}", op->str(), x);
    m.g.setName(x, op->str());
  }
}

void addAliaserConstraints(const Graph &graph,
                           PoprithmsAliaser &m,
                           const std::vector<Op *> &opSubset) {

  logging::ir::trace("Inserting constraints for PoprithmsAliaser, Graph \"{}\"",
                     graph.id);

  // Insert constraints:
  for (auto from : opSubset) {
    const auto tos        = graph.topoCons->getAfters(from);
    OpId fromOp           = from->id;
    const auto memoryFrom = m.getAll(fromOp);
    for (auto to : tos) {
      if (from != to) {
        OpId toOp = to->id;
        if (m.contains(toOp)) {
          const auto memoryTo = m.getAll(toOp);
          for (auto f : memoryFrom) {
            for (auto t : memoryTo) {
              m.g.constraint(f, t);
            }
          }
        }
      }
    }
  }
}

} // namespace

PoprithmsAliaser getPoprithmsAliaser(const Graph &graph,
                                     DataDependenciesOnly dataDepsOnly) {

  logging::ir::debug("\nGrowing PoprithmsAliaser");
  PoprithmsAliaser m;

  auto scopedStopwatch = graph.getIr().timePartitionLogger().scopedStopwatch(
      "Growing PoprithmsAliaser");

  // NOTE: This loop does not need a schedule that complies with topocons. It
  // may be possible to make this more efficient by getting an Op-order that is
  // only constrained by data order.
  for (auto op : graph.getOpSchedule({}, RequireOptimalSchedule::No)) {
    addAliaserOp(graph, m, op, AllowInsertingProducedTensors::No);
  }

  if (dataDepsOnly == DataDependenciesOnly::No) {

    // Get a vector of Ops.
    const auto &graphOpMap = graph.getOps();
    std::vector<Op *> ops(graphOpMap.size(), nullptr);
    std::transform(
        graphOpMap.begin(),
        graphOpMap.end(),
        ops.begin(),
        [](const auto &entry) -> Op * { return entry.second.get(); });

    addAliaserConstraints(graph, m, ops);
  }

  return m;
}

PoprithmsAliaser getPartialPoprithmsAliaser(const Graph &graph,
                                            const TensorId &tensorId,
                                            DataDependenciesOnly dataDepsOnly) {

  logging::ir::debug("\nGrowing partial PoprithmsAliaser ({})", tensorId);
  PoprithmsAliaser m;

  auto scopedStopwatch = graph.getIr().timePartitionLogger().scopedStopwatch(
      "Growing partial PoprithmsAliaser");

  // When growing a PoprithmsAliaser, by the nature of the Poprithms' API we
  // must grow ops in schedule order. However, we don't know which tensors
  // precede `tensorId` in the schedule order may be aliasing `tensorId`
  // without growing the PoprithmsAliaser first.
  //
  // In this function we overapproximate the Ops that need to be included in the
  // mapping by using the `Op::doesAlias` function, recursively looking at any
  // tensors that alias tensors we have found previously.
  //
  // To see this is an over-approximation, consider the graph:
  //
  //        [t1]
  //         |
  //         v
  //        Op1
  //         |
  //         v
  //        [t2]
  //         |
  //         v
  //        Op2
  //         |
  //         v
  //        [t3]
  //
  // In a graph like this it is not necessarily the case that if `Op::doesAlias`
  // states `t1` aliases `t2` and `t2` aliases `t3` that it follows `t1` aliases
  // `t3`. This is because aliasing may be on disjoint elements of `t2`.

  auto tensor = graph.getTensors().get(tensorId);

  // In the code below we are collating a subset of this graphs ops that need to
  // be grown in the alias graph. We do this in a number of data structures for
  // efficiency. Note we will later use poprithms::schedule::vanilla to
  // determine the order in which they need to be grown so also we maintain
  // mappings from ops to uint64_t (and vice versa) to achieve this.

  // Map containing the subset of ops as keys, mapping them to the uint64_t
  // value we will use in the poprithms::schedule::vanilla call.
  std::map<Op *, uint64_t> opToVanillaNum;

  // Inverse mapping of `opToVanillaNum`.
  std::vector<Op *> vanillaNumToOp;

  // An unordered set that contains the subset of ops.
  //
  // WARNING: This set should only be used for checking membership. That is, the
  // order of unordered sets is non-deterministic -- doubly so with the key type
  // being a a pointer, so these sets must never be iterated over so as to
  // ensure we don't introduce non-determinism in the compilation process. We
  // use this variable because of the constant time containment check, which is
  // better than a std::map.
  std::unordered_set<Op *> opsForContainmentCheck;

  // We find our subset of ops by 'recursively' looking for tensors that alias
  // our tensor. We have a list of tensors to process to allow us to implement
  // this without recursion.
  std::list<Tensor *> processQueue = {tensor};

  // The set of tensors we have already put forward for processing. We need this
  // to ensure we never put forward the same tensor twice. We use an
  // unordered_set for efficiency.
  //
  // WARNING: This set should only be used for checking
  // membership. That is, the order of unordered sets is non-deterministic --
  // doubly so with the key type being a a pointer, so these sets must never be
  // iterated over so as to ensure we don't introduce non-determinism in the
  // compilation process. We use this variable because of the constant time
  // containment check, which is better than a std::map.
  std::unordered_set<Tensor *> processedTensors;

  while (!processQueue.empty()) {

    // Get the front-most tensor.
    Tensor *t = processQueue.front();
    processQueue.pop_front();

    if (t->hasProducer()) {
      // Find the Op that produces 't' and process any of the inputs of said ops
      // that alias 't'.
      Op *producer = t->getProducer();

      // Flag to keep track if we should add this producer to the mapping.
      bool includeProducer = true;

      // Get the indices on which this producer outputs `t` (there is only 1).
      for (int outputIndex : producer->output->indices(t)) {
        // For every input, check if it can alias with the output.
        for (const auto &inputEntry : producer->input->tensorMap()) {
          const auto &inputIndex  = inputEntry.first;
          const auto &inputTensor = inputEntry.second;
          if (producer->doesAlias(inputIndex, outputIndex)) {
            // OK, this input may alias. So we need to include this op and
            // process the input tensor if we haven't already.
            includeProducer = true;

            if (processedTensors.find(inputTensor) == processedTensors.end()) {
              // Mechanism to ensure we never process the same tensor twice.
              processQueue.push_back(inputTensor);
              processedTensors.insert(inputTensor);
            }
          }
        }
      }

      if (includeProducer) {
        if (opsForContainmentCheck.find(producer) ==
            opsForContainmentCheck.end()) {
          // The producer is an op we need to grow. We will use
          // `vanillaNumToOp.size()` as the number to represent it when
          // scheduling in poprithms::schedule::vanilla.
          opToVanillaNum.insert({producer, vanillaNumToOp.size()});
          vanillaNumToOp.push_back(producer);
          opsForContainmentCheck.insert(producer);
        }
      }
    }

    // Iterate over all consumers.
    for (Op *consumer : t->consumers.getOps()) {
      // Flag to keep track if we should add this consumer to the mapping.
      bool includeConsumer = true;

      // Get the indices on which the consumer consumes `t`.
      for (int inputIndex : consumer->input->indices(t)) {
        // For every output, check if it can alias with the output.
        for (const auto &outputEntry : consumer->output->tensorMap()) {
          const auto &outputIndex  = outputEntry.first;
          const auto &outputTensor = outputEntry.second;
          if (consumer->doesAlias(inputIndex, outputIndex)) {
            // OK, this output may alias. So we need to include this op and
            // process the output tensor if we haven't already.
            includeConsumer = true;

            if (processedTensors.find(outputTensor) == processedTensors.end()) {
              // Mechanism to ensure we never process the same tensor twice.
              processQueue.push_back(outputTensor);
              processedTensors.insert(outputTensor);
            }
          }
        }
      }

      if (includeConsumer) {
        if (opsForContainmentCheck.find(consumer) ==
            opsForContainmentCheck.end()) {
          // The consumer is an op we need to grow. We will use
          // `vanillaNumToOp.size()` as the number to represent it when
          // scheduling in poprithms::schedule::vanilla.
          opToVanillaNum.insert({consumer, vanillaNumToOp.size()});
          vanillaNumToOp.push_back(consumer);
          opsForContainmentCheck.insert(consumer);
        }
      }
    }
  }

  // We have the subsets of ops we want to grow but we don't yet know the order
  // to grow them in. We don't want to call `graph.getOpSchedule` to get this
  // order because scheduling the whole graph is quite inefficient in some
  // cases. Instead, we use poprithms::schedule::vanilla to ensure we call
  // addAliaserOp in data order (topocons don't matter).

  // Nothing to do if there are no ops.
  if (!vanillaNumToOp.empty()) {

    // Here we construct an edges data structure so that we can obtain a
    // schedule of our ops in data order; starting with an empty transition
    // relation. The meaning of this vector of vectors is that an entry {4, 7}
    // at index 9 implies there is an edge 9->4 and an edge 9->7.
    poprithms::schedule::vanilla::Edges<uint64_t> vanillaEdges(
        vanillaNumToOp.size(), std::vector<uint64_t>());

    // Iterate over all the included ops and if any of producers of an input of
    // an op are also included in our subset of ops then include them as a
    // constraint to `vanillaEdges`.
    for (const auto &opToVanillaNumEntry : opToVanillaNum) {

      Op *op              = opToVanillaNumEntry.first;
      uint64_t vanillaNum = opToVanillaNumEntry.second;

      for (auto inputEntry : op->input->tensorMap()) {
        Tensor *t = inputEntry.second;
        if (t->hasProducer()) {
          Op *producer = t->getProducer();

          // The find on an std::map could be expensive, so use our faster
          // containment check first.
          if (opsForContainmentCheck.find(producer) !=
              opsForContainmentCheck.end()) {
            auto producerIt = opToVanillaNum.find(producer);
            if (producerIt != opToVanillaNum.end()) {
              // It's possible we're adding the same edge more than once but
              // it's likely more efficient to let poprithms deal with this than
              // to check for duplication here.
              auto &producerFwdEdges = vanillaEdges.at(producerIt->second);
              producerFwdEdges.push_back(vanillaNum);
            } else {
              throw internal_error("[PoprithmsAliaser] Unexpectedly unable to "
                                   "find {} in 'opToVanillaNum'",
                                   producer->str());
            }
          }
        }
      }
    }

    // Use poprithms to get the ops in data order.
    auto vanillaSchedule = poprithms::schedule::vanilla::getSchedule_u64(
        vanillaEdges,
        poprithms::schedule::vanilla::ErrorIfCycle::Yes,
        poprithms::schedule::vanilla::VerifyEdges::No);

    // Grow ops in data order.
    for (uint64_t vanillaNum : vanillaSchedule) {
      // Map poprithms number back to Op.
      Op *op = vanillaNumToOp.at(vanillaNum);
      // Add Op to the alias model.
      addAliaserOp(graph, m, op, AllowInsertingProducedTensors::Yes);
    }
  }

  // Make sure our tensor is represented.
  addTensor(graph, m, tensor, AllowInsertingProducedTensors::Yes);

  if (dataDepsOnly == DataDependenciesOnly::No) {
    // Add topocons to poprithms graph.
    addAliaserConstraints(graph, m, vanillaNumToOp);
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
