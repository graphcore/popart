#include <popart/alias/aliasmodelgrower.hpp>

#include <list>

#include <poprithms/logging/timepartitionlogger.hpp>
#include <poprithms/schedule/vanilla/vanilla.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/topocons.hpp>

#include <popart/alias/aliasmodel.hpp>

namespace popart {

AliasModelGrower::AliasModelGrower(AliasModel &aliasModel_)
    : aliasModel{aliasModel_} {}

AliasModel &AliasModelGrower::getAliasModelRef() { return aliasModel.get(); }

void AliasModelGrower::growFullGraph(const Graph &graph,
                                     DataDependenciesOnly dataDepsOnly) {
  logging::ir::debug("\nGrowing AliasModel");

  auto scopedStopwatch =
      graph.getIr().timePartitionLogger().scopedStopwatch(logging::format(
          "Growing full AliasModel for {}", graph.getGraphString()));

  // NOTE: This loop does not need a schedule that complies with topocons. It
  // may be possible to make this more efficient by getting an Op-order that is
  // only constrained by data order.
  for (auto op : graph.getOpSchedule({}, RequireOptimalSchedule::No)) {
    addAliaserOp(graph, op, AllowInsertingProducedTensors::No);
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

    addAliaserConstraints(graph, ops);
  }
}

void AliasModelGrower::growPartialGraph(const Graph &graph,
                                        const TensorId &tensorId,
                                        DataDependenciesOnly dataDepsOnly) {

  logging::ir::debug("\nGrowing partial AliasModel ({})", tensorId);

  auto scopedStopwatch = graph.getIr().timePartitionLogger().scopedStopwatch(
      "Growing partial AliasModel");

  // When growing a AliasModel, by the nature of the Poprithms' API we
  // must grow ops in schedule order. However, we don't know which tensors
  // that precede `tensorId` in the schedule order may be aliasing `tensorId`
  // without growing the AliasModel first.
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
              throw internal_error("[AliasModel] Unexpectedly unable to "
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
      addAliaserOp(graph, op, AllowInsertingProducedTensors::Yes);
    }
  }

  // Make sure our tensor is represented.
  addTensor(graph, tensor, AllowInsertingProducedTensors::Yes);

  if (dataDepsOnly == DataDependenciesOnly::No) {
    // Add topocons to poprithms graph.
    addAliaserConstraints(graph, vanillaNumToOp);
  }
}

void AliasModelGrower::addTensor(const Graph &graph,
                                 Tensor *t,
                                 AllowInsertingProducedTensors allow) {
  if (!aliasModel.get().contains(t->id)) { // to_.find(t0->id) == m.to.cend()) {

    if (allow == AllowInsertingProducedTensors::No && t->hasProducer()) {
      // The tensor should have been added by growing it's producer.
      throw error("[AliasModel] Tensor '{}' was not added to the alias "
                  "model by it's producer, {}",
                  t->id,
                  t->getProducer()->str());
    }

    const auto inId = (t->tensorType() == TensorType::Const)
                          ? aliasModel.get().g.constant(t->info.shape())
                          : aliasModel.get().g.variable(t->info.shape());

    aliasModel.get().insertTensor(inId, *t);
  }
}

void AliasModelGrower::addAliaserOp(const Graph &graph,
                                    Op *op,
                                    AllowInsertingProducedTensors allow) {

  logging::ir::trace("Inserting input Tensors for AliasModel, Graph \"{}\"",
                     graph.id);
  for (const auto &x : op->input->tensorMap()) {

    // If it's a new input tensor, register it:
    const auto t0 = x.second;
    addTensor(graph, t0, allow);
  }

  logging::ir::trace("Growing AliasModel for op\"{}\"", op->str());
  op->growAliasModel(aliasModel.get());

  // Check every output has been mapped.
  for (const auto &x : op->output->tensorMap()) {
    const auto t1 = x.second;
    if (!aliasModel.get().contains(t1->id)) {
      logging::ir::trace("Op {} failed to add mapping for output #{} ('') "
                         "in AliasModel",
                         op->str(),
                         x.first,
                         t1->id);
    }
  }

  for (auto x : aliasModel.get().getAll(op->id)) {
    logging::ir::trace(
        "Setting name for op \"{}\", poprithms OpId={}", op->str(), x);
    aliasModel.get().g.setName(x, op->str());
  }
}

void AliasModelGrower::addAliaserConstraints(
    const Graph &graph,
    const std::vector<Op *> &opSubset) {

  logging::ir::trace("Inserting constraints for AliasModel, Graph \"{}\"",
                     graph.id);

  // Insert constraints:
  for (auto from : opSubset) {
    const auto tos        = graph.topoCons->getAfters(from);
    OpId fromOp           = from->id;
    const auto memoryFrom = aliasModel.get().getAll(fromOp);
    for (auto to : tos) {
      if (from != to) {
        OpId toOp = to->id;
        if (aliasModel.get().contains(toOp)) {
          const auto memoryTo = aliasModel.get().getAll(toOp);
          for (auto f : memoryFrom) {
            for (auto t : memoryTo) {
              aliasModel.get().g.constraint(f, t);
            }
          }
        }
      }
    }
  }
}

} // namespace popart
