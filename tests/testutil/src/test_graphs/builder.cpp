#include <testutil/test_graphs/builder.hpp>

#include <popart/error.hpp>
#include <popart/filereader.hpp>
#include <popart/graph.hpp>
#include <popart/graphid.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/scheduler.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/topocons.hpp>

#include <map>
#include <memory>
#include <type_traits>
#include <vector>

using namespace popart;

namespace test_graphs {
namespace builder {

namespace {

/**
 * \brief Delete a std container. That is, destroy the objects and actually
 * reclaim the memory of the vector itself.
 *
 * \tparam C Container type, e.g. std::vector<int>. Must be default
 * constructable and have a `swap(C)` method.
 * \param c lvalue of type C that will be deleted.
 */
template <typename C> void deleteContainer(C &c) {
  // Delete by swapping with empty container.
  using D = std::decay_t<C>;
  D{}.swap(c);
}

void topoSort(std::vector<OpId> &opIds,
              const Graph &graph,
              const std::vector<OpOpEdge> &opOps,
              const std::vector<TensorOpEdge> &tenOps,
              const std::vector<OpTensorEdge> &opTens);

} // namespace

void withEdges(popart::Graph &graph,
               std::vector<std::unique_ptr<Op>> &upOps,
               const std::vector<OpOpEdge> &opOps,
               const std::vector<TensorOpEdge> &tenOps,
               const std::vector<OpTensorEdge> &opTens,
               const std::multimap<OpId, OpId> &topoCons) {

  std::vector<OpId> newOpIds;
  newOpIds.reserve(upOps.size());

  for (auto &&upOp : upOps) {
    newOpIds.push_back(upOp->id);
    graph.moveIntoGraph(std::move(upOp));
  }
  deleteContainer(upOps); // no longer needed.

  // outstanding[i] = n <=> Op with id newOpIds[i] has n incoming edges.
  std::vector<OpId> outstandingByIdx(newOpIds.size(), 0);
  std::vector<std::vector<OpId>> edgesByIdx;
  edgesByIdx.reserve(newOpIds.size());

  /*
    only if startOp and endOp in newOpIds, add to edges
  */

  // Helper for creating tensor ids.
  const auto mkOutId = [](const OpId &opId, const OutIndex outIdx) -> TensorId {
    return std::to_string(opId) + "-output-" + std::to_string(outIdx);
  };

  for (const auto &opOp : opOps) {
    auto startOp = graph.getOp(opOp.startId);

    // If tensor doesn't exist yet, create it (a tensor can be consumed by
    // multiple ops).
    if (!startOp->output->hasIndex(opOp.outIdx)) {
      startOp->createAndConnectOutTensor(opOp.outIdx,
                                         mkOutId(opOp.startId, opOp.outIdx));
    }

    const auto tId = startOp->output->id(opOp.outIdx);

    auto endOp = graph.getOp(opOp.endId);
    if (!endOp->input->hasIndex(opOp.inIdx)) {
      endOp->connectInTensor(opOp.inIdx, tId);
    } else {
      throw error("test_graphs::builder::withEdges: Attempting to connect "
                  "input tensor {} at index {} to Op {} that already has an "
                  "input tensor at that index: {}.",
                  tId,
                  opOp.inIdx,
                  endOp->id,
                  endOp->debugName());
    }
  }

  for (const auto &tenOp : tenOps) {
    if (!graph.getTensors().contains(tenOp.startId)) {
      throw error(
          "test_graphs::builder::withEdges: No tensor with given startId for "
          "`TensorOpEdge{{startId = {}, endId = {}, inIdx = {} }}`.",
          tenOp.startId,
          tenOp.endId,
          tenOp.inIdx);
    }

    const auto endOp = graph.getOp(tenOp.endId);
    endOp->connectInTensor(tenOp.inIdx, tenOp.startId);
  }

  for (const auto &opTen : opTens) {
    const auto startOp = graph.getOp(opTen.startId);

    // If endId given, assume that tensor exists and connect to it; else, use
    // createAndConnectOutTensor to create the tensor too.
    if (opTen.endId) {
      if (!graph.getTensors().contains(*opTen.endId)) {
        throw error(
            "test_graphs::builder::withEdges: No tensor found with given endId "
            "for `OpTensorEdge{{startId = {}, endId = {}, outIdx = {} }}`.",
            opTen.startId,
            *opTen.endId,
            opTen.outIdx);
      }

      startOp->connectOutTensor(opTen.outIdx, *opTen.endId);
    } else {
      startOp->createAndConnectOutTensor(opTen.outIdx,
                                         mkOutId(startOp->id, opTen.outIdx));
    }
  }

  // Once all the connections have been inserted, we can setup all the new ops.
  topoSort(newOpIds, graph, opOps, tenOps, opTens);
  for (const auto id : newOpIds) {
    graph.getOp(id)->setup();
  }

  // Finally, add the additional topo cons.
  for (const auto &tc : topoCons) {
    graph.topoCons->insert(graph.getOp(tc.first), graph.getOp(tc.second));
  }
}

namespace {
/**
 * Topologically sorts `opIds` according to the described edges.
 *
 * Will sort `opIds` inplace, but not in O(1) space.
 */
void topoSort(std::vector<OpId> &opIds,
              const Graph &graph,
              const std::vector<OpOpEdge> &opOps,
              const std::vector<TensorOpEdge> &tenOps,
              const std::vector<OpTensorEdge> &opTens) {

  /* 1. Build edge map */
  std::unordered_multimap<OpId, OpId> edges(opIds.size());
  {
    const auto isNewOp = [&opIds](const OpId opId) -> bool {
      return std::find(opIds.begin(), opIds.end(), opId) != opIds.end();
    };

    for (const auto &opOp : opOps) {
      edges.insert({opOp.startId, opOp.endId});
    }

    // There could be separate Op->Ten and Ten->Op edges with the same tensor,
    // thus implicitly creating an Op->Op edge. We only care about when both ops
    // are new. Using this and the fact that tensors already store their SINGLE
    // producer op, we can perform the below simplified algorithm.
    for (const auto &tenOp : tenOps) {
      const auto t = graph.getTensors().get(tenOp.startId);
      if (t->hasProducer()) {
        const auto startOpId = t->getProducerUnsafe()->id;
        if (isNewOp(startOpId)) {
          edges.insert({startOpId, tenOp.endId});
        }
      }
    }
  }

  /* 2. Perform topological sort on `opIds`. */
  {
    const auto nOps = opIds.size();

    // Build up outstanding and ready maps.
    std::unordered_map<OpId, int> outstanding(nOps);
    for (const auto &i : opIds) {
      outstanding[i] = 0;
    }
    for (const auto &e : edges) {
      outstanding[e.second] += 1;
    }
    std::vector<OpId> ready;
    for (const auto &entry : outstanding) {
      if (entry.second == 0) {
        ready.push_back(entry.first);
      }
    }

    // Perform Kahn's algo.

    // We will overwrite OpIds with the ops in schedule order. We can do this
    // because we have them all stored in other data structures too.
    opIds.clear();

    while (!ready.empty()) {
      const auto i = ready.back();
      ready.pop_back();

      opIds.push_back(i);

      auto range     = edges.equal_range(i);
      const auto end = range.second;
      for (auto cur = range.first; cur != end; cur = std::next(cur)) {
        const auto j = cur->second;

        --outstanding[j];
        if (outstanding[j] == 0) {
          ready.push_back(j);
        }
      }
    }

    if (opIds.size() != nOps) {
      throw error("test_graphs::builder::withEdges: Could not schedule edges.");
    }
  }
}
} // namespace

} // namespace builder
} // namespace test_graphs
