#include <testutil/test_graphs/dummy_builder.hpp>

#include <testutil/test_graphs/op/dummy.hpp>

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

#include <memory>
#include <string>

using namespace popart;

namespace test_graphs {
namespace dummy_builder {

popart::InIndex NoneInIndex   = -1;
popart::OutIndex NoneOutIndex = -1;

namespace {
void transferProperties(const Op *from, Op *to);
} // namespace

const popart::TensorInfo &defaultTensorInfo() {
  static const TensorInfo defaultTensorInfo = {
      //
      "FLOAT",
      std::vector<int64_t>{4, 4, 4, 4} //
  };
  return defaultTensorInfo;
}

void withEdges(Graph &graph,
               const std::vector<std::vector<OpId>> &edges,
               const std::multimap<OpId, OpId> &topoCons,
               const TensorInfo &tensorInfo) {

  const auto nOps = edges.size();

  /* 1. Create all the ops, assign their opIds, and move into graph. */

  const auto settings = [&graph](const OpId opId) {
    return Op::Settings{graph, "Dummy" + std::to_string(opId)};
  };

  for (auto opId = 0; opId < nOps; opId++) {
    auto op = std::make_unique<DummyOp>(graph, settings(opId));
    op->id  = opId;
    graph.moveIntoGraph(std::move(op));
  }

  /*
    2. Construct the graph.

    To do this, we essentially do a simple scheduling pass (kahn's algo) over
    the edges, constructing the tensors and connecting them to the ops as we go
    along.
  */

  // First, some helpers.

  const auto connectInTensor = [&graph](const OpId opId, const TensorId tId) {
    auto *op = dynamic_cast<DummyOp *>(graph.getOp(opId));
    op->connectInTensor(op->getNextInIndex(), tId);
  };

  const auto createAndConnectOutTensorAndConnectToConsumers =
      [&graph, &edges, connectInTensor](const OpId opId) {
        auto *op = dynamic_cast<DummyOp *>(graph.getOp(opId));

        const auto tOutId = TensorId{std::to_string(opId) + "-output"};

        op->createAndConnectOutTensor(DummyOp::getOutIndex(), tOutId);
        op->setup();

        for (const auto &out : edges[opId]) {
          connectInTensor(out, tOutId);
        }
      };

  // Initialise the scheduling pass.

  std::vector<OpId> outstanding(nOps, 0);
  // Lowest OpId is at front of queue.
  std::vector<OpId> ready;

  // Compute, for each op, how many incoming edges (dependencies) it has.
  for (const auto &consumersOfOp : edges) {
    for (const auto j : consumersOfOp) {
      outstanding[j]++;
    }
  }

  // Mark those ops with 0 dependencies as ready.
  // Note, in our scheduling pass, when an op gets scheduled, we create its
  // output tensor and hook it up as an input to the op's consumers. We do not
  // create the inputs; the invariant is that the input tensors will already
  // exist, created when their producer was scheduled. Thus, as part of
  // initialising the scheduling pass, we need to create the input ops' input
  // tensors.
  for (OpId i = 0; i < nOps; i++) {
    if (outstanding[i] == 0) {
      ready.push_back(i);

      const TensorId tId{std::to_string(i) + "-input"};
      graph.addInput(tId, tensorInfo);
      connectInTensor(i, tId);
    }
  }

  int nScheduled = 0;

  while (!ready.empty()) {
    const OpId i = ready.back();
    ready.pop_back();

    nScheduled++;

    createAndConnectOutTensorAndConnectToConsumers(i);

    for (const auto &j : edges[i]) {
      --outstanding[j];

      if (outstanding[j] == 0) {
        ready.push_back(j);
      }
    }
  }

  // Done!

  if (nScheduled != static_cast<int>(nOps)) {
    throw error("test_graphs::dummy_builder::withEdges: Proposed graph is not "
                "schedulable.");
  }

  /* 3. Now we've built the graph, add all the topo cons. */

  for (const auto &tc : topoCons) {
    graph.topoCons->insert(graph.getOp(tc.first), graph.getOp(tc.second));
  }
}

VerticesDisconnectedByReplacement
replaceOp(popart::Graph &graph,
          const popart::OpId opIdToReplace,
          std::unique_ptr<popart::Op> newOpUp,
          const std::vector<popart::InIndex> mapInputsToNewOp,
          const std::vector<popart::OutIndex> mapOutputsToNewOp) {

  Op *oldOp = nullptr;
  try {
    oldOp = graph.getOp(opIdToReplace);
  } catch (const error &err) {
    throw error("test_graphs::dummy_builder::replaceOp: Could not find op to "
                "replace in provided graph. Error from Popart was:\n\n{}",
                err.what());
  }

  Op *newOp = newOpUp.get();
  if (!newOp) {
    throw error("test_graphs::dummy_builder::replaceOp: newOp is null.");
  }

  // 1. Transfer properties.
  transferProperties(oldOp, newOp);

  VerticesDisconnectedByReplacement disconnectedVertices;

  // 2. Reconnect input tensors.

  const auto nIn = mapInputsToNewOp.size();

  for (InIndex oldIn = 0; oldIn < nIn; oldIn++) {
    const auto newIn = mapInputsToNewOp[oldIn];

    auto tensor = oldOp->inTensor(oldIn);
    oldOp->disconnectInTensor(oldIn);

    if (newIn != NoneInIndex) {
      newOp->connectInTensor(newIn, tensor->id);
    } else {
      disconnectedVertices.inTensors.emplace(std::make_pair(oldIn, tensor));
    }
  }

  // 3. Reconnect output tensors.

  const auto nOut = mapOutputsToNewOp.size();

  for (OutIndex oldOut = 0; oldOut < nOut; oldOut++) {
    const auto newOut = mapOutputsToNewOp[oldOut];

    auto tensor = oldOp->outTensor(oldOut);
    oldOp->disconnectOutTensor(tensor);

    if (newOut != NoneOutIndex) {
      newOp->connectOutTensor(newOut, tensor->id);
    } else {
      disconnectedVertices.outTensors.emplace(std::make_pair(oldOut, tensor));
    }
  }

  disconnectedVertices.op = oldOp;
  graph.moveIntoGraph(std::move(newOpUp));

  return disconnectedVertices;
}

namespace {

void transferProperties(const Op *from, Op *to) {
  // We want to replace id.
  to->id               = from->id;
  to->toLoss           = from->toLoss;
  to->fromLoss         = from->fromLoss;
  to->scheduledPreLoss = from->scheduledPreLoss;
  to->pruneable        = from->pruneable;
  // Preserve to->debugInfo. In the future, we should copy certain aspects
  // of from->debugInfo, but this is not possible yet.

  // Op::Settings we don't want to overwrite.
  auto savedName              = std::move(to->settings.name);
  const auto savedDebugInfoId = to->settings.debugInfoId;
  const auto savedOptimizerOp = to->settings.optimizerOp;

  // Can't move because we don't know that the user doesn't need from op.
  to->settings = from->settings;

  to->settings.name        = std::move(savedName);
  to->settings.debugInfoId = savedDebugInfoId;
  to->settings.optimizerOp = savedOptimizerOp;
}

} // namespace

} // namespace dummy_builder
} // namespace test_graphs
