// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <boost/algorithm/string.hpp>

#include <poprithms/logging/timepartitionlogger.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/exchange/multiexchange.hpp>
#include <popart/op/exchange/remote.hpp>
#include <popart/op/reducesumsquare.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensornames.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/accumulateouterfragmentparallelizer.hpp>
#include <popart/transforms/clipweightgradientsbynorm.hpp>

#include <algorithm>

using namespace std;

namespace popart {
namespace {
// An efficient way to check for intersection in sorted ranges.
template <class SortedContainerType>
bool efficientOverlapCheck(const SortedContainerType &set1,
                           const SortedContainerType &set2) {
  auto it1 = set1.begin();
  auto it2 = set2.begin();
  while (it1 != set1.end() && it2 != set2.end()) {
    if (*it1 < *it2)
      ++it1;
    else if (*it2 < *it1)
      ++it2;
    else
      return true;
  }
  return false;
}

bool isOptimizerLikeTensor(Tensor *t) {
  if (t->isOptimizerTensor()) {
    return true;
  }
  // Scaled Optimizer State
  return t->idIncludesPrefix(
      {reservedPreviousLossScalingPrefix(), reservedLossScalingRatioPrefix()});
}

} // namespace

AccumulateOuterFragmentParallelizer::OpCluster::OpCluster(const Graph *graph,
                                                          Op *op)
    : graph{graph}, ops{op}, adjacentOps{}, remoteLoadOps{}, remoteStoreOps{},
      virtualGraphId{op->hasVirtualGraphId() ? op->getVirtualGraphId()
                                             : unusedVGraphId},
      tensorIds{} {
  gatherTensorIds(op, tensorIds);

  // Populate adjacentOps.
  adjacentOps.clear();
  for (auto op : ops) {
    auto afters = graph->topoCons->getAfters(op);
    adjacentOps.insert(adjacentOps.end(), afters.begin(), afters.end());
    auto befores = graph->topoCons->getBefores(op);
    adjacentOps.insert(adjacentOps.end(), befores.begin(), befores.end());
  }
  std::sort(adjacentOps.begin(), adjacentOps.end());

  // Populate remoteLoadOps
  remoteLoadOps.clear();
  std::copy_if(ops.begin(),
               ops.end(),
               std::back_inserter(remoteLoadOps),
               [](Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); });

  // Populate remoteStoreOps
  remoteStoreOps.clear();
  std::copy_if(ops.begin(),
               ops.end(),
               std::back_inserter(remoteStoreOps),
               [](Op *op) { return op->isConvertibleTo<RemoteStoreOp>(); });

  // Populate remoteStoreOps
  multiExchangeOps.clear();
  std::copy_if(ops.begin(),
               ops.end(),
               std::back_inserter(multiExchangeOps),
               [](Op *op) { return op->isConvertibleTo<MultiExchangeOp>(); });

  // Calculate numLoadBytes.
  numLoadBytes = calcNumLoadBytes();

  // Populate loadShapes.
  loadShapes.clear();
  for (auto remoteLoadOp : remoteLoadOps) {
    // RemoteLoadOp will do exactly one load.
    loadShapes.insert(
        remoteLoadOp->outInfo(RemoteLoadInplaceOp::getLocalTensorOutIndex())
            .shape());
  }
  for (auto MultiExchangeOp : multiExchangeOps) {
    // Exchange can multiple loads as well as stores. Work out number of loads
    // by number of output tensors.
    auto numLoads = MultiExchangeOp->output->tensors().size();
    for (size_t load = 0; load < numLoads; ++load) {
      loadShapes.insert(MultiExchangeOp->outInfo(load).shape());
    }
  }
}

void AccumulateOuterFragmentParallelizer::OpCluster::gatherTensorIds(
    const Op *op,
    TensorIds &tensorIds) {
  tensorIds.reserve(op->input->tensors().size() + op->output->tensors().size());
  for (const auto tensor : op->input->tensors()) {
    if (!isOptimizerLikeTensor(tensor)) {
      tensorIds.push_back(tensor->id);
    }
  }
  for (const auto tensor : op->output->tensors()) {
    if (!isOptimizerLikeTensor(tensor)) {
      tensorIds.push_back(tensor->id);
    }
  }
  std::sort(tensorIds.begin(), tensorIds.end());
}

bool AccumulateOuterFragmentParallelizer::OpCluster::overlaps(
    const OpCluster &rhs) const {
  // Check if the same tensors are used. Trying to avoid using a double loop
  // here for efficiency reasons.
  if (efficientOverlapCheck(rhs.tensorIds, tensorIds)) {
    return true;
  }

  // Check there are any topological constraints between the two clusters by
  // checking if the ops in rhs are in our adjacent set.
  auto option1Size = rhs.ops.size() + adjacentOps.size();
  auto option2Size = ops.size() + rhs.adjacentOps.size();
  if (option1Size < option2Size) {
    if (efficientOverlapCheck(rhs.ops, adjacentOps)) {
      return true;
    }
  } else {
    if (efficientOverlapCheck(ops, rhs.adjacentOps)) {
      return true;
    }
  }

  return false;
}

bool AccumulateOuterFragmentParallelizer::OpCluster::hasRemoteOp() const {
  return std::any_of(ops.begin(), ops.end(), [](Op *op) {
    return op->isConvertibleTo<RemoteLoadOp>() ||
           op->isConvertibleTo<RemoteStoreOp>() ||
           op->isConvertibleTo<MultiExchangeOp>();
  });
}

void AccumulateOuterFragmentParallelizer::OpCluster::absorb(
    const OpCluster &rhs) {
  ops.insert(ops.end(), rhs.ops.begin(), rhs.ops.end());
  std::sort(ops.begin(), ops.end());
  adjacentOps.insert(
      adjacentOps.end(), rhs.adjacentOps.begin(), rhs.adjacentOps.end());
  std::sort(adjacentOps.begin(), adjacentOps.end());
  tensorIds.insert(tensorIds.end(), rhs.tensorIds.begin(), rhs.tensorIds.end());
  std::sort(tensorIds.begin(), tensorIds.end());
  remoteLoadOps.insert(
      remoteLoadOps.end(), rhs.remoteLoadOps.begin(), rhs.remoteLoadOps.end());
  remoteStoreOps.insert(remoteStoreOps.end(),
                        rhs.remoteStoreOps.begin(),
                        rhs.remoteStoreOps.end());
  multiExchangeOps.insert(multiExchangeOps.end(),
                          rhs.multiExchangeOps.begin(),
                          rhs.multiExchangeOps.end());
  numLoadBytes = calcNumLoadBytes();
  loadShapes.insert(rhs.loadShapes.begin(), rhs.loadShapes.end());
}

bool AccumulateOuterFragmentParallelizer::OpCluster::hasVirtualGraphId() const {
  if (!ops[0]->hasVirtualGraphId()) {
    return false;
  } else {
    auto vid = ops[0]->getVirtualGraphId();
    for (auto op : ops) {
      if (!op->hasVirtualGraphId() || op->getVirtualGraphId() != vid) {
        return false;
      }
    }
    return true;
  }
}

VGraphId
AccumulateOuterFragmentParallelizer::OpCluster::getVirtualGraphId() const {
  return ops[0]->hasVirtualGraphId() ? ops[0]->getVirtualGraphId()
                                     : unusedVGraphId;
}

int64_t
AccumulateOuterFragmentParallelizer::OpCluster::calcNumLoadBytes() const {
  // Calculate number of load bytes. If ops from multiple ops are joined (which
  // will be the case when working out bin constraints for parallelised
  // clusters, take the max over the virtual graph id).
  std::map<int64_t, int64_t> loadBytesPerVgid;

  for (auto remoteLoadOp : remoteLoadOps) {
    // RemoteLoadOp will do exactly one load.
    loadBytesPerVgid[remoteLoadOp->hasVirtualGraphId()
                         ? remoteLoadOp->getVirtualGraphId()
                         : unusedVGraphId] +=
        remoteLoadOp->outInfo(RemoteLoadInplaceOp::getLocalTensorOutIndex())
            .nbytes();
  }
  for (auto MultiExchangeOp : multiExchangeOps) {
    // Exchange can multiple loads as well as stores. Work out number of loads
    // by number of output tensors.
    auto numLoads = MultiExchangeOp->output->tensors().size();
    for (size_t load = 0; load < numLoads; ++load) {
      loadBytesPerVgid[MultiExchangeOp->hasVirtualGraphId()
                           ? MultiExchangeOp->getVirtualGraphId()
                           : unusedVGraphId] +=
          MultiExchangeOp->outInfo(load).nbytes();
    }
  }

  int64_t result = 0ll;
  for (auto entry : loadBytesPerVgid) {
    if (entry.second > result) {
      result = entry.second;
    }
  }

  return result;
}

size_t AccumulateOuterFragmentParallelizer::id() {
  return typeid(AccumulateOuterFragmentParallelizer).hash_code();
}

AccumulateOuterFragmentParallelizer::AccumulateOuterFragmentParallelizer()
    : Transform{} {}

AccumulateOuterFragmentParallelizer::~AccumulateOuterFragmentParallelizer() {}

bool AccumulateOuterFragmentParallelizer::apply(Graph &graph) const {

  const auto scopedStopwatch =
      graph.getIr().timePartitionLogger().scopedStopwatch(
          "AccumulateOuterFragmentParallelizer::apply");

  OpClustersMap clustersMap;
  // Get groups of ops that update optimizer state / weights.
  populateOpClusters(graph, clustersMap);
  // Filter out groups without remote ops.
  filterOpClusters(clustersMap);
  // Sort the groups by combined loaded bytes.
  sortOpClusters(clustersMap);

  // Settings object.
  const auto &settings =
      graph.getIr().getSessionOptions().accumulateOuterFragmentSettings;

  // Throw away clusters for vgids we don't want to parallelize.
  const auto &vgids = settings.excludedVirtualGraphs;
  for (const auto &vgid : vgids) {
    clustersMap.erase(vgid);
  }

  // Now we add things to the schedule to try and schedule
  // RemoteLoadOp/RemoteStoreOp together.
  while (!clustersMap.empty()) {
    // Pass around iterators to save a copy.
    OpClusters opClustersToParallelize;

    // Find clusters to parallelise. Remove those clusters from the clustersMap
    // to ensure the loop terminates.

    if (settings.schedule ==
        AccumulateOuterFragmentSchedule::OverlapCycleOptimized) {
      // If we're optimizing for time we just want to schedule one from each
      // virtual graph in parallel in each step, in increasing order of memory.
      for (auto it1 = clustersMap.begin(); it1 != clustersMap.end();) {
        auto &clusters = it1->second;
        if (!clusters.empty()) {
          opClustersToParallelize.push_back(clusters.front());
          clusters.pop_front();
        }

        // Update it1.
        if (clusters.empty()) {
          it1 = clustersMap.erase(it1);
        } else {
          ++it1;
        }
      }

    } else if (settings.schedule ==
               AccumulateOuterFragmentSchedule::OverlapMemoryOptimized) {
      // If we are trying to be gentle on memory usage we should only
      // parallelise those weight updates that are over identically-shaped
      // tensors, so that the resulting combined remote operations (see
      // MergeExchange) are better suited to outlining.

      // Find smallest.
      auto hasSmallest          = false;
      auto smallestNumLoadBytes = 0;
      auto smallestLoadShapes   = std::set<Shape>();
      for (auto it = clustersMap.begin(); it != clustersMap.end(); ++it) {
        auto &clusters = it->second;
        if (!clusters.empty() &&
            (!hasSmallest ||
             clusters.front().numLoadBytes < smallestNumLoadBytes)) {
          hasSmallest          = true;
          smallestNumLoadBytes = clusters.front().numLoadBytes;
          smallestLoadShapes   = clusters.front().loadShapes;
        }
      }

      // Only parallelise clusters with exact same tensor shapes.
      for (auto it1 = clustersMap.begin(); it1 != clustersMap.end();) {
        auto &clusters = it1->second;
        for (auto it2 = clusters.begin(); it2 != clusters.end(); ++it2) {
          if (it2->loadShapes == smallestLoadShapes) {
            opClustersToParallelize.push_back(*it2);
            clusters.erase(it2);
            break;
          }
        }

        // Update it1.
        if (clusters.empty()) {
          it1 = clustersMap.erase(it1);
        } else {
          ++it1;
        }
      }
    } else {
      throw error("Unable to process schedule setting");
    }

    tryToParallelizeOpClusters(graph, opClustersToParallelize);
  }

  return false;
}

vector<vector<Op *>> AccumulateOuterFragmentParallelizer::getBinConstraints(
    const Graph &graph) const {

  vector<vector<Op *>> binConstraints;

  OpClusters clusters;
  // Get groups of ops that update optimizer state / weights.
  populateOpClusters(graph, clusters);

  // Sort the groups by combined loaded bytes.
  sortOpClusters(clusters);

  vector<vector<Op *>> postBinConstaints;
  for (auto cluster : clusters) {
    if (cluster.tensorIds.empty()) {
      // If a cluster's tensorIds is empty then it only
      // consumes/produces optimizerLikeTensors. These should be placed at
      // the start to avoid scheduling conflicts between data and
      // bin constraints.
      binConstraints.push_back(cluster.ops);
    } else {
      postBinConstaints.push_back(cluster.ops);
    }
  }

  binConstraints.insert(
      binConstraints.end(), postBinConstaints.begin(), postBinConstaints.end());

  std::string debugString;
  for (size_t constraintIdx = 0; constraintIdx < binConstraints.size();
       constraintIdx++) {
    debugString += "\nBin constraint " + std::to_string(constraintIdx) +
                   " contains the following ops: ";
    for (auto op : binConstraints[constraintIdx]) {
      debugString += "\n\t" + op->debugName();
    }
  }
  logging::debug("[AccumulateOuterFragmentParallelizer::getBinConstraints] "
                 "There are {} bin constraints in AOF {}",
                 binConstraints.size(),
                 debugString);

  return binConstraints;
}

AccumulateOuterFragmentParallelizer::OpClusters
AccumulateOuterFragmentParallelizer::getGradientClippingClusters(
    const Graph &graph) const {
  // Gradient clipping ops create a link between all the ops in a gradient
  // clipping group, which was resulting in this function creating only a single
  // cluster per gradient clipping group. Here we gather all the gradient
  // clipping ops, and create a OpCluster for each gradient clipping group. Then
  // when we create the rest of the clusters we can exclude the gradient
  // clipping ops.
  using boost::algorithm::starts_with;

  // Find all the global norm tensors and get their producers.
  std::vector<Op *> globalNorms;
  for (auto tid : graph.getTensors().getIds(TensorType::ActGrad)) {
    if (starts_with(tid, reservedGlobalNormPrefix())) {
      auto t = graph.getTensors().get(tid);
      globalNorms.push_back(t->getProducer());
    }
  }

  OpClusters gradClipClusters;
  for (auto globalNormProducer : globalNorms) {
    // Use the global norm producer to create the initial op cluster.
    OpCluster gradCluster({&graph, globalNormProducer});

    // Add all ops that preceed the globalNormProducer and are in the
    // AccumulateOuterFragment. This will catch all the gradient clipping ops
    // that preceed the globalNormProducer, and also some other ops before
    // gradient clipping that it makes sense to absord into the gradient
    // clipping op clusters.
    OpSearchHelper opSearch;
    opSearch.pushInputProducers(globalNormProducer);
    while (!opSearch.empty()) {
      Op *x = opSearch.pop();
      if (x->settings.executionContext ==
          ExecutionContext::AccumulateOuterFragment) {
        gradCluster.absorb({&graph, x});
        opSearch.pushInputProducers(x);

        // All ops which are tied to another op in the gradient clipping
        // cluster must themselves be in the gradient clipping cluster
        // For instance when merging collectives
        for (auto before : graph.topoCons->getTiedBefores(x)) {
          opSearch.push(before);
        }
        for (auto after : graph.topoCons->getTiedAfters(x)) {
          opSearch.push(after);
        }
      }
    }

    // Add all ops following the globalNormProducer that are flagged as gradient
    // clipping ops.
    opSearch.pushOutputConsumers(globalNormProducer);
    while (!opSearch.empty()) {
      Op *x = opSearch.pop();
      if (x->isGradientClippingOp()) {
        gradCluster.absorb({&graph, x});
        opSearch.pushOutputConsumers(x);
      }
    }

    gradClipClusters.push_back(gradCluster);
  }

  return gradClipClusters;
}

void AccumulateOuterFragmentParallelizer::populateOpClusters(
    const Graph &graph,
    OpClusters &clusters) const {
  // Start from scratch.
  OpClusters clustersToGroup;

  auto gradClipClusters = getGradientClippingClusters(graph);
  // Gather all of the gradient clipping cluster ops into a set to make it
  // easier to exclude them.
  std::set<Op *> gradClipOps;
  for (auto &x : gradClipClusters) {
    gradClipOps.insert(x.ops.begin(), x.ops.end());
  }

  // Gather all relevant ops in single-op clusters first.
  for (const auto &x : graph.getOps()) {
    auto op = x.second.get();
    if (gradClipOps.count(op) == 0) {
      if (op->settings.executionContext ==
          ExecutionContext::AccumulateOuterFragment) {
        clustersToGroup.push_back(OpCluster(&graph, op));
      }
    }
  }

  // Start with nothing.
  clusters.clear();

  // Add the gradient clipping clusters at the start of the clusters as gradient
  // clipping ops need to be scheduled first.
  clusters.insert(
      clusters.begin(), gradClipClusters.begin(), gradClipClusters.end());

  while (!clustersToGroup.empty()) {
    // Pick a cluster and find all overlapping ones.
    clusters.push_back(clustersToGroup.back());
    clustersToGroup.pop_back();

    // Need to do this in a fixed point because overlap
    // is transitive.
    while (true) {
      bool absorbedCluster = false;
      for (auto it = clustersToGroup.begin(); it != clustersToGroup.end();) {
        if (clusters.back().overlaps(*it)) {
          clusters.back().absorb(*it);
          absorbedCluster = true;

          // Remove absorbed cluster from clustersToGroup while iterating over
          // it. This requires special treatment of the iterator.
          it = clustersToGroup.erase(it);
        } else {
          it++;
        }
      }

      if (!absorbedCluster) {
        break;
      }
    }
  }
}

void AccumulateOuterFragmentParallelizer::populateOpClusters(
    const Graph &graph,
    OpClustersMap &clustersMap) const {

  // Start from scratch.
  clustersMap.clear();

  // Get the clusters.
  OpClusters clusters;
  populateOpClusters(graph, clusters);

  // Turn it into a map.
  for (auto cluster : clusters) {
    if (cluster.hasVirtualGraphId()) {
      const auto vid = cluster.getVirtualGraphId();
      clustersMap[vid].push_back(cluster);
    } else {
      logging::warn("Unexpectedly encountered ops in the outer fragment "
                    "without a virtual graph id");
    }
  }
}

void AccumulateOuterFragmentParallelizer::filterOpClusters(
    OpClusters &clusters) const {
  // Filter out clusters that have no remote ops.
  auto end = std::remove_if(
      clusters.begin(), clusters.end(), [](OpCluster &opCluster) {
        return !opCluster.hasRemoteOp();
      });
  clusters.erase(end, clusters.end());
}

void AccumulateOuterFragmentParallelizer::filterOpClusters(
    OpClustersMap &clustersMap) const {
  // Filter out groups of clusters that have no remote ops.
  for (auto &clustersMapEntry : clustersMap) {
    auto &clusters = clustersMapEntry.second;
    filterOpClusters(clusters);
  }
}

void AccumulateOuterFragmentParallelizer::sortOpClusters(
    // Sort op groups in descending order of the number
    // of bytes that are loaded. This is so we can better
    // parallelise similar groups across virtual graphs.
    OpClusters &clusters) const {
  clusters.sort([](const OpCluster &c1, const OpCluster &c2) {
    // Descending order of size.
    return c1.numLoadBytes < c2.numLoadBytes;
  });
}

void AccumulateOuterFragmentParallelizer::sortOpClusters(
    // Sort op groups in descending order of the number
    // of bytes that are loaded. This is so we can better
    // parallelise similar groups across virtual graphs.
    OpClustersMap &clustersMap) const {
  for (auto &clustersMapEntry : clustersMap) {
    auto &clusters = clustersMapEntry.second;
    sortOpClusters(clusters);
  }
}

void AccumulateOuterFragmentParallelizer::tryToParallelizeOpClusters(
    Graph &graph,
    OpClusters &opClusters) const {
  OpCluster::Ops remoteLoadOps;
  OpCluster::Ops remoteStoreOps;
  for (auto cluster : opClusters) {
    remoteLoadOps.insert(remoteLoadOps.end(),
                         cluster.remoteLoadOps.begin(),
                         cluster.remoteLoadOps.end());
    remoteStoreOps.insert(remoteStoreOps.end(),
                          cluster.remoteStoreOps.begin(),
                          cluster.remoteStoreOps.end());
  }

  // Insert topocons with 'tied' flag to try and push
  // the loads and stores together.
  addOpConstraints(graph, remoteLoadOps);
  addOpConstraints(graph, remoteStoreOps);
}

void AccumulateOuterFragmentParallelizer::addOpConstraints(
    Graph &graph,
    const Ops &ops) const {
  // We use the 'tied' flag to try and make sure ops are scheduled next
  // to one another -- as a side effect we impose a serialization.
  Op *lastOp = nullptr;
  for (auto op : ops) {
    if (lastOp != nullptr) {
      graph.topoCons->insert(lastOp, op, /*tied*/ true);
    }
    lastOp = op;
  }
}

namespace {
bool init =
    Transform::registerTransform(new AccumulateOuterFragmentParallelizer);
}

} // namespace popart
