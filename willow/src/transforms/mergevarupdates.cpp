#include <tuple>
#include <poponnx/graph.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/concat.hpp>
#include <poponnx/op/flatten.hpp>
#include <poponnx/op/varupdate.hpp>
#include <poponnx/opidentifier.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/optionflags.hpp>
#include <poponnx/tensorindex.hpp>
#include <poponnx/tensors.hpp>
#include <poponnx/transforms/mergevarupdates.hpp>

namespace poponnx {

namespace {
std::string getConcatWeightsPrefix() { return "concatWeights___"; }

std::string getConcatGradsPrefix() { return "concatGrads___"; }

std::string getFlattenedPrefix() { return "flattened___"; }
} // namespace

std::size_t MergeAllVarUpdates::id() {
  return typeid(MergeAllVarUpdates).hash_code();
}

std::size_t MergeAutoVarUpdates::id() {
  return typeid(MergeAutoVarUpdates).hash_code();
}

MergeVarUpdates::PartitionId MergeVarUpdates::getPartitionId(Op *op) const {
  std::stringstream ss;
  ss << "vg_" << op->settings.vgraphId << '_';
  if (op->isConvertibleTo<ConstSGDVarUpdateOp>()) {
    auto csvu = dynamic_cast<ConstSGDVarUpdateOp *>(op);
    ss << "lr_" << csvu->getLearnRate() << '_' << "wd_"
       << csvu->getWeightDecay() << '_';
  } else if (op->isConvertibleTo<SGDVarUpdateOp>()) {
    auto svu = dynamic_cast<SGDVarUpdateOp *>(op);
    ss << "lri_" << svu->inId(svu->getLearnRateInIndex()) << '_' << "wdi_"
       << svu->inId(svu->getWeightDecayInIndex()) << '_';
  } else if (op->isConvertibleTo<CopyVarUpdateOp>()) {
    // there are no attributes to sub-partition CopyVarUpdatOps by
    ss << "copyVar";
  } else {
    throw error("{} is not a VarUpdateOp supported in Merge Pattern",
                op->str());
  }
  return ss.str();
}

MergeVarUpdates::VarUpdatePartition
MergeVarUpdates::getLargestGroupTargetsMap(const Graph &graph) const {
  VarUpdatePartition targetsMap;
  for (auto &id_upop : graph.getOps()) {
    auto op = id_upop.second.get();
    if (op->isConvertibleTo<VarUpdateOp>()) {
      auto vuop        = dynamic_cast<VarUpdateOp *>(op);
      auto partitionId = getPartitionId(vuop);
      if (targetsMap.find(partitionId) == targetsMap.end()) {
        targetsMap.insert({partitionId, {}});
      }
      targetsMap[partitionId].push_back(vuop);
    }
  }
  return targetsMap;
}

MergeVarUpdates::VarUpdatePartition
MergeAutoVarUpdates::getGroupTargetsMap(const Graph &g) const {

  int64_t thresholdMemory =
      g.getIr().getSessionOptions().mergeVarUpdateMemThreshold;

  if (thresholdMemory < 0) {
    throw error("Negative memory {} threshold detected in MergeAutoVarUpdates",
                thresholdMemory);
  }

  // the largest possible partitions
  // (same var-update types, i.e. var-updates that CAN be merged)
  auto largestTargetsMap = getLargestGroupTargetsMap(g);

  // check that there is any chance of a partition with more than 1 Op,
  // if not, sub-partitioning is not possible, so return early
  bool isNonTrivialPartition = false;
  for (auto x : largestTargetsMap) {
    if (x.second.size() > 1) {
      isNonTrivialPartition = true;
      break;
    }
  }
  if (!isNonTrivialPartition) {
    return largestTargetsMap;
  }

  auto opSched = g.getOpSchedule({});

  // a map from Ops to their position in the schedule
  std::map<Op *, int> schedIndex;
  for (int i = 0; i < opSched.size(); ++i) {
    schedIndex[opSched[i]] = i;
  }

  // find the point at which the forward part of the compute graph ends
  int switchIndex = -1;
  for (int i = 0; i < opSched.size(); ++i) {
    if (opSched[i]->getPhase() == Phase::FWD ||
        opSched[i]->getPhase() == Phase::LOSS) {
      switchIndex = i;
    }
  }
  if (switchIndex < 0) {
    throw error(
        "ILE: failed to set switchIndex, is the graph in training mode?");
  }

  // for every tensor which is
  // 1) created on the forward path and
  // 2) consumed on the backward path,
  // insert "+mem" at creation point and "-mem" at final consumption.
  // This vector will look something like,
  // ..+...+.+..+...+..S...-.-...-...-.-,
  //
  // where S above is switchIndex.
  //
  std::vector<int64_t> deltaMemFwdLiveForBwd(opSched.size(), 0);
  for (int i = 0; i < switchIndex; ++i) {
    for (Tensor *t : opSched[i]->output->tensors()) {
      // final consumption time
      int fct = -1;
      for (Op *consumer : t->consumers.getOps()) {
        fct = std::max<int>(fct, schedIndex.at(consumer));
      }
      if (fct > switchIndex) {
        deltaMemFwdLiveForBwd[i] += t->info.nbytes();
        deltaMemFwdLiveForBwd[fct] -= t->info.nbytes();
      }
    }
  }

  // cumulative sum of deltaMemFwdLiveForBwd
  std::vector<int64_t> cumMemFwdLiveForBwd(opSched.size(), 0);
  int64_t maxCumMemFwdLiveForBwd = 0;
  cumMemFwdLiveForBwd[0]         = deltaMemFwdLiveForBwd[0];
  for (int i = 1; i < opSched.size(); ++i) {
    cumMemFwdLiveForBwd[i] =
        deltaMemFwdLiveForBwd[i] + cumMemFwdLiveForBwd[i - 1];
    maxCumMemFwdLiveForBwd =
        std::max<int64_t>(maxCumMemFwdLiveForBwd, cumMemFwdLiveForBwd[i]);
  }

  if (cumMemFwdLiveForBwd[opSched.size() - 1] != 0) {
    throw error("ILE: expected final cumulative memory to be zero");
  }

  // An estimate of how much memory there is to use for delaying weight
  // updates without effecting max-liveness, looks something like
  //
  // *                         *
  // *                         *
  // **                       **
  // ****                 ******
  // *******        ************
  // **********   **************
  // -----------------------------> schedule index
  // where above: vertical is memory to play with
  // and horizontal is schedule position

  std::vector<int64_t> memToPlayWith(opSched.size(), 0);
  for (int i = 0; i < opSched.size(); ++i) {
    memToPlayWith[i] = maxCumMemFwdLiveForBwd - cumMemFwdLiveForBwd[i];
  }

  std::map<VarUpdateOp *, PartitionId> parentPartitionId;
  std::vector<std::tuple<int, VarUpdateOp *>> varUpdatesBySchedIndex;

  // variables to monitor memory as we perform sub-partitioning on
  // largestTargetsMap.
  // 1) the VarUpdates which we've delayed scheduling of
  VarUpdatePartition pendingVarUpdates;
  // 2) the total memory of the delayed updates for each parent partition id
  //    Example : CopyVarUpdate-xx:20, SGDVarUpdate-xx:50
  std::map<PartitionId, int64_t> pendingMemories;
  // 3) the gross total memory of the delayer partitions
  //    Eaxmple : 70
  int64_t totalPendingMemory = 0;

  // initialise the above variables
  for (auto x : largestTargetsMap) {
    auto id         = x.first;
    auto varUpdates = x.second;
    pendingMemories.insert({id, 0});
    pendingVarUpdates.insert({id, {}});
    for (Op *op : varUpdates) {
      auto varUpdateOp = dynamic_cast<VarUpdateOp *>(op);
      parentPartitionId.insert({varUpdateOp, id});
      std::tuple<int, VarUpdateOp *> tup(schedIndex.at(op), varUpdateOp);
      varUpdatesBySchedIndex.push_back(tup);
    }
  }

  // sort from earliest schedule position to last schedule position
  std::sort(varUpdatesBySchedIndex.begin(), varUpdatesBySchedIndex.end());

  // for every VarUpdateOp, what is the minimum memory to play with
  // from its schedule position to the next VarUpdateOps schedule position?
  std::vector<int64_t> minToPlayWithTilNextVarUpdate;
  for (int i = 0; i < varUpdatesBySchedIndex.size() - 1; ++i) {
    minToPlayWithTilNextVarUpdate.push_back(
        std::numeric_limits<int64_t>::max());
    for (int j = std::get<0>(varUpdatesBySchedIndex[i]);
         j < std::get<0>(varUpdatesBySchedIndex[i + 1]);
         ++j) {
      minToPlayWithTilNextVarUpdate[i] =
          std::min<int64_t>(minToPlayWithTilNextVarUpdate[i], memToPlayWith[j]);
    }
  }
  // we say that the final VarUpdateOp has no memory to play with, which
  // guratantees that the VarUpdateOps are all flushed at this point
  minToPlayWithTilNextVarUpdate.push_back(0);

  // Now prepare the sub-partitioning,
  VarUpdatePartition subPartitions;

  auto insertSubPartition =
      [&subPartitions](const std::string &parPartId,
                       const std::vector<VarUpdateOp *> &vuops) {
        auto newSubPartitionName =
            parPartId + "__spn__" + std::to_string(subPartitions.size());
        subPartitions.insert({newSubPartitionName, vuops});
      };

  // iterating through all the VarUpdateOps in order they appear in the schedule
  for (int varUpdateNumber = 0; varUpdateNumber < varUpdatesBySchedIndex.size();
       ++varUpdateNumber) {

    auto vu_tuple            = varUpdatesBySchedIndex[varUpdateNumber];
    VarUpdateOp *varUpdateOp = std::get<1>(vu_tuple);
    auto parPartId           = parentPartitionId.at(varUpdateOp);
    Tensor *toUpdate =
        varUpdateOp->input->tensor(varUpdateOp->getVarToUpdateInIndex());
    int64_t varMemSize = toUpdate->info.nbytes();

    // add the new VarUpdateOp to the list of pending VarUpdateOps and update
    // the memory monitoring variables
    totalPendingMemory += varMemSize;
    pendingMemories[parPartId] += varMemSize;
    pendingVarUpdates[parPartId].push_back(varUpdateOp);

    // check for a merger: is the pending memory too large (as compared to
    // threshold and as compared to memory to play with) ?
    while (totalPendingMemory >
               minToPlayWithTilNextVarUpdate[varUpdateNumber] ||
           totalPendingMemory > thresholdMemory) {
      // need to merge some VarUpdateOps as memory limit exceeded.
      // Which type of VarUpdateOps to merge?
      // We choose the one with the largest pending memory, found below,
      PartitionId largestLivePartitionId;
      int64_t largestLivePartitionSize = -1;
      for (auto id_size : pendingMemories) {
        if (id_size.second >= largestLivePartitionSize) {
          largestLivePartitionId   = id_size.first;
          largestLivePartitionSize = id_size.second;
        }
      }

      auto newSubPartition = pendingVarUpdates[largestLivePartitionId];
      pendingVarUpdates[largestLivePartitionId].clear();
      totalPendingMemory -= largestLivePartitionSize;
      pendingMemories[largestLivePartitionId] = 0;

      insertSubPartition(largestLivePartitionId, newSubPartition);
    }
  }

  // add any remaining var-updates
  for (auto x : pendingVarUpdates) {
    insertSubPartition(x.first, x.second);
  }

  return subPartitions;
}

bool MergeVarUpdates::apply(Graph &graph) const {

  // does this call to "apply" change the Graph input?
  // Will become true if any partition is not a singleton.
  bool changed = false;

  // flatten to shape (1, ni) for all tensors,
  const int flattenAxis = 0;
  // then concat to shape (1, n1 + n2 + ... + nT).
  const int concatAxis = 1;

  auto targetsMap = getGroupTargetsMap(graph);
  for (auto &targetMap : targetsMap) {
    auto target = targetMap.second;
    if (target.size() > 1) {
      auto partitionId = targetMap.first;
      // The variable update type
      changed = true;

      //  replace individual weight updates;
      //  ---------------------------------
      //
      //   W0  dW0     W1  dW1
      //   |    |       |   |
      //  VarUpdate   VarUpdate
      //     |            |
      //   W0new        W1new
      //
      //
      //   with a merged weight update:
      //   ----------------------------
      //
      //   W0           W1      dW0         dW1
      //   |            |        |           |
      // FlattenInplace |  FlattenInplace    |
      //   |     FlattenInplace |     FlattenInplace
      //   |            |       |            |
      //   \           /        \           /
      //   ConcatInplace        ConcatInplace
      //             \             /
      //               \          /
      //                 \       /
      //                  VarUpdate
      //                     |
      //                ConcatedWsNew
      //
      //  Similarly for non-const SGDVarUpdates and CopyUpdate

      // The Ops which flatten the Variable Tensors
      std::vector<Op *> flattenWeightOps;
      // The Ops which flatten the Updater Tensors (grads, sources of copies)
      std::vector<Op *> flattenUpdaterOps;

      // Build up the name of the ConcatInplaceOp for the weight concat.
      std::stringstream concatWeightsNameStream;
      concatWeightsNameStream << getConcatWeightsPrefix();

      std::stringstream concatUpdatersNameStream;
      concatUpdatersNameStream << getConcatGradsPrefix();

      Op::Settings canonSettings = target[0]->settings;
      // optimizer specific input tensor names
      auto optimizerInputs =
          dynamic_cast<VarUpdateOp *>(target[0])->optimizerInputs();
      auto upCanonTargetOp = target[0]->clone();
      auto canonTargetOp   = dynamic_cast<VarUpdateOp *>(upCanonTargetOp.get());

      for (auto singleUpdateOp : target) {

        auto makeFlattened =
            [canonSettings, flattenAxis, &graph](const TensorId &id) {
              // create FlattenInplaceOp
              auto tempOp = make_unique<FlattenInplaceOp>(
                  Onnx::CustomOperators::FlattenInplace,
                  flattenAxis,
                  canonSettings);
              auto op = tempOp.get();
              graph.moveIntoGraph(std::move(tempOp));
              op->connectInTensor(FlattenBaseOp::getInIndex(), id);
              op->createAndConnectOutTensor(FlattenBaseOp::getOutIndex(),
                                            getFlattenedPrefix() + id);
              op->setup();
              return op;
            };

        // create FlattenInplaceOp for the weight being updated
        auto weightIn =
            singleUpdateOp->inTensor(VarUpdateOp::getVarToUpdateInIndex());
        auto flWeOp = makeFlattened(weightIn->id);
        flattenWeightOps.push_back(flWeOp);
        concatWeightsNameStream << '_' << weightIn->id;

        // create FlattenInplaceOp for the gradient, or source of copy
        auto gIn = singleUpdateOp->inTensor(VarUpdateOp::getUpdaterInIndex());
        auto flGrOp = makeFlattened(gIn->id);
        flattenUpdaterOps.push_back(flGrOp);
        concatUpdatersNameStream << '_' << gIn->id;

        auto weightOutId =
            singleUpdateOp->outTensor(VarUpdateOp::getUpdatedVarOutIndex())->id;

        // disconnect and delete the single var updater and its output
        singleUpdateOp->disconnectAllInputs();
        singleUpdateOp->disconnectAllOutputs();
        graph.eraseOp(singleUpdateOp->id);
        graph.getTensors().remove(weightOutId);
      }

      auto getConcatInplace = [&graph, canonSettings](
                                  const std::vector<Op *> &flattened,
                                  TensorId newId) {
        // create ConcatInplaceOp for the flattened input tensors
        auto tempOp =
            std::unique_ptr<Op>(new ConcatInplaceOp(concatAxis, canonSettings));
        auto concatOp = tempOp.get();
        graph.moveIntoGraph(std::move(tempOp));
        for (int i = 0; i < flattened.size(); ++i) {
          concatOp->connectInTensor(
              i, flattened[i]->outTensor(FlattenBaseOp::getOutIndex())->id);
        }

        concatOp->createAndConnectOutTensor(ConcatOp::getOutIndex(), newId);
        concatOp->setup();
        return concatOp;
      };

      // create ConcatInplaceOp for the flattened weights
      auto concatWeightsOp =
          getConcatInplace(flattenWeightOps, concatWeightsNameStream.str());
      auto concatedWeightsTensorId =
          concatWeightsOp->outTensor(ConcatOp::getOutIndex())->id;

      // create ConcatInplaceOp for the flattened grads (or sources op copies)
      auto concatGradsOp =
          getConcatInplace(flattenUpdaterOps, concatUpdatersNameStream.str());
      auto concatedGradsTensorId =
          concatGradsOp->outTensor(ConcatOp::getOutIndex())->id;

      // create the new, merged variable update
      auto tempOp = canonTargetOp->cloneWithNewName(concatedWeightsTensorId);
      for (auto &x : optimizerInputs) {
        tempOp->connectInTensor(x.first, x.second);
      }

      Op *multiUpdateOp = tempOp.get();
      graph.moveIntoGraph(std::move(tempOp));
      multiUpdateOp->connectInTensor(VarUpdateOp::getVarToUpdateInIndex(),
                                     concatedWeightsTensorId);
      multiUpdateOp->connectInTensor(VarUpdateOp::getUpdaterInIndex(),
                                     concatedGradsTensorId);

      multiUpdateOp->createAndConnectOutTensor(
          VarUpdateOp::getUpdatedVarOutIndex(),
          "updated___" + concatedWeightsTensorId);

      multiUpdateOp->setup();
    }
  }

  return changed;
}

namespace {
bool initAll  = Transform::registerTransform(new MergeAllVarUpdates);
bool initAuto = Transform::registerTransform(new MergeAutoVarUpdates);
} // namespace

} // namespace poponnx
