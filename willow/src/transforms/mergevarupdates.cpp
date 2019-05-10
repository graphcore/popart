#include <poponnx/graph.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/concat.hpp>
#include <poponnx/op/flatten.hpp>
#include <poponnx/op/varupdate.hpp>
#include <poponnx/opidentifier.hpp>
#include <poponnx/opmanager.hpp>
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

MergeVarUpdates::PartitionId MergeVarUpdates::getPartitionId(Op *op) const {
  std::stringstream ss;
  ss << op->settings.vgraphId << '_';
  if (op->isConvertibleTo<ConstSGDVarUpdateOp>()) {
    auto csvu = dynamic_cast<ConstSGDVarUpdateOp *>(op);
    ss << csvu->getLearnRate() << '_' << csvu->getWeightDecay() << '_';
  } else if (op->isConvertibleTo<SGDVarUpdateOp>()) {
    auto svu = dynamic_cast<SGDVarUpdateOp *>(op);
    ss << svu->inId(svu->getLearnRateInIndex()) << '_'
       << svu->inId(svu->getWeightDecayInIndex()) << '_';
  } else if (op->isConvertibleTo<CopyVarUpdateOp>()) {
    // there are no attributes to sub-partition CopyVarUpdatOps by
  } else {
    throw error(
        "Unrecognised {}, is not a VarUpdateOp supported in Merge Pattern",
        op->str());
  }
  return ss.str();
}

std::map<MergeVarUpdates::PartitionId, std::vector<VarUpdateOp *>>
MergeVarUpdates::getLargestGroupTargetsMap(const Graph &graph) const {
  std::map<PartitionId, std::vector<VarUpdateOp *>> targetsMap;
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

bool MergeAllVarUpdates::apply(Graph &graph) const {

  // does this call to "apply" change the Graph input?
  // Will become true if any partition is not a singleton.
  bool changed = false;

  // flatten to shape (1, ni) for all tensors,
  const int flattenAxis = 0;
  // then concat to shape (1, n1 + n2 + ... + nT).
  const int concatAxis = 1;

  auto targetsMap = getLargestGroupTargetsMap(graph);
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
bool init = Transform::registerTransform(new MergeAllVarUpdates);
}

} // namespace poponnx
