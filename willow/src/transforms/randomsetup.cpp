// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/call.hpp>
#include <popart/op/dropout.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/modifyrandomseed.hpp>
#include <popart/op/randombase.hpp>
#include <popart/op/remote.hpp>
#include <popart/opidentifier.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/randomsetup.hpp>

#include <chrono>

#include <iostream>

// There are a few things we want to achieve in this transform:
//
// * Distinct random ops have distinct random seeds.
// * Backwards version of ops have identical seeds to the forward op.
// * Recomputation clones of ops have identical seeds to the op they
//       are recomputing.
// * Random seeds are distinct in new steps/batches.
//

namespace popart {

std::size_t RandomSetup::id() { return typeid(RandomSetup).hash_code(); }

bool RandomSetup::apply(Graph &graph) const {

  // In this transformation, which is ran after the backward pass, we make the
  // necessary changes to facilitate the distribution of random seeds to
  // random operations.
  //
  // For graphs that require random seeds (which is the case when there are any
  // random ops or when stochastic rounding is used) do the following:
  //
  // 1. Introduce a [randomSeed___fromHost] tensor.
  // 2. Introduce a [randomSeed___updated] tensor.
  // 3. Introduce a GetRandomSeedOp which increments both components of the
  //       [randomSeed___fromHost] and [randomSeed___updated] tensors by 1.
  // 4. For every group of random ops that needs the same seed (as determined by
  //       equivalence of the value returned by op->getRandomSeedPlaceholder(),
  //       the op's virtual graph id and pipeline stage) insert a
  //       ModifyRandomSeedOp. This op increments the right-hand side
  //       component of [randomSeed___updated] to make a new constant tensor,
  //       [SeedModifier___i], the value of which is different for each group
  //       and less than N. This part of the transformation looks like this
  //       (suppose random ops 0 and 1 want the same seed, and random op 2 wants
  //       a separate seed):
  //
  //   Before:
  //                                 _______________
  //                                | random op 0   |  --> ...
  //                                |_______________|
  //                                 _______________
  //                                | random op 1   |  --> ...
  //                                |_______________|
  //                                 _______________
  //                                | random op 2   |  --> ...
  //                                |_______________|
  //
  //   After:
  //                                 ____________________
  //   [randomSeed___fromHost] -->  | GetRandomSeedOp    |  ---
  //                                |____________________|     |
  //    _______________________________________________________|
  //   |                             ____________________
  //   +-->[randomSeed___updated]-->| ModifyRandomSeedOp |  ---
  //   |   [SeedModifier___0]  -->  |____________________|     |
  //   |                       ________________________________|
  //   |                      |      _______________
  //   |                      +-->  | random op 0   |  --> ...
  //   |                      |     |_______________|
  //   |                      |      _______________
  //   |                      +-->  | random op 1   |  --> ...
  //   |                            |_______________|
  //   |                             ____________________
  //   +-->[randomSeed___updated]-->| ModifyRandomSeedOp |  ---
  //       [SeedModifier___1]  -->  |____________________|     |
  //                           ________________________________|
  //                          |      _______________
  //                          +-->  | random op 2   |  --> ...
  //                                |_______________|
  //
  //  NOTE: There is a copy of ModifyRandomSeedOp for every distinct combination
  //  of virtual graph id and pipeline stage.
  //
  //  NOTE: We don't use Poplar's built-in seedModifier parameter for this
  //  because this results in random operations that are not outlinable.
  //
  using RandomOpSet = std::set<RandomBaseOp *>;
  using GroupKey    = std::tuple<VGraphId, PipelineStage>;

  auto &ir = graph.getIr();

  if (RandomSetup::requiresRandomSeed(ir)) {
    logging::debug("[RandomSetup] Started.");

    RandomOpSet randomOps;
    std::map<RandomSeedPlaceholder, std::map<GroupKey, RandomOpSet>>
        randomOpGroups;

    auto allOps                       = ir.getAllOps();
    bool allOtherOpsHavePipelineStage = true;
    for (auto op : allOps) {
      if (!op->hasPipelineStage()) {
        allOtherOpsHavePipelineStage = false;
      }
    }

    // Parition random ops into those that need distinct seeds.
    for (auto op : allOps) {
      if (op->isConvertibleTo<RandomBaseOp>()) {
        auto randomOp = dynamic_cast<RandomBaseOp *>(op);
        randomOps.insert(randomOp);
        randomOpGroups[randomOp->getRandomSeedPlaceholder()]
                      [getGroupKey(randomOp)]
                          .insert(randomOp);
      }
    }

    int64_t numModifyOps = 0ull;
    for (auto &entry : randomOpGroups) {
      numModifyOps += entry.second.size();
    }
    logging::debug("[RandomSetup] Need {} distinct random seed(s) produced by "
                   "{} ModifyRandomSeed op(s) for {} random op(s).",
                   randomOpGroups.size(),
                   numModifyOps,
                   randomOps.size());

    // 1. Create [randomSeed___fromHost] tensor.
    TensorId seedId = GetRandomSeedOp::getStreamedSeedTensorId();
    DataType dtype  = DataType::UINT32;
    TensorInfo info(dtype, {2});
    ir.getMainGraph().getTensors().addStream(seedId, {dtype, {2}});
    Tensor &seedTensor = *ir.getMainGraph().getTensors().get(seedId);
    seedTensor.setReplicatedStreamMode(Tensor::ReplicatedStreamMode::Replicate);

    // 3. Add GetRandomSeed op.
    Op::Settings settings(ir.getMainGraph(), "");
    auto getSeedOp = std::make_unique<GetRandomSeedOp>(
        Onnx::CustomOperators::GetRandomSeed, settings);

    if (ir.virtualGraphsEnabled()) {
      getSeedOp->setVirtualGraphId(0);
      if (ir.getSessionOptions().enablePipelining &&
          allOtherOpsHavePipelineStage) {
        getSeedOp->setPipelineStage(0);
      }
    }

    // 2. Create [randomSeed___updated] tensor.
    getSeedOp->connectInTensor(getSeedOp->getSeedInIndex(), seedId);
    TensorId updatedSeedId = GetRandomSeedOp::getUpdatedSeedTensorId();
    getSeedOp->createAndConnectOutTensor(
        GetRandomSeedOp::getUpdatedSeedOutIndex(), updatedSeedId);
    getSeedOp->setup();

    logging::debug("[RandomSetup] Added op {}.", getSeedOp->str());
    ir.getMainGraph().moveIntoGraph(std::move(getSeedOp));

    // 4. Add ModifyRandomSeedOp for every group of random ops that wants
    // the same seed.
    uint32_t modifier = 0u;
    for (auto &entry : randomOpGroups) {

      auto seedModifierId =
          ModifyRandomSeedOp::getSeedModifierTensorId(modifier);

      // Insert a constant tensor modifier for this op.
      std::vector<uint32_t> modifierData(1, {modifier});
      TensorInfo modifierInfo(DataType::UINT32, {});
      ir.getMainGraph().getTensors().addConstInit(
          seedModifierId,
          modifierInfo,
          reinterpret_cast<void *>(modifierData.data()));

      for (auto &groupEntry : entry.second) {
        auto &virtualGraphId = std::get<0>(groupEntry.first);
        auto &pipelineStage  = std::get<1>(groupEntry.first);
        auto &randomOps      = groupEntry.second;

        auto distinctSeedId =
            ModifyRandomSeedOp::getModifiedSeedTensorId(modifier);
        if (virtualGraphId >= 0) {
          distinctSeedId += "_" + std::to_string(virtualGraphId);
        }
        if (pipelineStage >= 0) {
          distinctSeedId += "_" + std::to_string(pipelineStage);
        }

        auto modifyOp = std::make_unique<ModifyRandomSeedOp>(
            Onnx::CustomOperators::ModifyRandomSeed, settings);

        if (ir.virtualGraphsEnabled()) {
          if (virtualGraphId >= 0) {
            modifyOp->setVirtualGraphId(virtualGraphId);
          }
          if (pipelineStage >= 0 && ir.getSessionOptions().enablePipelining &&
              allOtherOpsHavePipelineStage) {
            modifyOp->setPipelineStage(pipelineStage);
          }
        }

        modifyOp->connectInTensor(modifyOp->getSeedInIndex(), updatedSeedId);
        modifyOp->connectInTensor(modifyOp->getSeedModifierInIndex(),
                                  seedModifierId);
        modifyOp->createAndConnectOutTensor(
            ModifyRandomSeedOp::getModifiedSeedOutIndex(), distinctSeedId);
        modifyOp->setup();

        logging::debug(
            "[RandomSetup] Added op {} with modifier "
            "constant {} ({}) to provide modified seed for {} random op(s). "
            "Random ops are:",
            modifyOp->str(),
            seedModifierId,
            modifier,
            randomOps.size());

        ir.getMainGraph().moveIntoGraph(std::move(modifyOp));

        for (auto &randomOp : randomOps) {
          logging::debug("[RandomSetup]   - {}", randomOp->str());
          randomOp->connectInTensor(randomOp->getSeedInIndex(), distinctSeedId);
        }
      }

      ++modifier;
    }

    logging::debug("[RandomSetup] Done.");
    return true;

  } else {
    logging::debug("[RandomSetup] Nothing to do.");
    return false;
  }
}

bool RandomSetup::hasRandomOps(const Ir &ir) {
  auto ops = ir.getAllOps();
  return std::any_of(ops.begin(), ops.end(), [](const Op *op) {
    return op->isConvertibleTo<RandomBaseOp>();
  });
}

bool RandomSetup::requiresRandomSeed(const Ir &ir) {
  return (RandomSetup::hasRandomOps(ir) ||
          ir.getSessionOptions().enableStochasticRounding);
}

std::tuple<VGraphId, PipelineStage> RandomSetup::getGroupKey(const Op *op) {
  std::tuple<VGraphId, PipelineStage> key(-1, -1);
  if (op->hasVirtualGraphId()) {
    std::get<0>(key) = op->getVirtualGraphId();
  }
  if (op->hasPipelineStage()) {
    std::get<1>(key) = op->getPipelineStage();
  }
  return key;
}

bool RandomSetup::hasRandomSeed(const Ir &ir) {
  return ir.containsTensor(GetRandomSeedOp::getStreamedSeedTensorId());
}

TensorId RandomSetup::getStreamedSeedTensorId() {
  return GetRandomSeedOp::getStreamedSeedTensorId();
}

namespace {
// RandomSetup.
bool init = Transform::registerTransform(new RandomSetup());
} // namespace

} // namespace popart
