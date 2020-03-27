// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <vector>

#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>

#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/loss.hpp>
#include <popart/op/restore.hpp>
#include <popart/op/stash.hpp>
#include <popart/patterns/contiguateipucopyindices.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/pipeline.hpp>
#include <popart/vertex.hpp>

using boost::range::max_element;
using boost::range::min_element;

// Which pipelining scheme should we use? There are some considerations to
// make:
//  - Which order should the 5 progs (Fwd, Bwd, Stash, Restore, Sync)
//    run in the pipeline cycle?
//  - Should the activation be restored in-place?
//
// These decisions affect the stash sizes and the total number of activations
// that have to be stored in the pipelined model.
//
// We have decided on:
//  - In-place the activation tensors when restoring
//  - Running : Fwd/Stash/Restore/Bwd/Sync
// resulting in a stash size of 2*(IPUs to end) + 1, except for the last
// IPU, which has no stash.
//
// You can get away with a smaller stash, at the expense of having to out-place
// activations and changing the program order. Our approach is conceptually
// straight-forward and has no memory penalty.
//
// The transform
// -------------
//
// Before:
//
// FwdOp     t_act_grad
//   |          |
//  t_act --- BwdOp
//   |          |
//  ...      t_grad_in
//
// After:
//
// FwdOp
//   |
// t_act ----------           t_act_grad
//   | \           |             |
//   |   \      StashOp          |
//   |     \       |             |
//   |       \   t_act_stashed   |
//   |        |    |             |
//   |        |    |             |
//   |     RestoreOp             |
//   |       |                   |
//   |     t_act_alias ------- BwdOp
//   |                           |
//   |                       t_grad_in
//  ...

namespace popart {

std::size_t Pipeline::id() { return typeid(Pipeline).hash_code(); }

namespace {

VGraphId getVirtualGraphIdOrSourceIpu(Op *op) {
  if (op->isConvertibleTo<IpuCopyOp>()) {
    auto ipuCopyOp = dynamic_cast<popart::IpuCopyOp *>(op);
    return static_cast<int64_t>(ipuCopyOp->getSourceIpu());
  } else {
    return op->getVirtualGraphId();
  }
}

void setIPUCopyPipelineStage(IpuCopyOp *op) {

  auto in0 = op->inTensor(0);
  if (in0->hasProducer()) {
    auto producer = in0->getProducer();
    auto ps       = producer->getPipelineStage();
    op->setPipelineStage(ps);
  } else {
    if (in0->tensorType() == TensorType::Variable) {
      throw error("Can not copy variable tensor {} between virtual graphs when "
                  "pipelining. All pipeline stages using this tensor should be "
                  "on the same graph.",
                  in0->str());
    } else {
      // Const or Stream tensors
      auto ps = in0->consumers.findLowestPipelineStage();
      op->setPipelineStage(ps);
    }
  }
}

void checkOpsPipelineStage(Graph &graph) {
  // return the pipeline stage or -1 if not set
  // throw error if pipeline stage is negative
  auto getPipelineStage = [](auto x) -> PipelineStage {
    if (x->hasPipelineStage()) {
      auto ps = x->getPipelineStage();
      if (ps < 0) {
        throw error("Op has bad pipeline stage {}", ps);
      }
      return ps;
    } else {
      return -1;
    }
  };

  // collect all ops in each pipeline stage
  std::map<PipelineStage, std::vector<Op *>> pipelineStages;

  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (!op->isConvertibleTo<IpuCopyOp>()) {
      auto ps = getPipelineStage(op);
      pipelineStages[ps].push_back(op);
    }
  }

  // if no ops have had the pipeline stage attribute set, set it to the virtual
  // graph id
  if (pipelineStages.size() == 1 && pipelineStages.count(-1) != 0) {
    for (auto &id_op : graph.getOps()) {
      auto op = id_op.second.get();
      if (!op->isConvertibleTo<IpuCopyOp>()) {
        auto vgraphid = op->getVirtualGraphId();
        op->setPipelineStage(vgraphid);
      }
    }
  }

  std::ostringstream oss;
  oss << "For all IpuCopyOps (other than those copying optimizer tensors):\n"
      << " (1)  set pipeline stage\n"
      << " (2) insert topological constraint that it precedes "
      << "all non-IpuCopyOps in each pipeline stage";
  logging::transform::debug(oss.str());

  for (auto &id_op : graph.getOps()) {
    if (auto copyOp = dynamic_cast<IpuCopyOp *>(id_op.second.get())) {

      // Copies of optimizer Tensors do not run in the main program fragment
      if (!copyOp->copiesOptimizerTensors()) {
        // (1) set pipeline stage
        setIPUCopyPipelineStage(copyOp);
        // (2) insert topological constraint
        graph.topoCons->insert({{
            copyOp,                                       // Key
            pipelineStages.at(copyOp->getPipelineStage()) // OpsBeforeKey
        }});
      }
    }
  }
}

Op *getStashReferenceOp(Tensor *t) {
  // Choose an op for the stash op to copy the vgraph and pstage from.

  // If the tensor has no producer, or the producer is a copy op, then the
  // tensor has been streamed/copied onto this virtual graph just in time to be
  // consumed. There must also be a later consumer on the same virtual graph,
  // otherwise this tensor would not have been a candidate for stashing. Use the
  // consumer with the lowest pipeline stage as the stash ref op.
  if (!t->hasProducer() || t->getProducer()->isConvertibleTo<IpuCopyOp>()) {
    auto consumers  = t->consumers.getOps();
    auto stashRefOp = consumers.at(0);
    for (auto c : t->consumers.getOps()) {
      if (c->getPipelineStage() < stashRefOp->getPipelineStage()) {
        stashRefOp = c;
      }
    }

    return stashRefOp;
  }
  // The tensor has been produced by an op on this virtual graph, and is to be
  // consumed by an op on this virtual graph in a later pipeline stage.
  else {
    return t->getProducer();
  }
}

std::string zeroCandidatesError(Tensor *t, Op *stashRefOp) {
  std::stringstream ss;
  ss << "No candidates for restore op.";

  ss << logging::format("\nTensor: {}", t->id);
  if (t->hasProducer()) {
    auto prod = t->getProducer();
    ss << logging::format("\n  Producer: {}, ps: {}, vg: {}",
                          prod->debugName(),
                          prod->getPipelineStage(),
                          getVirtualGraphIdOrSourceIpu(prod));
  }
  ss << "\n  Consumers:";
  for (auto c : t->consumers.getOps()) {
    ss << logging::format("\n    {}, ps: {}, vg: {}",
                          c->debugName(),
                          c->getPipelineStage(),
                          getVirtualGraphIdOrSourceIpu(c));
  }

  ss << logging::format("\nStash Ref Op: {}, ps: {}, vg: {}",
                        stashRefOp->debugName(),
                        stashRefOp->getPipelineStage(),
                        getVirtualGraphIdOrSourceIpu(stashRefOp));

  return ss.str();
}

Op *searchForRestoreReferenceOp(Tensor *t, Op *stashRefOp) {
  // Find a restore reference Op by searching through the consumers but not
  // crossing IPU boundaries.
  OpSearchHelper toCheck;
  toCheck.pushConsumers(t);
  while (!toCheck.empty()) {
    auto op = toCheck.pop();
    if (!op->isConvertibleTo<IpuCopyOp>()) {
      if (op->getPipelineStage() > stashRefOp->getPipelineStage()) {
        return op;
      } else {
        toCheck.pushOutputConsumers(op);
      }
    }
  }
  return nullptr;
}

bool isProducedOnIPU(Tensor *tensor) {
  // Has a producer and it's a copy
  if (tensor->hasProducer() &&
      dynamic_cast<IpuCopyOp *>(tensor->getProducer())) {
    return false;
  }

  // Doesn't have a producer and it's a stream
  if (!tensor->hasProducer() && tensor->tensorType() == TensorType::Stream) {
    return false;
  }

  return true;
}

std::unique_ptr<IdentityOp> createIdenityCopyOp(Graph &graph,
                                                Tensor *tensor,
                                                VGraphId vGraphId,
                                                PipelineStage pStage) {
  Op::Settings identitySettings(graph, tensor->id + "_pipelineCopy");
  auto op = std::make_unique<IdentityOp>(Onnx::Operators::Identity_1,
                                         identitySettings);
  if (op == nullptr) {
    throw error("Failed to create op Identity Copy op, {}",
                Onnx::Operators::Identity_1);
  }

  // ensure that op is not inplaced
  op->settings.inplacePriorityVeto = {{"IdentityInplace", -1}};

  op->setVirtualGraphId(vGraphId);
  op->setPipelineStage(pStage);
  return op;
}

void insertClonesBeforeIpuCopyConsumers(Graph &graph,
                                        Tensor *tensor,
                                        VGraphId src_ipu,
                                        PipelineStage pStage) {
  // Modify the IR as shown below.
  //
  // Before:
  //
  //  Producer
  //     |
  //    tensor -------
  //     |    \       \
  //     |     \       \
  //     |      \       \
  //     |       \       \
  // Consumer  IpuCopy0  IpuCopy1
  //
  // After:
  //
  //  Producer
  //     |
  //    tensor
  //     |    \
  //     |   Identity
  //     |      \
  //     |     tenor_copy --
  //     |        \         \
  // Consumers   IpuCopy0  IpuCopy1

  std::vector<IpuCopyOp *> ipuCopyConsumers;
  for (auto *c : tensor->consumers.getOps()) {
    auto ipuCopyOp = dynamic_cast<IpuCopyOp *>(c);
    if (ipuCopyOp)
      ipuCopyConsumers.push_back(ipuCopyOp);
  }

  if (ipuCopyConsumers.size() > 0) {
    auto identity = createIdenityCopyOp(graph, tensor, src_ipu, pStage);
    TensorId identityOutput = identity->settings.name;
    identity->connectInTensor(0, tensor->id);
    identity->createAndConnectOutTensor(0, identityOutput);
    identity->setup();
    graph.moveIntoGraph(std::move(identity));

    for (auto *ipuCopyOp : ipuCopyConsumers) {
      for (auto inIndex : ipuCopyOp->input->indices(tensor)) {
        ipuCopyOp->disconnectInTensor(inIndex, tensor);
        ipuCopyOp->connectInTensor(inIndex, identityOutput, src_ipu);
      }
    }
  }
}

void insertCloneBeforeCopiesToHost(Graph &graph,
                                   Tensor *tensor,
                                   VGraphId vGraphId,
                                   PipelineStage pStage) {
  // Since we can't rename anchors (as the user is expecting anchor of
  // known name), we modify the IR as shown below.
  //
  // Before:
  //
  //  Producer
  //     |
  //  anchor_id
  //     |    \
  //     |     \
  //     |      \
  //     |       \
  // Consumers  ToHostStream
  //

  // Intermediate1:
  //
  //  Producer
  //     |
  //  new_tensor_id
  //     |
  //   Identity
  //
  //      anchor_id
  //       /      \
  // Consumers   ToHostStream

  // Final:
  //
  //  Producer
  //     |
  //  new_tensor_id
  //     |     \
  //     |    Identity
  //     |       \
  //     |       anchor_id
  //     |         \
  // Consumers   ToHostStream

  // Intermediate1
  auto producer = tensor->getProducer();
  auto outIndex = producer->outIndex(tensor);
  producer->disconnectOutTensor(tensor);
  TensorId substituteId = tensor->id + "_substitute";
  producer->createAndConnectOutTensor(outIndex, substituteId);
  producer->setup();
  auto identity = createIdenityCopyOp(graph, tensor, vGraphId, pStage);
  identity->connectInTensor(0, substituteId);

  // Final
  identity->connectOutTensor(0, tensor->id);
  identity->setup();
  graph.moveIntoGraph(std::move(identity));
  // for each consumer of anchor tensor, disconnect and reconnect at same
  // indices to its substitute
  for (auto op : tensor->consumers.getOps()) {
    for (auto inIndex : op->input->indices(tensor)) {
      op->disconnectInTensor(inIndex, tensor);
      op->connectInTensor(inIndex, substituteId);
    }
  }
}

Op *getRestoreReferenceOp(Tensor *t, Op *stashRefOp) {
  logging::debug("Collecting restore ref candidates");
  auto consumers = t->consumers.getOps();

  std::vector<Op *> restoreCandidates;
  for (auto c : consumers) {
    if (getVirtualGraphIdOrSourceIpu(c) ==
            getVirtualGraphIdOrSourceIpu(stashRefOp) &&
        c->getPipelineStage() != stashRefOp->getPipelineStage()) {
      restoreCandidates.push_back(c);
    }
  }

  if (restoreCandidates.size() == 0) {
    throw internal_error(zeroCandidatesError(t, stashRefOp));
  }

  // Check all candidates have the same pipeline stage
  PipelineStage restorePipelineStage =
      restoreCandidates.at(0)->getPipelineStage();
  for (auto c : restoreCandidates) {
    if (restorePipelineStage != c->getPipelineStage()) {
      throw error("Conflicting candidates for restore op pipeline stage");
    }
  }

  return restoreCandidates.at(0);
}

// clang-format off
// (a) -> [copy] -> (a_copy0 on pipeline stage N)
// (a) -> [copy] -> (a_copy1 on pipeline stage M)
//                     ==================>
// (a) -> [copy] -> (a_copy0 on pipeline stage N) -> [copy] -> (a_copy1 on pipeline stage M)
// clang-format on
void chainCopies(std::vector<IpuCopyOp *> &copies) {
  for (auto c : copies) {
    if (c->input->n() > 1) {
      // Chaining copies with more than 1 input is possible, but I don't think
      // it will ever occur.
      throw internal_error(
          "Attempting to chain a copy with more than one input.");
    }
  }

  std::sort(copies.begin(), copies.end(), [](auto &lhs, auto &rhs) {
    auto lhsStage = *lhs->outTensor(0)->consumers.findLowestPipelineStage();
    auto rhsStage = *rhs->outTensor(0)->consumers.findLowestPipelineStage();
    return lhsStage < rhsStage;
  });

  for (int i = 1; i < copies.size(); i++) {
    auto prevCopyOp = copies[i - 1];
    auto copyOp     = copies[i];
    auto newPStage =
        prevCopyOp->outTensor(0)->consumers.findLowestPipelineStage();

    copyOp->disconnectInTensor(0, copyOp->inTensor(0));
    copyOp->connectInTensor(0, prevCopyOp->outId(0), prevCopyOp->getDestIpu());
    copyOp->setPipelineStage(newPStage);
  }
}

// Look for and transform groups of copies that may be chained. This prevents
// duplicate copies being created by the contiguate copies transform where:
//   O -> N
//   O -> M
// would become:
//   O -> O+1 -> O+2 -> ... -> N
//   O -> O+1 -> O+2 -> ... -> N -> N+1 -> N+2 -> ... -> M
void chainCopiesTransform(Graph &graph) {
  std::map<TensorId, std::vector<IpuCopyOp *>> copyMap;
  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (!op->copiesOptimizerTensors() && op->isConvertibleTo<IpuCopyOp>()) {
      auto copyOp = dynamic_cast<IpuCopyOp *>(op);
      for (auto tensor : op->input->tensors()) {
        copyMap[tensor->id].push_back(copyOp);
      }
    }
  }

  for (auto &tenId_copies : copyMap) {
    auto &copies = tenId_copies.second;

    if (copies.size() > 1) {
      chainCopies(copies);
    }
  }
}

bool isFullRecompute(Graph &graph) {
  auto &ir = graph.getIr();
  return ir.getSessionOptions().autoRecomputation ==
         RecomputationType::Pipeline;
}

std::vector<TensorId> getStashCandidateTensors(Graph &graph) {

  bool full_recompute = isFullRecompute(graph);

  std::vector<TensorId> toStashCandidateTensors;
  for (auto &tid : graph.getTensors().getAllTensorIds()) {
    auto tensor = graph.getTensors().get(tid);

    if (tensor->consumers.getOps().empty() ||
        tensor->tensorType() == TensorType::Variable ||
        tensor->tensorType() == TensorType::Const ||
        tensor->isOptimizerTensor()) {
      continue;
    }

    // Full Recompute use stashes only on the inputs to an IPU
    // to complete any pipeline stage.
    if (full_recompute && isProducedOnIPU(tensor)) {
      continue;
    }

    auto onlyConsumedByCopies = [](Tensor *t) {
      for (auto consumer : t->consumers.getOps()) {
        if (!consumer->isConvertibleTo<IpuCopyOp>()) {
          return false;
        }
      }
      return true;
    };

    // Get all the stages the tensor is produced/consumed in.
    std::set<PipelineStage> tensorStages = tensor->getPipelineStages();

    // There is no need to stash a tensor that only appears in 1 stage.
    // Unless using full_recompute, then it must be consumed by something other
    // than a copy (it's not just "passing
    //  through")
    if (tensorStages.size() == 1 &&
        !(full_recompute && !onlyConsumedByCopies(tensor))) {
      continue;
    }

    logging::transform::debug("Adding {} to stash candidates", tid);
    toStashCandidateTensors.push_back(tid);
  }

  return toStashCandidateTensors;
}

bool isRecomputable(Op *op) {
  // Copy ops are never recomputable
  if (op->isConvertibleTo<IpuCopyOp>()) {
    return false;
  }
  // Dont recompute the GetRandomSeedOp, or the identity that clones it.
  auto clonesRandomSeed = [&] {
    if (op->isConvertibleTo<IdentityOp>()) {
      auto input = op->inTensor(0);
      return input->hasProducer() &&
             input->getProducer()->isConvertibleTo<GetRandomSeedOp>();
    }
    return false;
  };
  if (op->isConvertibleTo<GetRandomSeedOp>() || clonesRandomSeed()) {
    return false;
  }

  return true;
}

void setRecomputation(Graph &graph,
                      std::vector<TensorId> &toStashCandidateTensors) {
  bool full_recompute = isFullRecompute(graph);

  auto isConsumedByCopy = [](Op *op) {
    for (auto tensor : op->output->tensors()) {
      for (auto consumer : tensor->consumers.getOps()) {
        if (consumer->isConvertibleTo<IpuCopyOp>()) {
          return true;
        }
      }
    }
    return false;
  };

  // Initialise ops to be Recompute, except Ops whose output enters an IpuCopy.
  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (isRecomputable(op)) {
      // In full_recompute all forward ops are Recomputed
      if (!full_recompute && isConsumedByCopy(op)) {
        op->settings.recomputeType = RecomputeType::CHECKPOINT;
      } else {
        op->settings.recomputeType = RecomputeType::RECOMPUTE;
      }
    }
  }

  // Finding initial set of Tensors which are not produced on their IPUs and
  // are not stashed
  TensorSearchHelper frontier;
  auto isStashCandidate = [&](Tensor *t) {
    return std::find(toStashCandidateTensors.cbegin(),
                     toStashCandidateTensors.cend(),
                     t->id) != toStashCandidateTensors.cend();
  };

  for (auto tid : graph.getTensors().getAllTensorIds()) {
    Tensor *tensor = graph.getTensors().get(tid);
    if (!isProducedOnIPU(tensor) && !isStashCandidate(tensor)) {
      frontier.push(tensor);
    }
  }

  // Starting from the initial frontier found above,
  // propogate "CHECKPOINT" forward til either a Stash Tensor or an IPU copy
  // is reached.
  while (!frontier.empty()) {
    Tensor *tensor = frontier.pop();
    for (Op *consumer : tensor->consumers.getOps()) {
      consumer->settings.recomputeType = RecomputeType::CHECKPOINT;
      if (!dynamic_cast<IpuCopyOp *>(consumer)) {
        for (Tensor *consumerOut : consumer->output->tensors()) {
          if (!isStashCandidate(consumerOut)) {
            frontier.push(consumerOut);
          }
        }
      }
    }
  }
}

GetRandomSeedOp *findGetRandomSeedOp(Graph &graph) {
  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (auto x = dynamic_cast<GetRandomSeedOp *>(op)) {
      return x;
    }
  }
  throw error("Could not find an instance of GetRandomSeedOp in graph");
};

TensorId createStashableRandomSeed(GetRandomSeedOp *randomSeedOp) {
  auto randomSeed =
      randomSeedOp->outTensor(GetRandomSeedOp::getUpdatedSeedOutIndex());

  // Create the identity op to clone the random seed.
  logging::transform::debug("Adding Identity Copy for random seed tensor {}",
                            randomSeed->id);
  Op::Settings identitySettings(randomSeedOp->getGraph(),
                                randomSeed->id + "_pipelineCopyOp");
  TensorId identityOutput = randomSeed->id + "_pipelineCopy";
  IdentityOp *identityOp  = [&] {
    auto x = std::make_unique<IdentityOp>(Onnx::Operators::Identity_1,
                                          identitySettings);
    // ensure that op is not inplaced
    x->settings.inplacePriorityVeto = {{"IdentityInplace", -1}};
    auto op                         = x.get();
    randomSeedOp->getGraph().moveIntoGraph(std::move(x));
    return op;
  }();

  identityOp->connectInTensor(IdentityOp::getInIndex(), randomSeed->id);
  identityOp->createAndConnectOutTensor(IdentityOp::getOutIndex(),
                                        identityOutput);
  identityOp->setVirtualGraphId(randomSeedOp->getVirtualGraphId());
  identityOp->setPipelineStage(randomSeedOp->getPipelineStage());
  identityOp->setup();

  auto randomSeedClone = identityOp->outTensor(0);
  // Connect the consumers of random seed to the output of the identity op
  for (auto consumer : randomSeed->consumers.getOps()) {
    if (consumer != identityOp) {
      // Important to copy tensorMap here, as TensorIndex::tensorMap returns a
      // reference and we shall be modifiying it.
      auto inputMap = consumer->input->tensorMap();
      for (auto idx_tensor : inputMap) {
        auto idx         = idx_tensor.first;
        auto inputTensor = idx_tensor.second;
        if (inputTensor == randomSeed) {
          if (auto copyOp = dynamic_cast<IpuCopyOp *>(consumer)) {
            auto sourceIpu = copyOp->getSourceIpu();
            copyOp->disconnectInTensor(idx, randomSeed);
            copyOp->connectInTensor(idx, randomSeedClone->id, sourceIpu);
          } else {
            consumer->disconnectInTensor(idx, randomSeed);
            consumer->connectInTensor(idx, randomSeedClone->id);
          }
        }
      }
    }
  }

  return randomSeedClone->id;
}

bool containsSeedTensor(std::vector<TensorId> ids) {
  bool containsSeedFromHost =
      std::find(ids.begin(),
                ids.end(),
                GetRandomSeedOp::getStreamedSeedTensorId()) != ids.end();
  bool containsUpdatedSeed =
      std::find(ids.begin(),
                ids.end(),
                GetRandomSeedOp::getUpdatedSeedTensorId()) != ids.end();

  if (containsSeedFromHost || containsUpdatedSeed) {
    return true;
  }
  return false;
}

} // namespace

bool Pipeline::apply(Graph &graph) const {

  auto &ir            = graph.getIr();
  bool full_recompute = isFullRecompute(graph);
  // We use numIPUs // replicated graph count for the max vGraph ID.

  // First, some checks that pipelining is compatible with other user options:

  // 1. Pipelining uses the virtual graph API. This must be enabled
  if (!ir.virtualGraphsEnabled()) {
    throw error("Pipelining requires the 'virtualGraphMode' session option "
                "to not be VirtualGraphMode::Off.");
  }

  checkOpsPipelineStage(graph);

  // 2. There must be enough mini-batches of data to fill the pipeline
  int64_t numPipelineStages = ir.getNumPipelineStages();
  if (ir.getSessionOptions().enableGradientAccumulation) {
    if (ir.getSessionOptions().accumulationFactor < numPipelineStages) {
      // For replicated graphs we are replicating the entire pipeline, so these
      // conditions still hold.
      throw error("For pipelining, depth (gradient accumulation factor) must "
                  "equal at least the number of pipeline stages ({})",
                  numPipelineStages);
    }
  } else {
    int64_t bps = static_cast<int64_t>(ir.getDataFlow().batchesPerStep());
    if (bps < numPipelineStages) {
      throw error("For pipelining, depth (batchesPerStep) must equal at least "
                  "the number of pipeline stages ({})",
                  numPipelineStages);
    }
  }

  // 3. Currently recomputation is not supported with pipelining (TODO T9575)
  if (ir.getMainGraph().hasUserRecomputeOps()) {
    throw error("When pipelining is enabled, user annotation for recomputation "
                "is not allowed");
  }

  // 4. Forward layers must be sharded with increasing IPU index
  //    Examples violating this:
  //      Consider the fwd Graph : Op0 -> Op1 -> Op2 -> Op3
  //          e.g. 1) IPU0 : {Op2, Op3}, IPU1 : {Op0, Op1}
  //          e.g. 2) IPU0 : {Op0, Op2}, IPU1 : {Op1, Op3}

  // The checks:
  // 4.2 Copies in the correct direction

  auto getIpuCopyOps = [&graph] {
    // contiguating IpuCopyOps
    std::vector<popart::IpuCopyOp *> ipuCopies;
    for (auto &op_pair : graph.getOps()) {
      auto ipuCopyOp = dynamic_cast<popart::IpuCopyOp *>(op_pair.second.get());
      if (ipuCopyOp) {
        ipuCopies.push_back(ipuCopyOp);
      }
    }
    return ipuCopies;
  };

  chainCopiesTransform(graph);

  // Other sharding assumptions to check:

  // 5. Ir stream tensors cannot be consumed by ops on multiple IPUs
  for (TensorId tid : graph.getTensors().getIds(TensorType::Stream)) {
    auto tensor = graph.getTensors().get(tid);
    std::set<VGraphId> virtualGraphs;
    for (auto c : tensor->consumers.getOps()) {
      virtualGraphs.insert(getVirtualGraphIdOrSourceIpu(c));
    }

    if (virtualGraphs.size() > 1) {
      throw error("For pipelining, stream tensors can only be streamed "
                  "directly onto a single IPU");
    }
  }

  // Now apply the transform

  // 0. Contiguate the IPUCopies
  ContiguateIpuCopyIndicesPattern contiguator;
  for (auto ipuCopyOp : getIpuCopyOps()) {
    if (!ipuCopyOp->isExcludedFromPattern(&contiguator) &&
        contiguator.matches(ipuCopyOp)) {
      logging::transform::debug("Contiguating {}", ipuCopyOp->debugName());
      contiguator.apply(ipuCopyOp);
    }
  }
  ir.updateVertices();

  // verify that all IpuCopies are contiguous
  for (auto ipuCopyOp : getIpuCopyOps()) {
    if (!ipuCopyOp->copiesOptimizerTensors()) {
      auto sourceIpu = ipuCopyOp->getPipelineStage();
      auto destIpu =
          *ipuCopyOp->outTensor(0)->consumers.findLowestPipelineStage();
      auto delta = destIpu - sourceIpu;
      // only copies of optimizer may be non contiguous
      if (delta != 1 && delta != -1) {
        std::stringstream ss;
        ss << logging::format(
            "IpuCopy {} is not contiguous. It copies from IPU {} to "
            "IPU {}. Failed to contiguate all IpuCopyOps",
            ipuCopyOp->debugName(),
            sourceIpu,
            destIpu);
        ss << logging::format("\nin tensor 0: {}",
                              ipuCopyOp->inTensor(0)->str());
        ss << logging::format("\nin tensor 0 producer pipeline stage: {}",
                              sourceIpu);
        ss << logging::format("\nout tensor 0: {}",
                              ipuCopyOp->outTensor(0)->str());
        ss << logging::format(
            "\nout tensor 0 lowest consumer pipeline stage: {}", destIpu);
        throw internal_error(ss.str());
      }
    }
  }

  std::vector<TensorId> toStashCandidateTensors =
      getStashCandidateTensors(graph);

  if (ir.requiresRandomSeed() && containsSeedTensor(toStashCandidateTensors)) {
    // Neither the input or the output of a GetRandomSeedOp should be stashed.
    auto getRandomSeedOp = findGetRandomSeedOp(graph);
    boost::remove_erase(toStashCandidateTensors, getRandomSeedOp->inId(0));
    boost::remove_erase(toStashCandidateTensors, getRandomSeedOp->outId(0));
    // Instead, we need to clone the output of the random seed op and stash
    // that.
    auto stashableRandomSeed = createStashableRandomSeed(getRandomSeedOp);
    toStashCandidateTensors.push_back(stashableRandomSeed);
  }

  std::vector<TensorId> toStashTensors;
  // StashTensorId -> std::pair<StashRefOp, RestoreRefOp>
  std::map<TensorId, std::pair<Op *, Op *>> stashRestoreRefOps;
  // If there is no recomputation, then the candidates for stashing will all be
  // stashed.
  if (!ir.autoRecomputationEnabled()) {
    toStashTensors = toStashCandidateTensors;
  }

  // If there is recomputation, the candidate set is reduced.
  //
  // Candidate Tensors which can be recomputed from other stashing candidates,
  // are filtered out, and their producers are set to RECOMPUTE.
  //
  // The only exceptions are candidate stashing Tensors which are copied to
  // another IPU : these must be stashed even if they recomputable. This
  // guarantees that the correct Tensor is copied after fwd and bwd have
  // executed.
  //
  // Algorithm : initialize all pre-loss Ops to be RECOMPUTE, and then set to
  // CHECKPOINT if (1) cannot be computed from previous Stashed Tensors or (2)
  // must be copied to next IPU.
  else {
    setRecomputation(graph, toStashCandidateTensors);

    logging::transform::debug(
        "Reducing the set of stashing candidate Tensors for recomputation");

    // Filter stash candidates: only stash CHECKPOINT Ops
    for (auto tid : toStashCandidateTensors) {
      auto tensor = graph.getTensors().get(tid);
      if (!tensor->hasProducer() ||
          tensor->getProducer()->settings.recomputeType !=
              RecomputeType::RECOMPUTE) {
        // For full_recompute if a stash candidate doesn't have a
        // restoreReference then it is not required for recomputation during the
        // backwards pass.
        if (full_recompute && tensor->getPipelineStages().size() == 1) {
          auto stashRef   = getStashReferenceOp(tensor);
          auto restoreRef = searchForRestoreReferenceOp(tensor, stashRef);
          if (restoreRef == nullptr) {
            continue;
          }
          stashRestoreRefOps.insert({tid, {stashRef, restoreRef}});
        }
        toStashTensors.push_back(tid);
      }
    }

    // If the set of stash candidates has been reduced, recomputation needs to
    // be reset.
    if (toStashTensors.size() != toStashCandidateTensors.size()) {
      setRecomputation(graph, toStashTensors);
    }
  }

  logging::transform::debug("Final Stash Tensors");
  for (auto tid : toStashTensors) {
    logging::transform::debug("  {}", tid);
  }

  // 2. For each Tensor to be stashed, create a single stash
  //    and (in-place) restore op
  Op::Settings settings(graph, "");

  std::map<PipelineStage, std::vector<Op *>> restoreOps;

  for (auto &tid : toStashTensors) {
    auto tensor = graph.getTensors().get(tid);

    if (tensor->consumers.getOps().empty()) {
      throw internal_error("request to stash Tensor {} with no "
                           "consumers, bailing",
                           tensor->str());
    }

    Op *stashRefOp;
    Op *restoreRefOp;
    if (stashRestoreRefOps.find(tid) != stashRestoreRefOps.end()) {
      auto refs    = stashRestoreRefOps.at(tid);
      stashRefOp   = refs.first;
      restoreRefOp = refs.second;
    } else {
      stashRefOp   = getStashReferenceOp(tensor);
      restoreRefOp = getRestoreReferenceOp(tensor, stashRefOp);
    }

    auto stashSize =
        restoreRefOp->getPipelineStage() - stashRefOp->getPipelineStage() + 1;

    // Stash
    auto stashOp_up = std::make_unique<StashOp>(
        Onnx::CustomOperators::Stash, stashSize, settings);
    auto stashOp = stashOp_up.get();
    graph.moveIntoGraph(std::move(stashOp_up));
    stashOp->setVirtualGraphId(getVirtualGraphIdOrSourceIpu(stashRefOp));
    stashOp->setPipelineStage(stashRefOp->getPipelineStage());
    stashOp->connectInTensor(StashOp::getInIndex(), tid);
    auto stashId = stashOp->getStashedTensorId();
    stashOp->createAndConnectOutTensor(StashOp::getOutIndex(), stashId);
    stashOp->setup();

    logging::transform::debug("Adding stash of size {} of activations {} for "
                              "pipelining. Stash stage: {}, Restore stage {}",
                              stashOp->getStashSize(),
                              tensor->id,
                              stashOp->getPipelineStage(),
                              restoreRefOp->getPipelineStage());

    // Full Recomputation
    // If a preLoss stash tensors is consumed by an IpuCopy
    // it must not be inplace, but stashes needed for recomputation must be
    // inplace. To resolve this contradiction an IdentityOp is inserted between
    // the the stashed tensor and the IpuCopy
    if (full_recompute) {
      insertClonesBeforeIpuCopyConsumers(
          graph,
          tensor,
          getVirtualGraphIdOrSourceIpu(stashRefOp),
          stashRefOp->getPipelineStage());
    }

    // Restore
    auto tidConsumers = tensor->consumers.getOps();

    // Should op be Restore (outplace) or RestoreInplace?
    bool isInplace = true;
    if (ir.isAnchored(tid)) {
      isInplace = false;
    } else {
      for (Op *tidConsumer : tidConsumers) {
        if (tidConsumer->isIpuCopyOp()) {
          isInplace = false;
        }
      }
    }

    // RECOMPUTE ops must be inplace, confirm:
    for (Op *tidConsumer : tidConsumers) {
      if (tidConsumer->settings.recomputeType == RecomputeType::RECOMPUTE) {
        if (isInplace == false) {
          throw error("A recompute Op consumes a stashed Tensor, therefore "
                      "the stashing must be in-place. But some previous logic "
                      "has set the stashing to be non-inplace");
        }
      }
    }

    RestoreOp *restoreOp;
    if (isInplace) {
      restoreOp = addNewRestoreInplaceOp(graph, stashSize);
      // RestoreInplaceOp has an extra input - the act tensor it is in-place
      // restoring.
      restoreOp->connectInTensor(RestoreInplaceOp::getActToRestoreInIndex(),
                                 tid);
    } else {
      restoreOp = addNewRestoreOp(graph, stashSize);
    }

    restoreOp->setVirtualGraphId(getVirtualGraphIdOrSourceIpu(restoreRefOp));
    restoreOp->setPipelineStage(restoreRefOp->getPipelineStage());
    restoreOp->connectInTensor(RestoreOp::getStashInIndex(), stashId);
    auto restoreId = restoreOp->getRestoredTensorId();
    restoreOp->createAndConnectOutTensor(RestoreOp::getRestoredActOutIndex(),
                                         restoreId);

    // Disconnect tid from all consumers in the restore ops pipeline stage,
    // reconnect to restoreId
    bool inplaceRestoreRequiredForRecompute = false;
    for (Op *tidConsumer : tidConsumers) {
      if (tidConsumer->getPipelineStage() == restoreOp->getPipelineStage()) {
        for (auto i : tidConsumer->input->indicesMap().at(tensor)) {
          tidConsumer->disconnectInTensor(i, tensor);
          tidConsumer->connectInTensor(i, restoreId);
        }
      } else {
        logging::transform::debug("Not connecting consumer {} to restore op {} "
                                  "as they have different pipeline stages ",
                                  tidConsumer->str(),
                                  restoreOp->str());
      }
      if ((dynamic_cast<RestoreInplaceOp *>(restoreOp) &&
           tidConsumer->settings.recomputeType == RecomputeType::RECOMPUTE &&
           tidConsumer->getPipelineStage() < restoreOp->getPipelineStage())) {
        inplaceRestoreRequiredForRecompute = true;
      }
    }

    // This InplaceRestoreOp may have no consumers as they are all inplicit due
    // to recomputation. Therefore it must not be pruned.
    if (inplaceRestoreRequiredForRecompute) {
      restoreOp->pruneable = false;
    }

    restoreOp->setup();
    restoreOps[restoreRefOp->getPipelineStage()].push_back(restoreOp);

    // refresh consumers of tensor
    tidConsumers = tensor->consumers.getOps();

    bool noConsumersOfRestore =
        restoreOp->output->tensor(0)->consumers.getOps().empty();

    if (noConsumersOfRestore && !inplaceRestoreRequiredForRecompute) {
      bool noRecomputeConsumersOfStash = std::none_of(
          tidConsumers.cbegin(), tidConsumers.cend(), [](const Op *op) {
            return op->settings.recomputeType == RecomputeType::RECOMPUTE;
          });
      std::ostringstream oss3;
      oss3 << "The RestoreOp " << restoreOp->str() << " on pipeline stage "
           << restoreOp->getPipelineStage()
           << " has no consumers. This seems strange, so bailing. "
           << "noRecomputeConsumersOfStash = " << noRecomputeConsumersOfStash
           << " where the tensor being stashed is " << tensor->str();
      throw internal_error(oss3.str());
    }

    for (auto tidConsumer : tidConsumers) {

      // StashOp should be before all other consumers
      // (required for recompute to work)
      if (tidConsumer != stashOp) {
        graph.topoCons->insert(stashOp, tidConsumer);
      }

      // RestoreOp should be after all other consumers
      // (required for inplacing to work)
      if (tidConsumer != restoreOp && tidConsumer != stashOp) {
        graph.topoCons->insert(tidConsumer, restoreOp);
      }
    }
  }

  // Any tensor that is created by a recomputed op may be overwritten
  // by the recompute phase before it is copied to the next IPU, or back
  // to host (in the case of anchor tensors). So insert a identity
  // (clone) between the op and the copy.
  if (full_recompute) {
    for (auto &tid : graph.getTensors().getAllTensorIds()) {
      auto tensor = graph.getTensors().get(tid);

      // Stash tensors have already been covered above
      if (std::find(toStashTensors.cbegin(), toStashTensors.cend(), tid) !=
          toStashTensors.cend()) {
        continue;
      }

      if (tensor->hasProducer()) {
        Op *producer = tensor->getProducer();
        if (producer->settings.recomputeType == RecomputeType::RECOMPUTE) {
          insertClonesBeforeIpuCopyConsumers(
              graph,
              tensor,
              getVirtualGraphIdOrSourceIpu(producer),
              producer->getPipelineStage());

          if (ir.isAnchored(tid) && tensor->consumers.getTotal() > 0) {
            insertCloneBeforeCopiesToHost(
                graph,
                tensor,
                getVirtualGraphIdOrSourceIpu(producer),
                producer->getPipelineStage());
          }
        }
      }
    }
  }
  return true;
}

RestoreOp *Pipeline::addNewRestoreOp(Graph &graph, int64_t stashSize) const {
  Op::Settings settings(graph, "");
  auto restoreOp_up = std::make_unique<RestoreOp>(
      Onnx::CustomOperators::Restore, stashSize, settings);
  auto restoreOp = restoreOp_up.get();
  graph.moveIntoGraph(std::move(restoreOp_up));

  return restoreOp;
}

RestoreInplaceOp *Pipeline::addNewRestoreInplaceOp(Graph &graph,
                                                   int64_t stashSize) const {
  Op::Settings settings(graph, "");
  auto restoreOp_up = std::make_unique<RestoreInplaceOp>(
      Onnx::CustomOperators::RestoreInplace, stashSize, settings);
  auto restoreOp = restoreOp_up.get();
  graph.moveIntoGraph(std::move(restoreOp_up));

  return restoreOp;
}

namespace {
bool init = Transform::registerTransform(new Pipeline);
}

} // namespace popart
