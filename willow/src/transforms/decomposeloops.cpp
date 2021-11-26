// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <queue>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/graphutils.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/exchange/hostcopy.hpp>
#include <popart/op/exchange/remote.hpp>
#include <popart/op/init.hpp>
#include <popart/op/iotilecopy.hpp>
#include <popart/op/loop.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/decomposeloops.hpp>
#include <popart/util.hpp>

namespace popart {

std::ostream &operator<<(std::ostream &os, const DecomposeLoopOpType &dlopt) {
  switch (dlopt) {
  case DecomposeLoopOpType::AuxiliaryBefore:
    os << "AuxiliaryBefore";
    break;
  case DecomposeLoopOpType::IoBeforeCompute:
    os << "IoBeforeCompute";
    break;
  case DecomposeLoopOpType::IoToCompute:
    os << "IoToCompute";
    break;
  case DecomposeLoopOpType::Compute:
    os << "Compute";
    break;
  case DecomposeLoopOpType::ComputeToIo:
    os << "ComputeToIo";
    break;
  case DecomposeLoopOpType::IoAfterCompute:
    os << "IoAfterCompute";
    break;
  case DecomposeLoopOpType::AuxiliaryAfter:
    os << "AuxiliaryAfter";
    break;
  default:
    os << "Undefined";
    break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const DecomposeLoopModel &m) {
  os << m.getName();
  return os;
}

std::size_t DecomposeLoops::id() { return typeid(DecomposeLoops).hash_code(); }

DecomposeLoopModel::DecomposeLoopModel()
    : topoConLevelBefore(DecomposeTopoConLevel::Full),
      topoConLevelLoop(DecomposeTopoConLevel::Full),
      topoConLevelAfter(DecomposeTopoConLevel::Full),
      computeLikeExchangeStrategies{ExchangeStrategy::JustInTime} {}

DecomposeLoopModel::DecomposeLoopModel(
    DecomposeTopoConLevel topoConLevelBefore_,
    DecomposeTopoConLevel topoConLevelLoop_,
    DecomposeTopoConLevel topoConLevelAfter_,
    const std::set<ExchangeStrategy> &computeLikeExchangeStrategies_)
    : topoConLevelBefore(topoConLevelBefore_),
      topoConLevelLoop(topoConLevelLoop_),
      topoConLevelAfter(topoConLevelAfter_),
      computeLikeExchangeStrategies(computeLikeExchangeStrategies_) {}

bool DecomposeLoopModel::hasDependencyConflict(
    LoopIteration iterFrom,
    LoopIteration iterTo,
    DecomposeLoopOpType typeFrom,
    DecomposeLoopOpType typeTo) const {
  return typeToPosition(typeFrom, iterFrom) > typeToPosition(typeTo, iterTo);
}

int DecomposeLoopUnrollModel::typeToPosition(DecomposeLoopOpType type,
                                             LoopIteration iteration) const {
  return static_cast<int>(type) +
         iteration * static_cast<int>(DecomposeLoopOpType::N);
}

LoopIteration
DecomposeLoopUnrollModel::getApparentIteration(DecomposeLoopOpType type,
                                               int unrollIndex) const {
  if (unrollIndex == 0) {
    return 0;
  } else if (unrollIndex == 1) {
    return 2;
  } else {
    return 1;
  }
}

bool DecomposeLoopUnrollModel::isBeforeLoop(DecomposeLoopOpType type,
                                            int unrollIndex) const {
  return unrollIndex == 0;
}

DecomposeLoopOverlapModel::DecomposeLoopOverlapModel(
    DecomposeTopoConLevel topoConLevelBefore_,
    DecomposeTopoConLevel topoConLevelLoop_,
    DecomposeTopoConLevel topoConLevelAfter_,
    const std::set<ExchangeStrategy> &computeLikeExchangeStrategies_)
    : DecomposeLoopModel(topoConLevelBefore_,
                         topoConLevelLoop_,
                         topoConLevelAfter_,
                         computeLikeExchangeStrategies_) {}

int DecomposeLoopOverlapModel::typeToPosition(DecomposeLoopOpType type,
                                              LoopIteration iteration) const {
  std::vector<int> lookup{0,  3,  8,   // AuxiliaryBefore
                          1,  4,  10,  // IoBeforeCompute
                          2,  6,  13,  // IoToCompute
                          5,  11, 16,  // Compute
                          7,  14, 18,  // ComputeToIo
                          9,  15, 19,  // IoAfterCompute
                          12, 17, 20}; // AuxiliaryAfter
  return lookup.at(static_cast<int>(type) * 3 + iteration);
}

LoopIteration
DecomposeLoopOverlapModel::getApparentIteration(DecomposeLoopOpType type,
                                                int unrollIndex) const {
  if (unrollIndex == -1) {
    if (type >= DecomposeLoopOpType::IoAfterCompute) {
      return 0;
    }
    if (type >= DecomposeLoopOpType::Compute) {
      return 1;
    }
    if (type >= DecomposeLoopOpType::AuxiliaryBefore) {
      return 2;
    }
  } else if (unrollIndex == 0) {
    if (type >= DecomposeLoopOpType::IoAfterCompute) {
      return 1;
    } else {
      return 0;
    }
  } else if (unrollIndex == 1) {
    if (type >= DecomposeLoopOpType::Compute) {
      return 2;
    } else {
      return 1;
    }
  }
  return -1;
}

bool DecomposeLoopOverlapModel::isBeforeLoop(DecomposeLoopOpType type,
                                             int unrollIndex) const {
  if (type <= DecomposeLoopOpType::ComputeToIo && unrollIndex <= 0) {
    return true;
  }
  if (type <= DecomposeLoopOpType::IoToCompute && unrollIndex <= 1) {
    return true;
  }
  return false;
}

namespace {

// An Op that should be classified as Compute
bool isComputeOp(Op *op) { return op->settings.tileSet == TileSet::Compute; }

// An Op that is IO, and on IO tiles
bool isIOOp(Op *op) {
  return op->isConvertibleTo<RemoteLoadOp>() ||
         op->isConvertibleTo<RemoteStoreOp>() ||
         op->isConvertibleTo<HostLoadOp>() ||
         op->isConvertibleTo<HostStoreOp>();
}

// An Op that is IO, and on IO tiles, but still to be classified as Compute
bool isComputeLikeIOOp(std::set<ExchangeStrategy> computeLikeStrategies,
                       Op *op) {
  auto &ir             = op->getIr();
  bool isComputeLikeIo = false;

  auto isOpComputeLike = [&ir, &computeLikeStrategies](Op *opToCheck) {
    if (auto hostLoadOp = dynamic_cast<HostLoadOp *>(opToCheck)) {
      auto exchangeStrategy = ir.getTensor(hostLoadOp->getHostStreamTensorId())
                                  ->inputSettings.exchangeStrategy();
      if (hostLoadOp->settings.tileSet == TileSet::IO &&
          computeLikeStrategies.find(exchangeStrategy) !=
              computeLikeStrategies.end()) {
        return true;
      }
      return false;
    }
    if (auto hostStoreOp = dynamic_cast<HostStoreOp *>(opToCheck)) {
      auto art = ir.getDataFlow().getAnchorReturnTypeMap().at(
          hostStoreOp->getHostStreamTensorId());
      auto exchangeStrategy = art.exchangeStrategy();
      if (hostStoreOp->settings.tileSet == TileSet::IO &&
          computeLikeStrategies.find(exchangeStrategy) !=
              computeLikeStrategies.end()) {
        return true;
      }
      return false;
    }
    return false;
  };

  isComputeLikeIo |= isOpComputeLike(op);

  graphutils::traverse(
      op->output->tensors(),
      [&isComputeLikeIo, &isOpComputeLike](Tensor *t) -> bool {
        for (auto consumer : t->consumers.getOps()) {
          isComputeLikeIo |= isOpComputeLike(consumer);
        }
        return true;
      },
      [op](Op *c, Tensor *t0, Tensor *t1) -> bool {
        if (c->getGraph().id != op->getGraph().id) {
          return false;
        }
        return c->isConvertibleTo<HostLoadOp>() ||
               c->isConvertibleTo<HostStoreOp>() ||
               c->isConvertibleTo<InitOp>();
      },
      graphutils::TraversalType::DepthFirst,
      graphutils::VisitType::Pre,
      graphutils::TraversalDirection::Forward);

  return isComputeLikeIo;
}

} // namespace

bool DecomposeLoops::addTopoCon(Graph &graph,
                                Op *before,
                                Op *after,
                                bool tied) const {
  graph.topoCons->insert(before, after, tied);
  return true;
}

DecomposeLoopOpType
DecomposeLoops::getType(const DecomposeLoopModel &model,
                        const std::map<Op *, DecomposeLoopOpType> &opToType,
                        Op *op,
                        std::set<DecomposeLoopOpType> prevTypes) const {

  prevTypes.insert(DecomposeLoopOpType::AuxiliaryBefore);

  for (auto input : op->input->tensorMap()) {
    if (input.second->hasProducer()) {
      prevTypes.insert(opToType.at(input.second->getProducer()));
    }
  }

  auto befores = op->getGraph().topoCons->getBefores(op);
  for (Op *before : befores) {
    prevTypes.insert(opToType.at(before));
  }

  if (isComputeOp(op) ||
      isComputeLikeIOOp(model.getComputeLikeExchangeStrategies(), op)) {
    if (op->isConvertibleTo<IoTileCopyOp>()) {
      if (*prevTypes.rbegin() < DecomposeLoopOpType::Compute) {
        return DecomposeLoopOpType::IoToCompute;
      } else if (*prevTypes.rbegin() < DecomposeLoopOpType::ComputeToIo) {
        return DecomposeLoopOpType::Compute;
      } else {
        return DecomposeLoopOpType::AuxiliaryAfter;
      }
    }
    if (*prevTypes.rbegin() < DecomposeLoopOpType::ComputeToIo) {
      return DecomposeLoopOpType::Compute;
    } else {
      return DecomposeLoopOpType::AuxiliaryAfter;
    }
  } else {
    if (isIOOp(op)) {
      if (*prevTypes.rbegin() < DecomposeLoopOpType::IoToCompute) {
        return DecomposeLoopOpType::IoBeforeCompute;
      } else if (*prevTypes.rbegin() < DecomposeLoopOpType::AuxiliaryAfter) {
        return DecomposeLoopOpType::IoAfterCompute;
      }
    }
    if (op->isConvertibleTo<IoTileCopyOp>() &&
        *prevTypes.rbegin() < DecomposeLoopOpType::IoAfterCompute) {
      return DecomposeLoopOpType::ComputeToIo;
    }
    if (*prevTypes.rbegin() < DecomposeLoopOpType::IoBeforeCompute) {
      return DecomposeLoopOpType::AuxiliaryBefore;
    } else if (*prevTypes.rbegin() >= DecomposeLoopOpType::IoAfterCompute) {
      return DecomposeLoopOpType::AuxiliaryAfter;
    } else if (*prevTypes.rbegin() >= DecomposeLoopOpType::Compute) {
      return DecomposeLoopOpType::ComputeToIo;
    } else {
      return *prevTypes.rbegin();
    }
  }
}

void DecomposeLoops::decomposeLoop(Graph &graph,
                                   LoopOp *loopOp,
                                   const DecomposeLoopModel &model) const {

  auto &ir = graph.getIr();

  Graph &subgraph  = loopOp->getCalledGraph();
  int unrollFactor = 2;

  if (loopOp->getTripCountValue() < unrollFactor) {
    return;
  }

  logging::transform::trace("[DecomposeLoops] Decomposing {} with model {} "
                            "(compute like IO strategies: {})",
                            loopOp->debugName(),
                            model,
                            model.getComputeLikeExchangeStrategies());

  std::map<DecomposeLoopOpType, std::vector<Op *>> opsByType;
  std::map<Op *, DecomposeLoopOpType> opToType;

  auto schedule = subgraph.getOpSchedule({}, RequireOptimalSchedule::No);

  // Classify operations
  bool keep_going       = true;
  int opToTypeIteration = 0;
  while (keep_going) {
    logging::transform::trace(
        "[DecomposeLoops] Classifying operations, iteration {}",
        opToTypeIteration);
    keep_going = false;
    for (size_t i = 0; i < schedule.size(); ++i) {
      Op *op = schedule.at(i);

      DecomposeLoopOpType type = DecomposeLoopOpType::AuxiliaryBefore;
      if (opToTypeIteration > 0) {
        type = opToType[op];
        for (auto &input : op->input->tensorMap()) {
          if (input.second->isLoopInput() &&
              !input.second->isImplicitLoopInput()) {
            InIndex sgInIndex   = input.second->getGraphInputIndex();
            OutIndex sgOutIndex = sgInIndex - 1;
            Tensor *t           = loopOp->getCalledGraph().getTensors().get(
                loopOp->getCalledGraph().getOutputId(sgOutIndex));
            if (t->hasProducer()) {
              Op *producer                     = t->getProducer();
              DecomposeLoopOpType producerType = opToType.at(producer);
              if (model.hasDependencyConflict(0, 1, producerType, type) ||
                  model.hasDependencyConflict(1, 2, producerType, type)) {
                auto nextType = static_cast<DecomposeLoopOpType>(
                    static_cast<int>(type) + 1);
                logging::transform::trace(
                    "[DecomposeLoops] Classifying operations, conflict {} {} < "
                    "{} {}, trying type {}",
                    op->debugName(),
                    type,
                    producer->debugName(),
                    producerType,
                    nextType);
                keep_going = true;
                type       = nextType;
                break;
              }
            }
          }
        }
      } else {
        keep_going = true;
      }

      type         = getType(model, opToType, op, {type});
      opToType[op] = type;
    }
    ++opToTypeIteration;
  }

  // Clone operations
  std::map<Op *, std::vector<Op *>> clones;
  std::map<Op *, Op *> originals;

  // Map to preserve the original inputs/outputs for rewiring purposes
  std::map<Op *, std::map<int, Tensor *>> inputMaps;
  std::map<Op *, std::map<int, Tensor *>> outputMaps;

  for (size_t i = 0; i < schedule.size(); ++i) {
    Op *op = schedule.at(i);

    auto type = opToType[op];
    opsByType[type].push_back(op);

    logging::transform::trace("[DecomposeLoops] Op {} type {} tile set {}",
                              op->debugName(),
                              type,
                              op->settings.tileSet);

    for (int j = 0; j < unrollFactor; ++j) {
      auto cloneOpUp = op->clone();
      Op *cloneOp    = cloneOpUp.get();

      // Move from loop body graph to parent graph
      graph.moveIntoGraph(std::move(cloneOpUp));
      cloneOp->setScope(graph.getScope());

      cloneOp->setPipelineStage(loopOp->getOptionalPipelineStage());
      cloneOp->setExecutionPhase(loopOp->getOptionalExecutionPhase());

      clones[op].push_back(cloneOp);
      originals[cloneOp] = op;
    }
    inputMaps[op]  = op->input->tensorMap();
    outputMaps[op] = op->output->tensorMap();
  }

  // Decodes which iteration an operation would belong to
  // (assuming a trip count of 3 before unrolling)
  // (see enum class DecomposeLoopOpType diagram)
  // Only valid for unrollFactor == 2
  auto getApparentIteration = [&opToType, &model](Op *op, int unrollIndex) {
    DecomposeLoopOpType type = opToType.at(op);
    return model.getApparentIteration(type, unrollIndex);
  };

  // Only valid for unrollFactor == 2
  auto isBeforeLoop = [&opToType, &model](Op *op, int unrollIndex) {
    DecomposeLoopOpType type = opToType.at(op);
    return model.isBeforeLoop(type, unrollIndex);
  };

  // Only valid for unrollFactor == 2
  auto isLastBeforeLoop = [&isBeforeLoop](Op *op, int unrollIndex) {
    return isBeforeLoop(op, unrollIndex) &&
           (unrollIndex == 1 || !isBeforeLoop(op, 1));
  };

  LoopTensorMap beforeLoopTensorIterMap;
  LoopTensorMap loopTensorIterMap;
  LoopTensorMap afterLoopTensorIterMap;

  // 1.) Hook up Ops before the loop
  for (size_t i = 0; i < schedule.size(); ++i) {
    Op *op = schedule.at(i);
    for (auto &input : inputMaps[op]) {
      if (input.second->isLoopInput()) {
        InIndex sgInIndex = input.second->getGraphInputIndex();
        InIndex opInIndex = loopOp->subgraphInToOpInIndex(sgInIndex);
        beforeLoopTensorIterMap[{input.second->id, 0}] =
            loopOp->inTensor(opInIndex)->id;
        if (input.second->isImplicitLoopInput()) {
          beforeLoopTensorIterMap[{input.second->id, 1}] =
              loopOp->inTensor(opInIndex)->id;
        }
      }
    }
  }

  for (int j = 0; j < unrollFactor; ++j) {
    for (size_t i = 0; i < schedule.size(); ++i) {
      Op *op = schedule.at(i);
      if (isBeforeLoop(op, j)) {
        // Inputs
        for (auto &input : inputMaps[op]) {
          auto it = beforeLoopTensorIterMap.find(
              {input.second->id, getApparentIteration(op, j)});
          if (it != beforeLoopTensorIterMap.end()) {
            clones[op][j]->connectInTensor(input.first, it->second);
          } else if (input.second->tensorType() == TensorType::Const) {
            TensorId newConstId;
            if (input.second->id.find(reservedConstValuePrefix()) !=
                std::string::npos) {
              newConstId = removeScope(op->getGraph(), input.second->id);
            } else {
              newConstId = ir.createIntermediateTensorId(
                  removeScope(op->getGraph(), input.second->id));
            }
            newConstId = addScope(graph, newConstId);
            if (!graph.getTensors().getConstIds().contains(newConstId)) {
              graph.getTensors().addConstInit(
                  newConstId,
                  input.second->info,
                  input.second->tensorData()->data());
            }
            clones[op][j]->connectInTensor(input.first, newConstId);
          } else {
            throw error(
                "[DecomposeLoops] Cannot connect {} input {} unrollIndex {}",
                clones[op][j]->debugName(),
                input.first,
                j);
          }
        }
        // Outputs
        for (auto &output : outputMaps[op]) {
          TensorId outTensorId =
              addScope(graph, removeScope(op->getGraph(), output.second->id));
          TensorId newOutTensorId = ir.createIntermediateTensorId(outTensorId);
          clones[op][j]->createAndConnectOutTensor(output.first,
                                                   newOutTensorId);

          beforeLoopTensorIterMap[{
              output.second->id, getApparentIteration(op, j)}] = newOutTensorId;

          if (output.second->isGraphOutput()) {
            OutIndex sgOutIndex = output.second->getGraphOutputIndex();
            InIndex sgInIndex   = sgOutIndex + 1;
            InIndex opInIndex   = loopOp->subgraphInToOpInIndex(sgInIndex);
            TensorId sgOutId    = output.second->id;
            TensorId sgInId = loopOp->getCalledGraph().getInputId(sgInIndex);
            if (isLastBeforeLoop(op, j)) {
              loopOp->disconnectInTensor(opInIndex);
              loopOp->connectInTensor(sgInIndex, newOutTensorId);
              loopTensorIterMap[{sgInId, getApparentIteration(op, j) + 1}] =
                  sgInId;
            }
            beforeLoopTensorIterMap[{sgInId, getApparentIteration(op, j) + 1}] =
                newOutTensorId;
            logging::transform::trace("[DecomposeLoops] Iteration {} output {} "
                                      "is iteration {} input {}",
                                      getApparentIteration(op, j),
                                      sgOutId,
                                      getApparentIteration(op, j) + 1,
                                      sgInId);
          }
        }
        logging::transform::trace(
            "[DecomposeLoops] Setting up op {} unrollIndex {}",
            clones[op][j]->debugName(),
            j);
        clones[op][j]->setup();
      }
    }
  }

  for (auto insideTensor : loopTensorIterMap) {
    logging::transform::trace(
        "[DecomposeLoops] Inside loop tensor {} -> {}, iteration {}",
        insideTensor.first.first,
        insideTensor.second,
        insideTensor.first.second);
  }

  // 2.) Hook up Ops inside the loop
  for (size_t i = 0; i < schedule.size(); ++i) {
    Op *op                          = schedule.at(i);
    LoopIteration apparentIteration = getApparentIteration(op, -1);
    // Register outputs
    for (auto &output : outputMaps[op]) {
      loopTensorIterMap[{output.second->id, apparentIteration}] =
          output.second->id;
      if (output.second->isGraphOutput()) {
        OutIndex sgOutIndex = output.second->getGraphOutputIndex();
        OutIndex opOutIndex = loopOp->subgraphOutToOpOutIndex(sgOutIndex);
        InIndex sgInIndex   = sgOutIndex + 1;
        TensorId sgOutId    = output.second->id;
        TensorId sgInId     = loopOp->getCalledGraph().getInputId(sgInIndex);
        TensorId opOutId    = loopOp->outId(opOutIndex);
        loopTensorIterMap[{sgInId, apparentIteration + 1}]      = sgOutId;
        afterLoopTensorIterMap[{sgInId, apparentIteration + 1}] = opOutId;
      }
    }
  }

  for (size_t i = 0; i < schedule.size(); ++i) {
    Op *op                          = schedule.at(i);
    LoopIteration apparentIteration = getApparentIteration(op, -1);
    for (auto &input : inputMaps[op]) {
      if (input.second->tensorType() == TensorType::Const) {
        // Constants remain connected (no iteration dependency)
        continue;
      }
      if (input.second->isImplicitLoopInput()) {
        // Implicit loop inputs remain connected (no iteration dependency)
        continue;
      }

      // Check if the input is already available inside the loop
      auto itIn = loopTensorIterMap.find({input.second->id, apparentIteration});
      if (itIn != loopTensorIterMap.end()) {
        op->disconnectInTensor(input.first);
        op->connectInTensor(input.first, itIn->second);
      } else {
        // Check if the input exists before the loop
        auto itOut =
            beforeLoopTensorIterMap.find({input.second->id, apparentIteration});
        if (itOut != beforeLoopTensorIterMap.end()) {
          TensorId opTensorIdIn  = itOut->second;
          Tensor *opTensorIn     = graph.getTensors().get(itOut->second);
          Op *producer           = opTensorIn->getProducer();
          TensorId opTensorIdOut = ir.createIntermediateTensorId(opTensorIdIn);
          TensorId sgTensorIdIn =
              ir.createIntermediateTensorId(input.second->id);
          TensorId sgTensorIdOut;

          // Get Op inside the loop matching the producer outside of the loop
          auto origIt = originals.find(producer);
          if (origIt == originals.end()) {
            throw error("[DecomposeLoops] Could not find original to Op {}, "
                        "indicating it is not a cloned operation.",
                        producer->debugName());
          }
          Op *orig = origIt->second;

          if (getApparentIteration(orig, -1) == apparentIteration + 1) {
            // Next iteration is inside the loop, wire output -> input
            sgTensorIdOut = orig->output->id(producer->outIndex(opTensorIn));
          } else {
            // Next iteration still occurring before the loop,
            auto itInNext = loopTensorIterMap.find(
                {input.second->id, apparentIteration + 1});
            if (itInNext != loopTensorIterMap.end()) {
              // a.) The next iteration is already connected into the loop
              sgTensorIdOut = itInNext->second;
            } else {
              // b.) The next iteration needs to be connected into the loop
              auto itOutNext = beforeLoopTensorIterMap.find(
                  {input.second->id, apparentIteration + 1});
              TensorId opTensorIdInNext = itOutNext->second;
              TensorId sgTensorIdOutNext =
                  orig->output->id(producer->outIndex(opTensorIn));
              TensorId opTensorIdOutNext =
                  ir.createIntermediateTensorId(opTensorIdIn);
              sgTensorIdOut = ir.createIntermediateTensorId(input.second->id);

              if (itOutNext != beforeLoopTensorIterMap.end()) {
                loopOp->addLoopInput(LoopOp::getFirstInputInIndex(),
                                     opTensorIdInNext,
                                     sgTensorIdOut,
                                     false);
                loopOp->addLoopOutput(LoopOp::getFirstOutputOutIndex(),
                                      opTensorIdOutNext,
                                      sgTensorIdOutNext,
                                      false);

                loopTensorIterMap[{input.second->id, apparentIteration + 1}] =
                    sgTensorIdOut;
                afterLoopTensorIterMap[{input.second->id,
                                        apparentIteration + 2}] =
                    opTensorIdOutNext;
              } else {
                throw error(
                    "[DecomposeLoops] Cannot connect {} input {} iteration {}",
                    op->debugName(),
                    input.first,
                    apparentIteration);
              }
            }
          }

          loopOp->addLoopInput(LoopOp::getFirstInputInIndex(),
                               opTensorIdIn,
                               sgTensorIdIn,
                               false);
          loopOp->addLoopOutput(LoopOp::getFirstOutputOutIndex(),
                                opTensorIdOut,
                                sgTensorIdOut,
                                false);

          loopTensorIterMap[{input.second->id, apparentIteration}] =
              sgTensorIdIn;
          afterLoopTensorIterMap[{input.second->id, apparentIteration + 1}] =
              opTensorIdOut;

          op->disconnectInTensor(input.first);
          op->connectInTensor(input.first, sgTensorIdIn);
        } else {
          throw error(
              "[DecomposeLoops] Cannot connect {} input {} iteration {}",
              op->debugName(),
              input.first,
              apparentIteration);
        }
      }
      op->setup();
    }
  }
  loopOp->setup();

  for (auto afterTensor : afterLoopTensorIterMap) {
    logging::transform::trace(
        "[DecomposeLoops] After loop tensor {} -> {}, iteration {}",
        afterTensor.first.first,
        afterTensor.second,
        afterTensor.first.second);
  }

  // 3.) Hook up Ops after the loop
  for (int j = 0; j < unrollFactor; ++j) {
    for (size_t i = 0; i < schedule.size(); ++i) {
      Op *op = schedule.at(i);
      if (!isBeforeLoop(op, j)) {
        // Outputs
        auto apparentIteration = getApparentIteration(op, j);
        for (auto &output : outputMaps[op]) {
          if (apparentIteration == 2 && output.second->isGraphOutput()) {
            OutIndex sgOutIndex = output.second->getGraphOutputIndex();
            OutIndex opOutIndex = loopOp->subgraphOutToOpOutIndex(sgOutIndex);
            if (loopOp->output->hasIndex(opOutIndex)) {
              // Remap final outputs and loop outputs
              Tensor *loopOutTensor = loopOp->output->tensor(opOutIndex);
              TensorId newLoopOutTensorId =
                  ir.createIntermediateTensorId(loopOutTensor->id);
              loopOp->disconnectOutTensor(loopOutTensor);
              loopOp->createAndConnectOutTensor(opOutIndex, newLoopOutTensorId);
              loopOp->setup();
              clones[op][j]->connectOutTensor(output.first, loopOutTensor->id);
              for (auto &after : afterLoopTensorIterMap) {
                if (after.second == loopOutTensor->id) {
                  after.second = newLoopOutTensorId;
                }
              }
              afterLoopTensorIterMap[{output.second->id, apparentIteration}] =
                  loopOutTensor->id;
            }
          } else {
            TensorId outTensorId =
                removeScope(op->getGraph(), output.second->id);
            TensorId newOutTensorId =
                addScope(graph, ir.createIntermediateTensorId(outTensorId));
            clones[op][j]->createAndConnectOutTensor(output.first,
                                                     newOutTensorId);
            afterLoopTensorIterMap[{output.second->id, apparentIteration}] =
                newOutTensorId;
            if (output.second->isGraphOutput()) {
              afterLoopTensorIterMap[{output.second->id,
                                      apparentIteration + 1}] = newOutTensorId;
            }
          }
        }
      }
    }
  }

  for (int j = 0; j < unrollFactor; ++j) {
    for (size_t i = 0; i < schedule.size(); ++i) {
      Op *op = schedule.at(i);
      if (!isBeforeLoop(op, j)) {
        // Inputs
        auto apparentIteration = getApparentIteration(op, j);
        for (auto &input : inputMaps[op]) {
          if (input.second->isImplicitLoopInput()) {
            // Directly connect to the inputs connected to the LoopOp
            InIndex sgInIndex = input.second->getGraphInputIndex();
            InIndex opInIndex = loopOp->subgraphInToOpInIndex(sgInIndex);
            if (loopOp->hasInput(opInIndex)) {
              clones[op][j]->connectInTensor(input.first,
                                             loopOp->inTensor(opInIndex)->id);
            } else {
              throw error("[DecomposeLoops] LoopOp internally produced "
                          "tensors ({}) cannot be unrolled.",
                          input.second->id);
            }
          } else if (input.second->tensorType() == TensorType::Const) {
            TensorId newConstId;
            if (input.second->id.find(reservedConstValuePrefix()) !=
                std::string::npos) {
              newConstId = removeScope(op->getGraph(), input.second->id);
            } else {
              newConstId = ir.createIntermediateTensorId(
                  removeScope(op->getGraph(), input.second->id));
            }
            newConstId = addScope(graph, newConstId);
            if (!graph.getTensors().getConstIds().contains(newConstId)) {
              graph.getTensors().addConstInit(
                  newConstId,
                  input.second->info,
                  input.second->tensorData()->data());
            }
            clones[op][j]->connectInTensor(input.first, newConstId);
          } else {
            TensorId inTensorId = afterLoopTensorIterMap.at(
                {input.second->id, apparentIteration});
            clones[op][j]->connectInTensor(input.first, inTensorId);
          }
        }
        clones[op][j]->setup();
      }
    }
  }

  // Add and remove topocons
  std::map<int, std::vector<Op *>> beforeLoopBins;
  std::map<int, std::vector<Op *>> insideLoopBins;
  std::map<int, std::vector<Op *>> afterLoopBins;

  std::map<int, std::vector<Op *>> apparentIterationMap;

  for (size_t i = 0; i < schedule.size(); ++i) {
    Op *op                 = schedule.at(i);
    auto apparentIteration = getApparentIteration(op, -1);
    apparentIterationMap[apparentIteration].push_back(op);
    auto pos = model.typeToPosition(opToType.at(op), apparentIteration);
    insideLoopBins[pos].push_back(op);
    for (int j = 0; j < unrollFactor; ++j) {
      Op *cloneOp            = clones[schedule.at(i)][j];
      auto apparentIteration = getApparentIteration(op, j);
      apparentIterationMap[apparentIteration].push_back(op);
      auto pos = model.typeToPosition(opToType.at(op), apparentIteration);
      if (isBeforeLoop(op, j)) {
        beforeLoopBins[pos].push_back(cloneOp);
      } else {
        afterLoopBins[pos].push_back(cloneOp);
      }
    }
  }

  // Log Op bins
  auto logBins = [](std::string name, std::map<int, std::vector<Op *>> bins) {
    std::stringstream ss;
    for (auto bin : bins) {
      ss << std::endl;
      std::vector<std::string> names;
      names.reserve(bin.second.size());
      for (auto op : bin.second) {
        names.push_back(op->debugName());
      }
      ss << "    " << bin.first << ": ";
      ss << logging::join(names.begin(), names.end(), ", ");
    }
    logging::trace("[DecomposeLoops] {} bins: {}", name, ss.str());
  };

  if (logging::shouldLog(logging::Module::transform, logging::Level::Trace)) {
    logBins("before loop", beforeLoopBins);
    logBins("inside loop", insideLoopBins);
    logBins("after loop", afterLoopBins);
  }

  // Remove any topocons spanning multiple apparent iterations (see
  // DecomposeLoopOpType enum), since these are now invalid and may block
  // overlap
  for (auto &iteration0 : apparentIterationMap) {
    for (auto &iteration1 : apparentIterationMap) {
      if (iteration0.first != iteration1.first) {
        for (auto op0 : iteration0.second) {
          for (auto op1 : iteration1.second) {
            auto &topo = op0->getGraph().topoCons;
            topo->remove(op0, op1);
          }
        }
      }
    }
  }

  if (model.getTopoConLevelBefore() == DecomposeTopoConLevel::Full) {
    // 1.) Constraints before the loop
    for (auto bin0 : beforeLoopBins) {
      for (Op *before : bin0.second) {
        addTopoCon(graph, before, loopOp, false);
      }
      for (auto bin1 : beforeLoopBins) {
        if (bin0.first < bin1.first) {
          for (Op *before : bin0.second) {
            for (Op *after : bin1.second) {
              addTopoCon(graph, before, after, false);
            }
          }
        }
      }
    }
  }

  if (model.getTopoConLevelLoop() == DecomposeTopoConLevel::Full) {
    // 2.) Constraints inside the loop
    for (auto bin0 : insideLoopBins) {
      for (auto bin1 : insideLoopBins) {
        if (bin0.first < bin1.first) {
          for (Op *before : bin0.second) {
            for (Op *after : bin1.second) {
              addTopoCon(loopOp->getCalledGraph(), before, after, false);
            }
          }
        }
      }
    }
  }

  if (model.getTopoConLevelAfter() == DecomposeTopoConLevel::Full) {
    // 3.) Constraints after the loop
    for (auto bin0 : afterLoopBins) {
      for (Op *before : bin0.second) {
        addTopoCon(graph, loopOp, before, false);
      }
      for (auto bin1 : afterLoopBins) {
        if (bin0.first < bin1.first) {
          for (Op *before : bin0.second) {
            for (Op *after : bin1.second) {
              addTopoCon(graph, before, after, false);
            }
          }
        }
      }
    }
  }

  // Update trip count
  loopOp->setTripCountValue(loopOp->getTripCountValue() - unrollFactor);
  loopOp->setup();

  // Log the result
  logging::transform::trace("[DecomposeLoops] Summary; Decomposed LoopOp: {}",
                            loopOp->debugName());

  for (int j = 0; j < unrollFactor; ++j) {
    for (size_t i = 0; i < schedule.size(); ++i) {
      Op *op = schedule.at(i);
      if (isBeforeLoop(op, j)) {
        logging::transform::trace(
            "[DecomposeLoops] Summary; Op before loop: {}",
            clones[op][j]->debugName());
      }
    }
  }
  for (size_t i = 0; i < schedule.size(); ++i) {
    Op *op = schedule.at(i);
    logging::transform::trace("[DecomposeLoops] Summary; Op inside loop: {}",
                              op->debugName());
  }
  for (int j = 0; j < unrollFactor; ++j) {
    for (size_t i = 0; i < schedule.size(); ++i) {
      Op *op = schedule.at(i);
      if (!isBeforeLoop(op, j)) {
        logging::transform::trace("[DecomposeLoops] Summary; Op after loop: {}",
                                  clones[op][j]->debugName());
      }
    }
  }
}

bool DecomposeLoops::apply(Graph &graph) const {
  auto schedule = graph.getOpSchedule({}, RequireOptimalSchedule::No);

  // Build sets of mergeable loops
  for (size_t i = 0; i < schedule.size(); ++i) {
    Op *op = schedule.at(i);
    if (LoopOp *loopOp = dynamic_cast<LoopOp *>(op)) {
      decomposeLoop(graph, loopOp, DecomposeLoopOverlapModel());
    }
  }

  return true;
}

namespace {
bool init = Transform::registerTransform(new DecomposeLoops);
}

} // namespace popart
