// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <queue>
#include <popart/aliasesmap.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/graphutils.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/exchange/hostcopy.hpp>
#include <popart/op/exchange/multiexchange.hpp>
#include <popart/op/exchange/remote.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/init.hpp>
#include <popart/op/iotilecopy.hpp>
#include <popart/op/loop.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/decomposeloops.hpp>
#include <popart/util.hpp>

namespace popart {

std::ostream &operator<<(std::ostream &os,
                         const DecomposeLoopOpTypeWrapper &dlopt) {
  os << *(dlopt.getType<DecomposeLoopOpType>());
  return os;
}

std::ostream &operator<<(std::ostream &os, const DecomposeLoopOpType &dlopt) {
  dlopt.output(os);
  return os;
}

std::ostream &operator<<(std::ostream &os,
                         const DecomposeLoopOpTypeEnum &dlopt) {
  switch (dlopt) {
  case DecomposeLoopOpTypeEnum::AuxiliaryBefore:
    os << "AuxiliaryBefore";
    break;
  case DecomposeLoopOpTypeEnum::IoBeforeCompute:
    os << "IoBeforeCompute";
    break;
  case DecomposeLoopOpTypeEnum::IoToCompute:
    os << "IoToCompute";
    break;
  case DecomposeLoopOpTypeEnum::Compute:
    os << "Compute";
    break;
  case DecomposeLoopOpTypeEnum::ComputeToIo:
    os << "ComputeToIo";
    break;
  case DecomposeLoopOpTypeEnum::IoAfterCompute:
    os << "IoAfterCompute";
    break;
  case DecomposeLoopOpTypeEnum::AuxiliaryAfter:
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

DecomposeLoopOpIOOverlapType::DecomposeLoopOpIOOverlapType(
    DecomposeLoopOpTypeEnum type_)
    : DecomposeLoopOpType(), type(type_) {}

bool DecomposeLoopOpIOOverlapType::
operator<(const DecomposeLoopOpType &other) const {
  if (auto castOther =
          dynamic_cast<const DecomposeLoopOpIOOverlapType *>(&other)) {
    return this->getType() < castOther->getType();
  }
  return false;
}

bool DecomposeLoopOpIOOverlapType::
operator==(const DecomposeLoopOpType &other) const {
  if (auto castOther =
          dynamic_cast<const DecomposeLoopOpIOOverlapType *>(&other)) {
    return this->getType() == castOther->getType();
  }
  return false;
}

bool DecomposeLoopOpIOOverlapType::
operator!=(const DecomposeLoopOpType &other) const {
  if (auto castOther =
          dynamic_cast<const DecomposeLoopOpIOOverlapType *>(&other)) {
    return this->getType() != castOther->getType();
  }
  return true;
}

std::ostream &DecomposeLoopOpIOOverlapType::output(std::ostream &os) const {
  os << type;
  return os;
}

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

int DecomposeLoopModel::getTypeGroup(DecomposeLoopOpTypeWrapper type) const {
  return 0;
}

bool DecomposeLoopModel::hasDependencyConflict(
    LoopIteration iterA,
    LoopIteration iterB,
    DecomposeLoopOpTypeWrapper typeA,
    DecomposeLoopOpTypeWrapper typeB) const {
  return typeToPosition(typeA, iterA) > typeToPosition(typeB, iterB);
}

std::string DecomposeLoopModel::getModelString() {
  auto types = getDecomposeLoopOpTypesToCheck();

  std::map<DecomposeLoopOpTypeWrapper, std::set<std::pair<int, int>>> positions;
  std::map<DecomposeLoopOpTypeWrapper, std::string> stringRepresentations;
  std::set<std::tuple<int, int, DecomposeLoopOpTypeWrapper>> orderedTypes;

  int maxRepresentationLength = 0;
  int minPos                  = std::numeric_limits<int>::max();
  int maxPos                  = std::numeric_limits<int>::min();

  // Insert vertical bars when the apparent iteration changes
  std::map<LoopIteration, int> verticalBars;

  for (auto &type : types) {
    std::stringstream ss;
    ss << type;
    std::string stringRepresentation = ss.str();
    stringRepresentations[type]      = stringRepresentation;
    maxRepresentationLength =
        std::max(maxRepresentationLength,
                 static_cast<int>(stringRepresentation.length()));

    for (int unrollIndex = -1; unrollIndex < getUnrollFactor(); ++unrollIndex) {
      int barOffset = 0;
      if (!isBeforeLoop(type, unrollIndex)) {
        if (unrollIndex == -1) {
          barOffset = 1;
        } else {
          barOffset = 2;
        }
      }

      auto ai  = getApparentIteration(type, unrollIndex);
      auto pos = typeToPosition(type, ai) + barOffset;
      positions[type].insert({pos, ai});

      minPos = std::min(minPos, pos);
      maxPos = std::max(maxPos, pos);

      verticalBars[barOffset] =
          verticalBars.find(barOffset) == verticalBars.end()
              ? pos - 1
              : std::min(verticalBars[barOffset], pos - 1);
    }
    orderedTypes.insert(
        {getTypeGroup(type), positions.at(type).begin()->first, type});
  }

  // Template
  std::string dots(maxPos - minPos + 1, '.');

  // Create line to separate groups visually
  std::string groupSeparator;
  {
    std::stringstream ss;
    std::string spaces(maxRepresentationLength + 5, ' ');
    std::string lines(maxPos - minPos + 1, '-');
    ss << spaces << lines << "\n";
    groupSeparator = ss.str();
  }

  for (auto &aiBar : verticalBars) {
    if (aiBar.first > 0) {
      dots.replace(aiBar.second - minPos, 1, "|");
      groupSeparator.replace(
          maxRepresentationLength + 5 + aiBar.second - minPos, 1, "+");
    }
  }

  std::stringstream ss;

  auto lastGroup = std::get<0>(*orderedTypes.begin());
  for (auto &type : orderedTypes) {
    auto currentGroup = std::get<0>(type);

    // Separate groups visually
    if (currentGroup != lastGroup) {
      lastGroup = currentGroup;
      ss << groupSeparator;
    }

    auto typeDots = dots;
    for (auto &pos : positions.at(std::get<2>(type))) {
      std::string posString =
          std::to_string(std::max(0, std::min(9, pos.second)));
      auto position = pos.first - minPos;
      auto length   = posString.size();
      typeDots.replace(position, length, posString);
    }

    auto &stringRep = stringRepresentations.at(std::get<2>(type));
    stringRep.insert(
        stringRep.end(), maxRepresentationLength - stringRep.size() + 5, ' ');
    ss << stringRep << typeDots << std::endl;
  }

  return ss.str();
}

std::map<Op *, DecomposeLoopOpTypeWrapper, POpCmp>
DecomposeLoopModel::classifyOperations(Graph &subgraph) const {
  auto schedule = subgraph.getOpSchedule({}, RequireOptimalSchedule::No);

  std::map<Op *, DecomposeLoopOpTypeWrapper, POpCmp> opToDecomposeLoopOpType;

  // Settings for getting the Op type
  bool allowSeeding  = false;
  bool allowDelaying = false;

  // Classify operations
  bool shouldContinue                  = true;
  int changedCount                     = 0;
  int opToDecomposeLoopOpTypeIteration = 0;

  // Loop until no classification changes
  while (opToDecomposeLoopOpType.size() < schedule.size() || shouldContinue) {

    // Copy the previous state
    const auto lastOpToDecomposeLoopOpType = opToDecomposeLoopOpType;

    shouldContinue = false;

    if (changedCount == 0 && opToDecomposeLoopOpTypeIteration > 0) {
      if (!allowSeeding && !allowDelaying) {
        // First, if there is no change anymore, we allow seeding
        allowSeeding   = true;
        shouldContinue = true;
      } else if (allowSeeding && !allowDelaying) {
        // Second, if there is no change anymore, we also allow delaying
        allowDelaying  = true;
        shouldContinue = true;
      }
    }

    changedCount = 0;

    // Loop through all operations in the schedule
    for (size_t i = 0; i < schedule.size(); ++i) {
      bool changed = false;
      Op *op       = schedule.at(i);

      auto oldType = lastOpToDecomposeLoopOpType.find(op);
      auto newType = getDecomposeLoopOpType(
          opToDecomposeLoopOpType, op, allowSeeding, allowDelaying);

      if (newType.hasValue() && (oldType == lastOpToDecomposeLoopOpType.end() ||
                                 oldType->second != newType)) {
        changed = true;
        ++changedCount;
      }

      std::stringstream newTypeSS;
      if (newType.hasValue()) {
        opToDecomposeLoopOpType[op] = newType;
        newTypeSS << newType;
      } else {
        auto typeIt = opToDecomposeLoopOpType.find(op);
        if (typeIt != opToDecomposeLoopOpType.end()) {
          newTypeSS << typeIt->second;
        } else {
          newTypeSS << "not set";
        }
      }

      logging::transform::trace(
          "[DecomposeLoopModel::classifyOperations] "
          "Classifying operation {} type {} (changed: {})",
          op->debugName(),
          newTypeSS.str(),
          changed);
    }

    logging::transform::trace(
        "[DecomposeLoopModel::classifyOperations] Classifying operations, "
        "iteration {}, changed {}",
        opToDecomposeLoopOpTypeIteration,
        changedCount);

    ++opToDecomposeLoopOpTypeIteration;
    shouldContinue |= (changedCount > 0);
  }

  for (auto opWithType : opToDecomposeLoopOpType) {
    // Log
    logging::transform::trace("[DecomposeLoopHelper::classifyOperations] Final "
                              "Op {} type {} tile set {}",
                              opWithType.first->debugName(),
                              opWithType.second,
                              opWithType.first->settings.tileSet);
  }

  return opToDecomposeLoopOpType;
}

DecomposeLoopIOModel::DecomposeLoopIOModel(
    DecomposeTopoConLevel topoConLevelBefore_,
    DecomposeTopoConLevel topoConLevelLoop_,
    DecomposeTopoConLevel topoConLevelAfter_,
    const std::set<ExchangeStrategy> &computeLikeExchangeStrategies_)
    : DecomposeLoopModel(topoConLevelBefore_,
                         topoConLevelLoop_,
                         topoConLevelAfter_,
                         computeLikeExchangeStrategies_) {}

std::set<DecomposeLoopOpTypeWrapper>
DecomposeLoopIOModel::getDecomposeLoopOpTypesToCheck() const {
  std::set<DecomposeLoopOpTypeWrapper> typeSet{
      DecomposeLoopOpIOOverlapType{DecomposeLoopOpTypeEnum::AuxiliaryBefore},
      DecomposeLoopOpIOOverlapType{DecomposeLoopOpTypeEnum::IoBeforeCompute},
      DecomposeLoopOpIOOverlapType{DecomposeLoopOpTypeEnum::IoToCompute},
      DecomposeLoopOpIOOverlapType{DecomposeLoopOpTypeEnum::Compute},
      DecomposeLoopOpIOOverlapType{DecomposeLoopOpTypeEnum::ComputeToIo},
      DecomposeLoopOpIOOverlapType{DecomposeLoopOpTypeEnum::IoAfterCompute},
      DecomposeLoopOpIOOverlapType{DecomposeLoopOpTypeEnum::AuxiliaryAfter}};

  return typeSet;
}

DecomposeLoopUnrollModel::DecomposeLoopUnrollModel()
    : DecomposeLoopIOModel(), unrollBefore(0), unrollAfter(0) {}

DecomposeLoopUnrollModel::DecomposeLoopUnrollModel(
    int unrollBefore_,
    int unrollAfter_,
    DecomposeTopoConLevel topoConLevelBefore_,
    DecomposeTopoConLevel topoConLevelLoop_,
    DecomposeTopoConLevel topoConLevelAfter_,
    const std::set<ExchangeStrategy> &computeLikeExchangeStrategies_)
    : DecomposeLoopIOModel(topoConLevelBefore_,
                           topoConLevelLoop_,
                           topoConLevelAfter_,
                           computeLikeExchangeStrategies_),
      unrollBefore(unrollBefore_), unrollAfter(unrollAfter_) {}

int DecomposeLoopUnrollModel::typeToPosition(DecomposeLoopOpTypeWrapper type,
                                             LoopIteration iteration) const {
  return static_cast<int>(unwrap(type)) +
         iteration *
             (static_cast<int>(DecomposeLoopOpTypeEnum::N) -
              static_cast<int>(DecomposeLoopOpTypeEnum::AuxiliaryBefore));
}

LoopIteration
DecomposeLoopUnrollModel::getApparentIteration(DecomposeLoopOpTypeWrapper type,
                                               int unrollIndex) const {
  if (unrollIndex == -1) {
    return unrollBefore;
  } else if (unrollIndex < unrollBefore) {
    return unrollIndex;
  } else if (unrollIndex < (unrollBefore + unrollAfter)) {
    return unrollIndex + 1;
  } else {
    throw error("[DecomposeLoopUnrollModel::getApparentIteration] Unexpected "
                "unrollIndex {}",
                unrollIndex);
  }
}

bool DecomposeLoopUnrollModel::isBeforeLoop(DecomposeLoopOpTypeWrapper type,
                                            int unrollIndex) const {
  return unrollIndex < unrollBefore;
}

DecomposeLoopOverlapModel::DecomposeLoopOverlapModel() {}

DecomposeLoopOverlapModel::DecomposeLoopOverlapModel(
    DecomposeTopoConLevel topoConLevelBefore_,
    DecomposeTopoConLevel topoConLevelLoop_,
    DecomposeTopoConLevel topoConLevelAfter_,
    const std::set<ExchangeStrategy> &computeLikeExchangeStrategies_)
    : DecomposeLoopIOModel(topoConLevelBefore_,
                           topoConLevelLoop_,
                           topoConLevelAfter_,
                           computeLikeExchangeStrategies_) {}

int DecomposeLoopOverlapModel::typeToPosition(DecomposeLoopOpTypeWrapper type,
                                              LoopIteration iteration) const {
  auto utype = unwrap(type);

  // Lookup table numbers are derived from the ASCII chart
  // (see DecomposeLoopOpTypeEnum) to place the operations in a pattern
  // that enables overlapped IO and compute. The ASCII representation can
  // be verified by getModelString(), which turns these numbers back into
  // an ASCII chart.
  std::vector<int> lookup{0,  3,  8,   // AuxiliaryBefore
                          1,  4,  10,  // IoBeforeCompute
                          2,  6,  13,  // IoToCompute
                          5,  11, 16,  // Compute
                          7,  14, 18,  // ComputeToIo
                          9,  15, 19,  // IoAfterCompute
                          12, 17, 20}; // AuxiliaryAfter
  int numApparentIterations = getUnrollFactor() + 1;
  return lookup.at(static_cast<int>(utype) * numApparentIterations + iteration);
}

LoopIteration
DecomposeLoopOverlapModel::getApparentIteration(DecomposeLoopOpTypeWrapper type,
                                                int unrollIndex) const {
  auto utype = unwrap(type);

  if (unrollIndex == -1) {
    if (utype >= DecomposeLoopOpTypeEnum::IoAfterCompute) {
      return 0;
    }
    if (utype >= DecomposeLoopOpTypeEnum::Compute) {
      return 1;
    }
    if (utype >= DecomposeLoopOpTypeEnum::AuxiliaryBefore) {
      return 2;
    }
  } else if (unrollIndex == 0) {
    if (utype >= DecomposeLoopOpTypeEnum::IoAfterCompute) {
      return 1;
    } else {
      return 0;
    }
  } else if (unrollIndex == 1) {
    if (utype >= DecomposeLoopOpTypeEnum::Compute) {
      return 2;
    } else {
      return 1;
    }
  }
  return -1;
}

bool DecomposeLoopOverlapModel::isBeforeLoop(DecomposeLoopOpTypeWrapper type,
                                             int unrollIndex) const {
  if (unrollIndex < 0) {
    return false;
  } else {
    auto utype = unwrap(type);
    if (utype <= DecomposeLoopOpTypeEnum::ComputeToIo && unrollIndex <= 0) {
      return true;
    }
    if (utype <= DecomposeLoopOpTypeEnum::IoToCompute && unrollIndex <= 1) {
      return true;
    }
  }
  return false;
}

// An Op that should be classified as Compute
bool DecomposeLoops::isComputeOp(Op *op) {
  return op->settings.tileSet == TileSet::Compute;
}

// An Op that is IO, and on IO tiles
bool DecomposeLoops::isIOOp(Op *op) {
  return op->isConvertibleTo<RemoteLoadOp>() ||
         op->isConvertibleTo<RemoteLoadInplaceOp>() ||
         op->isConvertibleTo<RemoteStoreOp>() ||
         op->isConvertibleTo<HostLoadOp>() ||
         op->isConvertibleTo<HostStoreOp>() ||
         op->isConvertibleTo<MultiExchangeOp>();
}

// An Op that is IO, and on IO tiles, but still to be classified as Compute
bool DecomposeLoops::isComputeLikeIOOp(
    std::set<ExchangeStrategy> computeLikeStrategies,
    Op *op) {
  auto &ir             = op->getIr();
  bool isComputeLikeIo = false;

  auto isOpComputeLike = [&ir, &computeLikeStrategies](Op *opToCheck) {
    if (auto hostLoadOp = dynamic_cast<HostLoadOp *>(opToCheck)) {
      auto exchangeStrategy = ir.getTensor(hostLoadOp->getHostStreamTensorId())
                                  ->inputSettings.exchangeStrategy();
      return (hostLoadOp->settings.tileSet == TileSet::IO &&
              computeLikeStrategies.find(exchangeStrategy) !=
                  computeLikeStrategies.end());
    }
    if (auto hostStoreOp = dynamic_cast<HostStoreOp *>(opToCheck)) {
      auto art = ir.getDataFlow().getAnchorReturnTypeMap().at(
          hostStoreOp->getHostStreamTensorId());
      auto exchangeStrategy = art.exchangeStrategy();
      return (hostStoreOp->settings.tileSet == TileSet::IO &&
              computeLikeStrategies.find(exchangeStrategy) !=
                  computeLikeStrategies.end());
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
               c->isConvertibleTo<RemoteLoadOp>() ||
               c->isConvertibleTo<RemoteLoadInplaceOp>() ||
               c->isConvertibleTo<RemoteStoreOp>() ||
               c->isConvertibleTo<InitOp>() ||
               c->isConvertibleTo<IoTileCopyOp>();
      },
      graphutils::TraversalType::DepthFirst,
      graphutils::VisitType::Pre,
      graphutils::TraversalDirection::Forward);

  graphutils::traverse(
      op->output->tensors(),
      [&isComputeLikeIo, &isOpComputeLike](Tensor *t) -> bool {
        if (t->hasProducer()) {
          isComputeLikeIo |= isOpComputeLike(t->getProducer());
        }
        return true;
      },
      [op](Op *c, Tensor *t0, Tensor *t1) -> bool {
        if (c->getGraph().id != op->getGraph().id) {
          return false;
        }
        return c->isConvertibleTo<HostLoadOp>() ||
               c->isConvertibleTo<HostStoreOp>() ||
               c->isConvertibleTo<RemoteLoadOp>() ||
               c->isConvertibleTo<RemoteLoadInplaceOp>() ||
               c->isConvertibleTo<RemoteStoreOp>() ||
               c->isConvertibleTo<InitOp>() ||
               c->isConvertibleTo<IoTileCopyOp>();
      },
      graphutils::TraversalType::DepthFirst,
      graphutils::VisitType::Pre,
      graphutils::TraversalDirection::Backward);

  return isComputeLikeIo;
}

bool DecomposeLoops::DecomposeLoopHelper::addTopoCon(Graph &graph,
                                                     Op *before,
                                                     Op *after,
                                                     bool tied) const {
  graph.topoCons->insert(before, after, tied);
  return true;
}

namespace {
/**
 * Helper class to organize functions required to classify operations
 */
class DecomposeLoopOpTypeIOOverlapHelper {
public:
  DecomposeLoopOpTypeIOOverlapHelper(const DecomposeLoopIOModel &model_,
                                     Op *op_,
                                     bool allowSeeding_,
                                     bool allowDelaying_)
      : model(model_), op(op_), tileSet(op->settings.tileSet),
        allowSeeding(allowSeeding_), allowDelaying(allowDelaying_) {

    isIoOp           = DecomposeLoops::isIOOp(op);
    isIoTileCopy     = op->isConvertibleTo<IoTileCopyOp>();
    isOnIOTiles      = tileSet == TileSet::IO;
    isOnComputeTiles = tileSet == TileSet::Compute;
    computeLike =
        (isOnIOTiles && DecomposeLoops::isComputeLikeIOOp(
                            model_.getComputeLikeExchangeStrategies(), op)) ||
        isOnComputeTiles;
  }

  /**
   * Register any type of operation occuring before the current operation.
   * \param opToDecomposeLoopOpType Existing Op to type mapping.
   * \param bop                     Operation, which occurs before the Op
   *                                associated with this class, to register.
   */
  void registerBefore(const std::map<Op *, DecomposeLoopOpTypeWrapper, POpCmp>
                          &opToDecomposeLoopOpType,
                      Op *bop) {
    auto it = opToDecomposeLoopOpType.find(bop);
    if (it != opToDecomposeLoopOpType.end()) {
      beforeTypes.insert(model.unwrap(it->second));
    }
  }

  /**
   * Register any type of operation occuring after the current operation.
   * \param opToDecomposeLoopOpType Existing Op to type mapping.
   * \param aop                     Operation, which occurs after the Op
   *                                associated with this class, to register.
   */
  void registerAfter(const std::map<Op *, DecomposeLoopOpTypeWrapper, POpCmp>
                         &opToDecomposeLoopOpType,
                     Op *aop) {
    auto it = opToDecomposeLoopOpType.find(aop);
    if (it != opToDecomposeLoopOpType.end()) {
      afterTypes.insert(model.unwrap(it->second));
    }
  }

  /**
   * Register all relations the current Op has to operations occuring `before`
   * and `after` in the schedule.
   * \param opToDecomposeLoopOpType Existing Op to type mapping.
   */
  void
  registerRelations(const std::map<Op *, DecomposeLoopOpTypeWrapper, POpCmp>
                        &opToDecomposeLoopOpType) {
    for (auto &input : op->input->tensorMap()) {
      if (input.second->hasProducer()) {
        Op *producer = input.second->getProducer();
        registerBefore(opToDecomposeLoopOpType, producer);
      }
    }

    for (auto &output : op->output->tensorMap()) {
      for (Op *consumer : output.second->consumers.getOps()) {
        registerAfter(opToDecomposeLoopOpType, consumer);
      }
    }

    auto befores = op->getGraph().topoCons->getBefores(op);
    for (Op *before : befores) {
      registerBefore(opToDecomposeLoopOpType, before);
    }

    auto afters = op->getGraph().topoCons->getAfters(op);
    for (Op *after : afters) {
      registerAfter(opToDecomposeLoopOpType, after);
    }
  }

  /**
   * Check if any s1 is in s0.
   * \param s0 Set to check for occureces in s1.
   * \param s1 Set to check occurences of s0 in.
   * \return   True if any element of s1 is in s0.
   */
  bool anyTypes(std::set<DecomposeLoopOpIOOverlapType> s0,
                std::set<DecomposeLoopOpIOOverlapType> s1) {
    return std::any_of(
        s1.begin(), s1.end(), [&s0](const DecomposeLoopOpIOOverlapType &t1) {
          return std::find(s0.begin(), s0.end(), t1) != s0.end();
        });
  }

  /**
   * Return if any types in s1 occur before the current Op.
   * \param s1 Set of types to check.
   * \return   True if any types in s1 occur in beforeTypes, false otherwise.
   */
  bool anyBefore(std::set<DecomposeLoopOpIOOverlapType> s1) {
    return anyTypes(beforeTypes, s1);
  }

  /**
   * Return if any types in s1 occur after the current Op.
   * \param s1 Set of types to check.
   * \return   True if any types in s1 occur in afterTypes, false otherwise.
   */
  bool anyAfter(std::set<DecomposeLoopOpIOOverlapType> s1) {
    return anyTypes(afterTypes, s1);
  }

  DecomposeLoopOpTypeWrapper ioTileCopyOpRule() {
    if (isIoTileCopy) {
      if (isOnComputeTiles) {
        if (anyBefore({computeToIO, ioAfter, auxAfter})) {
          return auxAfter;
        }
        if (anyBefore({compute})) {
          if (computeLike) {
            return compute;
          }
        }
        return ioToCompute;
      } else {
        if (computeLike) {
          return compute;
        }
        return computeToIO;
      }
    }
    return {};
  }

  DecomposeLoopOpTypeWrapper opsBeforeEverythingRule() {
    if ((isOnComputeTiles && anyAfter({ioBefore})) || anyAfter({auxBefore})) {
      return auxBefore;
    }
    return {};
  }

  DecomposeLoopOpTypeWrapper opsAfterEverythingRule() {
    if ((isOnComputeTiles && anyBefore({ioAfter})) || anyBefore({auxAfter})) {
      return auxAfter;
    }
    return {};
  }

  DecomposeLoopOpTypeWrapper ioOperationRule() {
    if (isIoOp) {
      if (computeLike) {
        return compute;
      } else {
        if (anyBefore({compute, ioToCompute, computeToIO}) ||
            (allowDelaying &&
             !anyAfter(
                 {auxBefore, ioBefore, ioToCompute, compute, computeToIO}))) {
          return ioAfter;
        } else {
          return ioBefore;
        }
      }
    }
    return {};
  }

  DecomposeLoopOpTypeWrapper computeOperationRule() {
    if (isOnComputeTiles && !isIoTileCopy && !isIoOp &&
        anyBefore({ioToCompute, ioBefore, compute})) {
      return compute;
    }
    return {};
  }

  DecomposeLoopOpTypeWrapper seedRule() {
    if (allowSeeding) {
      if (isOnIOTiles) {
        if (!afterTypes.empty()) {
          auto type = *afterTypes.begin();
          if (type == ioBefore) {
            // Ensuring we do not block IO merging by pushing the Op earlier
            type = auxBefore;
          }
          return type;
        }
        if (!beforeTypes.empty()) {
          auto type = *beforeTypes.rbegin();
          if (type == ioAfter) {
            // Ensuring we do not block IO merging by pushing the Op later
            type = auxAfter;
          }
          return type;
        }
        if (allowDelaying) {
          // If there is no dependency, run as earlier as possible
          return auxBefore;
        }
      }
      if (isOnComputeTiles) {
        return compute;
      }
    }
    return {};
  }

  const DecomposeLoopIOModel &model;

  Op *op;
  TileSet tileSet;

  bool allowSeeding;
  bool allowDelaying;

  std::set<DecomposeLoopOpIOOverlapType> beforeTypes;
  std::set<DecomposeLoopOpIOOverlapType> afterTypes;

  // Op properties
  bool isIoOp;
  bool isIoTileCopy;
  bool isOnIOTiles;
  bool isOnComputeTiles;
  bool computeLike;

  // Short-hands of valid types
  DecomposeLoopOpIOOverlapType auxBefore{
      DecomposeLoopOpTypeEnum::AuxiliaryBefore};
  DecomposeLoopOpIOOverlapType ioBefore{
      DecomposeLoopOpTypeEnum::IoBeforeCompute};
  DecomposeLoopOpIOOverlapType ioToCompute{
      DecomposeLoopOpTypeEnum::IoToCompute};
  DecomposeLoopOpIOOverlapType compute{DecomposeLoopOpTypeEnum::Compute};
  DecomposeLoopOpIOOverlapType computeToIO{
      DecomposeLoopOpTypeEnum::ComputeToIo};
  DecomposeLoopOpIOOverlapType ioAfter{DecomposeLoopOpTypeEnum::IoAfterCompute};
  DecomposeLoopOpIOOverlapType auxAfter{
      DecomposeLoopOpTypeEnum::AuxiliaryAfter};
};
} // namespace

DecomposeLoopOpTypeWrapper DecomposeLoopIOModel::getDecomposeLoopOpType(
    const std::map<Op *, DecomposeLoopOpTypeWrapper, POpCmp>
        &opToDecomposeLoopOpType,
    Op *op,
    bool allowSeeding,
    bool allowDelaying) const {

  DecomposeLoopOpTypeIOOverlapHelper helper(
      *this, op, allowSeeding, allowDelaying);

  // Register relations of the current Op to any operations before and after
  helper.registerRelations(opToDecomposeLoopOpType);

  // Classification rules:

  // 1.) IoTileCopyOp copies from or to IO tiles
  auto type = helper.ioTileCopyOpRule();
  if (type != DecomposeLoopOpTypeWrapper()) {
    return type;
  }

  // 2.) Ops that run before everything
  type = helper.opsBeforeEverythingRule();
  if (type != DecomposeLoopOpTypeWrapper()) {
    return type;
  }

  // 3.) Ops that run after everything
  type = helper.opsAfterEverythingRule();
  if (type != DecomposeLoopOpTypeWrapper()) {
    return type;
  }

  // 4.) IO operations
  type = helper.ioOperationRule();
  if (type != DecomposeLoopOpTypeWrapper()) {
    return type;
  }

  // 5.) Compute operations
  type = helper.computeOperationRule();
  if (type != DecomposeLoopOpTypeWrapper()) {
    return type;
  }

  // 6.) Seed operations that didn't naturally get a type assigned
  type = helper.seedRule();
  if (type != DecomposeLoopOpTypeWrapper()) {
    return type;
  }

  // Empty default
  return type;
}

void DecomposeLoops::decomposeLoop(Graph &graph,
                                   LoopOp *loopOp,
                                   const DecomposeLoopModel &model) const {
  auto &ir        = graph.getIr();
  Graph &subgraph = loopOp->getCalledGraph();

  if (loopOp->getTripCountValue() < model.getUnrollFactor()) {
    logging::warn("[DecomposeLoops::decomposeLoop] Attempted to unroll {} "
                  "iterations of LoopOp {}, but the trip count is only {}.",
                  model.getUnrollFactor(),
                  loopOp->debugName(),
                  loopOp->getTripCountValue());
    return;
  }

  DecomposeLoopHelper helper(ir, graph, model, loopOp, subgraph);

  helper.adjustLoop();
  helper.createBackupStructure();
  helper.prepare();
  helper.clone();
  helper.hookUpBeforeLoop();
  helper.hookUpInLoop();
  helper.hookUpAfterLoop();
  helper.fixTopoCons();
  helper.updateLoop();
  helper.cleanup();
}

DecomposeLoops::DecomposeLoopHelper::DecomposeLoopHelper(
    Ir &ir_,
    Graph &graph_,
    const DecomposeLoopModel &model_,
    LoopOp *loopOp,
    Graph &subgraph_)
    : ir(ir_), graph(graph_), model(model_), loopOp(loopOp),
      subgraph(subgraph_), backupGraphId("___tmp___" + subgraph.id.str()),
      backupLoopOp(nullptr) {

  logging::transform::trace(
      "[DecomposeLoopHelper] Decomposing {} with model {} "
      "(compute like IO strategies: {})",
      loopOp->debugName(),
      model,
      model.getComputeLikeExchangeStrategies());

  // Check modified input tensors
  for (auto &input : loopOp->input->tensorMap()) {
    auto modifies = loopOp->modifiesIndex(input.first);
    auto sgTensor =
        loopOp->getCalledGraph().getTensor(loopOp->getCalledGraph().getInputId(
            loopOp->opInToSubgraphInIndex(input.first)));
    if (!modifies && sgTensor->isModified(false)) {
      // TODO T56806: Turn warning into error once all early inplace
      // optimisations have been removed and are left to the inplacing
      // algorithm.
      logging::warn(
          "[DecomposeLoops::DecomposeLoopHelper] Tensor {} is being inplace "
          "modified in the subgraph, but the LoopOp does not promote that "
          "modification. DecomposeLoop cannot safely decompose loops that "
          "contain inplace operations for optimisation purposes.",
          sgTensor->id);
    }
  }
}

void DecomposeLoops::DecomposeLoopHelper::adjustLoop() {
  // Find problematic cases and insert IdentityOps
  auto &calledGraph   = loopOp->getCalledGraph();
  auto &ops           = calledGraph.getOps();
  bool opsHavePStages = std::any_of(ops.begin(), ops.end(), [](const auto &it) {
    return it.second->hasPipelineStage();
  });

  OptionalPipelineStage maxPipelineStage;

  if (loopOp->getIr().getSessionOptions().explicitPipeliningEnabled() &&
      opsHavePStages) {
    maxPipelineStage = loopOp->getIr().getNumPipelineStages() - 1;
  }

  for (auto &outputId : calledGraph.getOutputIds()) {
    auto output = calledGraph.getTensor(outputId);
    if ((output->isLoopInput() &&
         output->getGraphInputIndex() >= LoopOp::getFirstInputInIndex()) ||
        output->tensorType() == TensorType::Const) {
      // Separator identity required.
      // TODO T57045: Stop skipping input 0 and 1 (they are currently
      // unsupported in loop decomposition)

      auto intermediateTensorId =
          calledGraph.getIr().createIntermediateTensorId(outputId);

      Op::Settings settings(calledGraph, "LoopOutputSeparatorIdentityOp");

      auto vgidAndTileSet = output->getVirtualGraphIdAndTileSetUnsafe();

      if (vgidAndTileSet.first != unusedVGraphId) {
        settings.vgraphId = vgidAndTileSet.first;
      }
      settings.tileSet       = vgidAndTileSet.second;
      settings.pipelineStage = maxPipelineStage;

      IdentityInplaceOp *op = calledGraph.createConnectedOp<IdentityInplaceOp>(
          {{IdentityInplaceOp::getInIndex(), outputId}},
          {{IdentityInplaceOp::getOutIndex(), intermediateTensorId}},
          Onnx::CustomOperators::IdentityInplace,
          settings);

      logging::trace("[DecomposeLoopHelper::adjustLoop] Adding loop output "
                     "separator identity operator {}",
                     op->debugName());

      calledGraph.markAsOutput(
          calledGraph.getOutputIndex(outputId), intermediateTensorId, true);
    }
  }
}

void DecomposeLoops::DecomposeLoopHelper::createBackupStructure() {
  // Create a backup of the original subgraph for lookup purposes
  backupMaps = ir.cloneGraph(subgraph.id, backupGraphId);

  // Clone the operator
  auto backupLoopOpUp = loopOp->clone();

  // Change ownership of the cloned operator after obtaining the raw
  // pointer
  backupLoopOp = static_cast<LoopOp *>(backupLoopOpUp.get());
  graph.moveIntoGraph(std::move(backupLoopOpUp));
  backupLoopOp->setCalledGraph(getBackupGraph());

  backupMaps.opIdMap[loopOp->id]       = backupLoopOp->id;
  backupMaps.opIdMap[backupLoopOp->id] = loopOp->id;

  for (auto &input : loopOp->input->tensorMap()) {
    backupLoopOp->connectInTensor(input.first, input.second->id);
    backupMaps.tensorIdMap[input.second->id] = input.second->id;
  }

  for (auto &output : loopOp->output->tensorMap()) {
    TensorId backupOutputId = ir.createIntermediateTensorId(output.second->id);
    backupLoopOp->createAndConnectOutTensor(output.first, backupOutputId);
    backupMaps.tensorIdMap[output.second->id] = backupOutputId;
    backupMaps.tensorIdMap[backupOutputId]    = output.second->id;
  }

  backupLoopOp->setup();
}

void DecomposeLoops::DecomposeLoopHelper::prepare() {
  // Explicit inputs that are modified need to be re-assessed
  for (auto &input : loopOp->input->tensorMap()) {
    auto modifies = loopOp->modifiesIndex(input.first);
    if (modifies && input.first < loopOp->getNumExplicitInputs()) {
      loopOp->removeModified(input.first);
    }
  }

  schedule = subgraph.getOpSchedule({}, RequireOptimalSchedule::No);

  // Disconnect everything that is being reconnected later
  for (auto op : schedule) {
    op->disconnectAllInputs();
    op->disconnectAllOutputs();
  }

  auto numExplicitInputs = loopOp->getNumExplicitInputs();

  for (auto explicitInputIndex = numExplicitInputs - 1;
       explicitInputIndex >= LoopOp::getFirstInputInIndex();
       --explicitInputIndex) {
    loopOp->removeLoopInput(explicitInputIndex);
    loopOp->removeLoopOutput(explicitInputIndex -
                             LoopOp::getFirstInputInIndex() +
                             LoopOp::getFirstOutputOutIndex());
  }

  backupSchedule =
      getBackupGraph().getOpSchedule({}, RequireOptimalSchedule::No);

  opToDecomposeLoopOpType = model.classifyOperations(getBackupGraph());

  for (size_t i = 0; i < backupSchedule.size(); ++i) {
    Op *backupOp = backupSchedule.at(i);

    auto type = opToDecomposeLoopOpType[backupOp];
    opsByType[type].push_back(backupOp);
  }
}

void DecomposeLoops::DecomposeLoopHelper::clone() {
  int unrollFactor = model.getUnrollFactor();

  for (size_t i = 0; i < schedule.size(); ++i) {
    Op *op = schedule.at(i);

    // Set to check that no iteration is covered twice
    // (checks the model for correctness)
    std::set<LoopIteration> notCoveredIterations;
    for (LoopIteration i = 0; i < unrollFactor + 1; ++i) {
      notCoveredIterations.insert(i);
    }

    auto verifyApparentIteration = [this, &notCoveredIterations](
                                       Op *op, LoopIteration iteration) {
      if (notCoveredIterations.find(iteration) == notCoveredIterations.end()) {
        throw error("[DecomposeLoops::DecomposeLoopHelper::clone] The "
                    "model to decompose the loop is inconsistent on "
                    "apparent iteration {} for Op {} type {}. The model"
                    "needs to be adjusted such that all apparent iterations"
                    "contain each operation exactly once.",
                    iteration,
                    op->debugName(),
                    opToDecomposeLoopOpType.at(getBackupOp(op)));
      }
      notCoveredIterations.erase(iteration);
    };

    verifyApparentIteration(op, getApparentIteration(getBackupOp(op), -1));

    for (int unrollIndex = 0; unrollIndex < unrollFactor; ++unrollIndex) {
      auto cloneOpUp = op->clone();
      Op *cloneOp    = cloneOpUp.get();

      // Move from loop body graph to parent graph
      graph.moveIntoGraph(std::move(cloneOpUp));
      cloneOp->setScope(graph.getScope());

      cloneOp->setPipelineStage(loopOp->getOptionalPipelineStage());
      cloneOp->setExecutionPhase(loopOp->getOptionalExecutionPhase());

      clones[op].push_back(cloneOp);
      originals[cloneOp] = op;

      verifyApparentIteration(
          op, getApparentIteration(getBackupOp(op), unrollIndex));
    }

    if (!notCoveredIterations.empty()) {
      throw error(
          "[DecomposeLoops::DecomposeLoopHelper::clone] The "
          "model to decompose the loop is inconsistent for Op {} type {},"
          "lacks apparent iteration(s) {}. The model"
          "needs to be adjusted such that all apparent iterations"
          "contain each operation exactly once.",
          op->debugName(),
          opToDecomposeLoopOpType.at(getBackupOp(op)),
          notCoveredIterations);
    }
  }
}

Graph &DecomposeLoops::DecomposeLoopHelper::getBackupGraph() const {
  return ir.getGraph(backupGraphId);
}

Op *DecomposeLoops::DecomposeLoopHelper::getBackupOp(Op *op) const {
  return getBackupGraph().getOp(backupMaps.opIdMap.at(op->id));
}

LoopIteration DecomposeLoops::DecomposeLoopHelper::getApparentIteration(
    Op *op,
    int unrollIndex) const {
  DecomposeLoopOpTypeWrapper type = opToDecomposeLoopOpType.at(op);
  return model.getApparentIteration(type, unrollIndex);
}

bool DecomposeLoops::DecomposeLoopHelper::isBeforeLoop(Op *op,
                                                       int unrollIndex) const {
  DecomposeLoopOpTypeWrapper type = opToDecomposeLoopOpType.at(op);
  return model.isBeforeLoop(type, unrollIndex);
}

bool DecomposeLoops::DecomposeLoopHelper::isLastBeforeLoop(
    Op *op,
    int unrollIndex) const {
  // unrollIndex == -1 is inside the loop
  // Check if the current unrollIndex is before the loop, but the next one isn't
  return unrollIndex != -1 && isBeforeLoop(op, unrollIndex) &&
         !isBeforeLoop(op, unrollIndex + 1);
}

void DecomposeLoops::DecomposeLoopHelper::hookUpBeforeLoopInitialize() {
  int unrollFactor = model.getUnrollFactor();

  // Prepare tensors passed to the loop (initial inputs)
  for (auto &inputId : backupLoopOp->getCalledGraph().getInputIds()) {
    auto input = backupLoopOp->getCalledGraph().getTensor(inputId);
    TensorId originalInputId = backupMaps.tensorIdMap.at(inputId);

    InIndex sgInIndex = input->getGraphInputIndex();
    InIndex opInIndex = backupLoopOp->subgraphInToOpInIndex(sgInIndex);

    if (backupLoopOp->hasInput(opInIndex)) {

      // Seed inputs to apparent iteration 0
      beforeLoopTensorIterMap[{originalInputId, 0}] =
          backupMaps.tensorIdMap.at(backupLoopOp->inTensor(opInIndex)->id);

      // Iteration 0 inputs match "apparent iteration -1" outputs
      auto sgOutIndex = sgInIndex - 1;
      if (sgOutIndex >= 0 &&
          sgOutIndex < backupLoopOp->getCalledGraph().getOutputIds().size()) {
        auto originalOutputId = backupMaps.tensorIdMap.at(
            backupLoopOp->getCalledGraph().getOutputId(sgOutIndex));
        beforeLoopTensorIterMap[{originalOutputId, -1}] =
            backupMaps.tensorIdMap.at(backupLoopOp->inTensor(opInIndex)->id);
      }

      // Implicit input stays the same for every apparent iteration
      if (input->isImplicitLoopInput()) {
        for (LoopIteration i = -1; i < unrollFactor + 1; ++i) {
          beforeLoopTensorIterMap[{originalInputId, i}] =
              backupMaps.tensorIdMap.at(backupLoopOp->inTensor(opInIndex)->id);
          auto sgInIndex = backupLoopOp->opInToSubgraphInIndex(opInIndex);
          loopTensorIterMap[{originalInputId, i}] = backupMaps.tensorIdMap.at(
              backupLoopOp->getCalledGraph().getInputId(sgInIndex));
        }
      }
    }
  }
}

void DecomposeLoops::DecomposeLoopHelper::hookUpBeforeLoopOutputs() {
  int unrollFactor = model.getUnrollFactor();

  for (int j = 0; j < unrollFactor; ++j) {
    for (size_t i = 0; i < schedule.size(); ++i) {
      Op *op       = schedule.at(i);
      Op *backupOp = getBackupOp(op);
      if (isBeforeLoop(backupOp, j)) {
        // Outputs
        for (auto &output : backupOp->output->tensorMap()) {
          TensorId originalOutputId =
              backupMaps.tensorIdMap.at(output.second->id);

          TensorId outTensorId =
              addScope(graph, removeScope(op->getGraph(), originalOutputId));
          TensorId newOutTensorId = ir.createIntermediateTensorId(outTensorId);
          clones[op][j]->createAndConnectOutTensor(output.first,
                                                   newOutTensorId);

          beforeLoopTensorIterMap[{originalOutputId,
                                   getApparentIteration(backupOp, j)}] =
              newOutTensorId;

          if (output.second->isGraphOutput()) {
            OutIndex sgOutIndex = backupOp->getGraph()
                                      .getTensor(output.second->id)
                                      ->getGraphOutputIndex();
            InIndex sgInIndex = sgOutIndex + 1;
            TensorId backupSgInId =
                backupLoopOp->getCalledGraph().getInputId(sgInIndex);
            TensorId sgOutId = originalOutputId;
            TensorId sgInId  = backupMaps.tensorIdMap.at(backupSgInId);

            beforeLoopTensorIterMap[{sgInId,
                                     getApparentIteration(backupOp, j) + 1}] =
                newOutTensorId;
            logging::transform::trace("[DecomposeLoopHelper::hookUpBeforeLoop] "
                                      "Iteration {} output {} "
                                      "is iteration {} input {}",
                                      getApparentIteration(backupOp, j),
                                      sgOutId,
                                      getApparentIteration(backupOp, j) + 1,
                                      sgInId);
          }
        }
      }
    }
  }
}

void DecomposeLoops::DecomposeLoopHelper::hookUpBeforeLoopInputs() {
  int unrollFactor = model.getUnrollFactor();

  for (int j = 0; j < unrollFactor; ++j) {
    for (size_t i = 0; i < schedule.size(); ++i) {
      Op *op       = schedule.at(i);
      Op *backupOp = getBackupOp(op);
      if (isBeforeLoop(backupOp, j)) {
        // Inputs
        for (auto &input : backupOp->input->tensorMap()) {
          TensorId originalInputId =
              backupMaps.tensorIdMap.at(input.second->id);

          if (input.second->isImplicitLoopInput()) {
            // Directly connect to the inputs connected to the LoopOp
            InIndex sgInIndex =
                op->getGraph().getTensor(originalInputId)->getGraphInputIndex();
            InIndex opInIndex = loopOp->subgraphInToOpInIndex(sgInIndex);
            if (loopOp->hasInput(opInIndex)) {
              clones[op][j]->connectInTensorLike(
                  op, input.first, loopOp->inTensor(opInIndex)->id);
            } else {
              throw error("[DecomposeLoopHelper::hookUpAfterLoop] LoopOp "
                          "internally produced "
                          "tensors ({}) cannot be unrolled.",
                          originalInputId);
            }
          } else if (input.second->tensorType() == TensorType::Const) {
            TensorId newConstId;
            if (originalInputId.find(reservedConstValuePrefix()) !=
                std::string::npos) {
              newConstId = removeScope(op->getGraph(), originalInputId);
            } else {
              newConstId = ir.createIntermediateTensorId(
                  removeScope(op->getGraph(), originalInputId));
            }
            newConstId = addScope(graph, newConstId);
            if (!graph.getTensors().getConstIds().contains(newConstId)) {
              graph.getTensors().addConstInit(
                  newConstId,
                  input.second->info,
                  input.second->tensorData()->data());
            }
            clones[op][j]->connectInTensorLike(
                backupOp, input.first, newConstId);
          } else {
            auto it = beforeLoopTensorIterMap.find(
                {originalInputId, getApparentIteration(backupOp, j)});
            if (it != beforeLoopTensorIterMap.end()) {
              clones[op][j]->connectInTensorLike(
                  backupOp, input.first, it->second);
            } else {
              throw error("[DecomposeLoopHelper::hookUpBeforeLoop] Cannot "
                          "connect {} input {} unrollIndex {}",
                          clones[op][j]->debugName(),
                          input.first,
                          j);
            }
          }
        }
        logging::transform::trace(
            "[DecomposeLoopHelper::hookUpBeforeLoopInputs] "
            "Setting up op {} unrollIndex {}",
            clones[op][j]->debugName(),
            j);
        clones[op][j]->setup();
      }
    }
  }
}

void DecomposeLoops::DecomposeLoopHelper::hookUpBeforeLoop() {
  hookUpBeforeLoopInitialize();
  hookUpBeforeLoopOutputs();
  hookUpBeforeLoopInputs();
}

void DecomposeLoops::DecomposeLoopHelper::hookUpInLoopOutputs() {
  for (size_t i = 0; i < schedule.size(); ++i) {
    Op *op                          = schedule.at(i);
    Op *backupOp                    = getBackupOp(op);
    LoopIteration apparentIteration = getApparentIteration(backupOp, -1);
    // Register outputs
    for (auto &output : backupOp->output->tensorMap()) {
      TensorId originalOutputId = backupMaps.tensorIdMap.at(output.second->id);

      op->connectOutTensor(output.first, originalOutputId);

      loopTensorIterMap[{originalOutputId, apparentIteration}] =
          originalOutputId;

      if (output.second->isGraphOutput()) {
        OutIndex sgOutIndex = backupOp->getGraph()
                                  .getTensor(output.second->id)
                                  ->getGraphOutputIndex();
        InIndex sgInIndex = sgOutIndex + 1;
        TensorId sgInId = backupLoopOp->getCalledGraph().getInputId(sgInIndex);
        TensorId originalSgInId = backupMaps.tensorIdMap.at(sgInId);
        loopTensorIterMap[{originalSgInId, apparentIteration + 1}] =
            originalOutputId;
      }
    }
  }
}

TensorId DecomposeLoops::DecomposeLoopHelper::getThreadThroughLoopOutputId(
    TensorId originalInputId,
    Tensor *backupTensor,
    LoopIteration iterationForOutput) {
  // If the tensor is a graph output, it is also a loop output
  if (backupTensor->isGraphOutput() &&
      iterationForOutput == model.getUnrollFactor()) {
    auto backupSgOutIndex =
        backupLoopOp->getCalledGraph().getOutputIndex(backupTensor->id);
    auto backupOpOutIndex =
        backupLoopOp->subgraphOutToOpOutIndex(backupSgOutIndex);
    return backupMaps.tensorIdMap.at(
        backupLoopOp->output->id(backupOpOutIndex));
  }
  // If the tensor is an explicit loop input, it is simultaneously one of the
  // previous' iterations loop outputs
  if (backupTensor->isExplicitLoopInput() &&
      iterationForOutput == model.getUnrollFactor() + 1) {
    auto backupSgInIndex =
        backupLoopOp->getCalledGraph().getInputIndex(backupTensor->id);
    auto backupSgOutIndex = backupSgInIndex - 1;
    auto backupOpOutIndex =
        backupLoopOp->subgraphOutToOpOutIndex(backupSgOutIndex);
    return backupMaps.tensorIdMap.at(
        backupLoopOp->output->id(backupOpOutIndex));
  }
  return ir.createIntermediateTensorId(
      removeScope(loopOp->getCalledGraph(), originalInputId));
}

void DecomposeLoops::DecomposeLoopHelper::threadThroughLoop(
    TensorId originalInputId,
    LoopIteration apparentIteration) {
  InIndex loopInIndex   = LoopOp::getFirstInputInIndex();
  OutIndex loopOutIndex = LoopOp::getFirstOutputOutIndex();
  auto firstIteration   = apparentIteration;

  TensorId backupTensorId = backupMaps.tensorIdMap.at(originalInputId);
  Tensor *backupTensor =
      backupLoopOp->getCalledGraph().getTensor(backupTensorId);

  // If the requested apparentIteration is a graph output, start one iteration
  // earlier
  if (backupTensor->isGraphOutput() &&
      firstIteration == model.getUnrollFactor()) {
    firstIteration -= 1;
  }
  auto currentIteration = firstIteration;

  while (true) {
    // Tensor before the loop, connecting to the LoopOp (input)
    TensorId opTensorIdIn;
    // Tensor inside the loop, matching opTensorIdIn
    TensorId sgTensorIdIn;
    // Tensor inside the loop, matching opTensorIdOut
    TensorId sgTensorIdOut;
    // Tensor after the loop, connecting to the LoopOp (output)
    TensorId opTensorIdOut;

    // Check if the input is already available inside the loop
    auto insideItIn = loopTensorIterMap.find(
        std::make_pair(originalInputId, currentIteration));

    // Check if the input is available before the loop
    auto beforeItIn = beforeLoopTensorIterMap.find(
        std::make_pair(originalInputId, currentIteration));

    if (insideItIn != loopTensorIterMap.end()) {
      sgTensorIdIn = insideItIn->second;
      if (loopOp->getCalledGraph().getTensor(sgTensorIdIn)->isGraphOutput()) {
        // Already an output
        logging::trace("[DecomposeLoopHelper::threadThroughLoop] Stopping at "
                       "graph output {}:{}",
                       loopOp->getCalledGraph().getOutputIndex(sgTensorIdIn),
                       sgTensorIdIn);
        break;
      }
    } else if (beforeItIn != beforeLoopTensorIterMap.end()) {
      opTensorIdIn = beforeItIn->second;
      sgTensorIdIn = ir.createIntermediateTensorId(originalInputId);
    } else {
      // No further tensors
      logging::trace("[DecomposeLoopHelper::threadThroughLoop] No further "
                     "tensors to thread for {} apparent iteration {}",
                     originalInputId,
                     currentIteration);
      break;
    }

    if (currentIteration != firstIteration) {
      sgTensorIdOut = sgTensorIdIn;
      opTensorIdOut = getThreadThroughLoopOutputId(
          originalInputId, backupTensor, currentIteration);
    }

    logging::trace(
        "[DecomposeLoopHelper::threadThroughLoop] Loop "
        "input {}:{} -> {} -> output {} -> {}:{} (apparentIteration: "
        "{})",
        loopInIndex,
        opTensorIdIn,
        sgTensorIdIn,
        sgTensorIdOut,
        loopOutIndex,
        opTensorIdOut,
        currentIteration);

    if (!opTensorIdIn.empty() && !sgTensorIdIn.empty()) {
      loopOp->addLoopInput(loopInIndex, opTensorIdIn, sgTensorIdIn, false);
      loopInIndex++;
    }

    if (!opTensorIdOut.empty() && !sgTensorIdOut.empty()) {
      loopOp->addLoopOutput(loopOutIndex, opTensorIdOut, sgTensorIdOut, false);
      loopOutIndex++;
    }

    if (!sgTensorIdIn.empty()) {
      loopTensorIterMap[{originalInputId, currentIteration}] = sgTensorIdIn;
    }

    if (!opTensorIdOut.empty()) {
      afterLoopTensorIterMap[{originalInputId, currentIteration}] =
          opTensorIdOut;
    }

    if (backupTensor->isGraphOutput()) {
      OutIndex sgOutIndex = backupTensor->getGraphOutputIndex();
      InIndex sgInIndex   = sgOutIndex + 1;
      TensorId sgInId = backupLoopOp->getCalledGraph().getInputId(sgInIndex);
      TensorId originalSgInId = backupMaps.tensorIdMap.at(sgInId);
      loopTensorIterMap[{originalSgInId, apparentIteration + 1}] = sgTensorIdIn;
    }

    ++currentIteration;
  }

  if (loopInIndex - LoopOp::getFirstInputInIndex() !=
      loopOutIndex - LoopOp::getFirstOutputOutIndex()) {
    throw error("[DecomposeLoops::threadThroughLoop] Unbalanced loop carried "
                "inputs: {} / "
                "outputs: {} for tensor {}",
                loopInIndex - LoopOp::getFirstInputInIndex(),
                loopOutIndex - LoopOp::getFirstOutputOutIndex(),
                originalInputId);
  }
}

void DecomposeLoops::DecomposeLoopHelper::hookUpInLoopInputs() {
  for (size_t i = 0; i < schedule.size(); ++i) {
    Op *op                          = schedule.at(i);
    Op *backupOp                    = getBackupOp(op);
    LoopIteration apparentIteration = getApparentIteration(backupOp, -1);
    for (auto &input : backupOp->input->tensorMap()) {
      TensorId originalInputId = backupMaps.tensorIdMap.at(input.second->id);

      if (input.second->tensorType() == TensorType::Const) {
        // Constants remain connected (no iteration dependency)
        op->connectInTensorLike(backupOp, input.first, originalInputId);
      } else if (input.second->isImplicitLoopInput()) {
        // Implicit loop inputs remain connected (no iteration dependency)
        op->connectInTensorLike(backupOp, input.first, originalInputId);
      } else {
        // Ensures required tensors are available
        threadThroughLoop(originalInputId, apparentIteration);
        op->connectInTensorLike(
            backupOp,
            input.first,
            loopTensorIterMap.at({originalInputId, apparentIteration}));
      }

      if (!op->hasInput(input.first)) {
        throw error("[DecomposeLoops] No tensor for tensor {} "
                    "apparentIteration {} required by {}",
                    originalInputId,
                    apparentIteration,
                    op->debugName());
      }
    }
    op->setup();
  }
  loopOp->setup();
} // namespace popart

void DecomposeLoops::DecomposeLoopHelper::hookUpInLoop() {
  // Hook up Ops inside the loop
  hookUpInLoopOutputs();

  for (auto insideTensor : loopTensorIterMap) {
    logging::transform::trace("[DecomposeLoopHelper::hookUpInLoop] Inside loop "
                              "tensor {} -> {}, iteration {}",
                              insideTensor.first.first,
                              insideTensor.second,
                              insideTensor.first.second);
  }

  hookUpInLoopInputs();
}

void DecomposeLoops::DecomposeLoopHelper::hookUpAfterLoop() {
  // Hook up the outputs of all operations for all iterations first ...
  hookUpAfterLoopOutputs();

  for (auto afterTensor : afterLoopTensorIterMap) {
    // Log all available tensors per iteration
    logging::transform::trace("[DecomposeLoopHelper::hookUpAfterLoop] After "
                              "loop tensor {} -> {}, iteration {}",
                              afterTensor.first.first,
                              afterTensor.second,
                              afterTensor.first.second);
  }

  // ... so they can be used as inputs for subsequent iteration
  hookUpAfterLoopInputs();
}

void DecomposeLoops::DecomposeLoopHelper::hookUpAfterLoopOutputs() {
  int unrollFactor = model.getUnrollFactor();

  // Hook up Ops after the loop
  for (int j = 0; j < unrollFactor; ++j) {
    for (size_t i = 0; i < schedule.size(); ++i) {
      Op *op       = schedule.at(i);
      Op *backupOp = getBackupOp(op);
      if (!isBeforeLoop(backupOp, j)) {
        // Outputs
        auto apparentIteration = getApparentIteration(backupOp, j);
        for (auto &output : backupOp->output->tensorMap()) {
          TensorId originalOutputId =
              backupMaps.tensorIdMap.at(output.second->id);

          bool isOriginalGraphOutput = output.second->isGraphOutput();
          bool isLastIterationOutput =
              (apparentIteration == unrollFactor) && isOriginalGraphOutput;
          bool isUnrolledLoopOutput =
              op->output->tensor(output.first)->isGraphOutput();

          logging::trace("[DecomposeLoopHelper::hookUpAfterLoopOutputs] "
                         "Processing tensor {} (isOriginalGraphOutput: {}, "
                         "isLastIterationOutput: {}, isUnrolledLoopOutput: {})",
                         originalOutputId,
                         isOriginalGraphOutput,
                         isLastIterationOutput,
                         isUnrolledLoopOutput);

          // If the iteration is the last one
          if (isLastIterationOutput) {
            OutIndex sgOutIndex = output.second->getGraphOutputIndex();
            OutIndex opOutIndex =
                backupLoopOp->subgraphOutToOpOutIndex(sgOutIndex);
            if (backupLoopOp->output->hasIndex(opOutIndex)) {
              // Map final outputs and loop outputs
              Tensor *backupLoopOutTensor =
                  backupLoopOp->output->tensor(opOutIndex);

              TensorId originalLoopOutTensorId =
                  backupMaps.tensorIdMap.at(backupLoopOutTensor->id);
              Tensor *originalOpOut = graph.getTensor(originalLoopOutTensorId);

              clones[op][j]->connectOutTensor(output.first, originalOpOut->id);

              // Tensor in last iteration
              afterLoopTensorIterMap[{originalOutputId, apparentIteration}] =
                  originalOpOut->id;
            }
          } else {
            TensorId outTensorId =
                removeScope(op->getGraph(), originalOutputId);
            TensorId newOutTensorId =
                addScope(graph, ir.createIntermediateTensorId(outTensorId));
            clones[op][j]->createAndConnectOutTensor(output.first,
                                                     newOutTensorId);
            afterLoopTensorIterMap[{originalOutputId, apparentIteration}] =
                newOutTensorId;

            // Is this a loop output of the unrolled loop?
            if (isUnrolledLoopOutput) {
              afterLoopTensorIterMap[{originalOutputId,
                                      apparentIteration + 1}] = newOutTensorId;
            }

            // Is this an output of the original, non-unrolled loop?
            if (isOriginalGraphOutput) {
              // Unravel which subgraph input is fed (loop carried) by the
              // subgraph output
              OutIndex sgOutIndex = backupOp->getGraph()
                                        .getTensor(output.second->id)
                                        ->getGraphOutputIndex();
              InIndex sgInIndex = sgOutIndex + 1;
              TensorId sgInId =
                  backupLoopOp->getCalledGraph().getInputId(sgInIndex);
              TensorId originalSgInId = backupMaps.tensorIdMap.at(sgInId);
              afterLoopTensorIterMap[{originalSgInId, apparentIteration + 1}] =
                  newOutTensorId;
            }
          }
        }
      }
    }
  }

  // Make sure all tensors consumed after the loop exist
  for (auto &backupOut : backupLoopOp->output->tensorMap()) {
    auto sgOutIndex   = backupLoopOp->opOutToSubgraphOutIndex(backupOut.first);
    InIndex sgInIndex = sgOutIndex + 1;
    auto opOutId      = backupMaps.tensorIdMap.at(backupOut.second->id);
    auto sgOutId      = backupMaps.tensorIdMap.at(
        backupLoopOp->getCalledGraph().getOutputId(sgOutIndex));
    auto sgInId = backupMaps.tensorIdMap.at(
        backupLoopOp->getCalledGraph().getInputId(sgInIndex));

    Tensor *opOutTensor = loopOp->getGraph().getTensor(opOutId);
    if (!opOutTensor->hasProducer()) {
      logging::trace("[DecomposeLoopHelper::hookUpAfterLoopOutputs] Require "
                     "producer for {} {}->{}",
                     sgInId,
                     sgOutId,
                     opOutId);
      threadThroughLoop(sgOutId, unrollFactor);
    }
    if (!opOutTensor->hasProducer()) {
      throw error("[DecomposeLoopHelper::hookUpAfterLoopOutputs] Producer for "
                  "{} not found.",
                  opOutId);
    }
  }
}

void DecomposeLoops::DecomposeLoopHelper::hookUpAfterLoopInputs() {
  int unrollFactor = model.getUnrollFactor();

  for (int j = 0; j < unrollFactor; ++j) {
    for (size_t i = 0; i < schedule.size(); ++i) {
      Op *op       = schedule.at(i);
      Op *backupOp = getBackupOp(op);
      if (!isBeforeLoop(backupOp, j)) {
        // Inputs
        auto apparentIteration = getApparentIteration(backupOp, j);
        for (auto &input : backupOp->input->tensorMap()) {
          TensorId originalInputId =
              backupMaps.tensorIdMap.at(input.second->id);

          if (input.second->isImplicitLoopInput()) {
            // Directly connect to the inputs connected to the LoopOp
            InIndex sgInIndex =
                op->getGraph().getTensor(originalInputId)->getGraphInputIndex();
            InIndex opInIndex = loopOp->subgraphInToOpInIndex(sgInIndex);
            if (loopOp->hasInput(opInIndex)) {
              clones[op][j]->connectInTensorLike(
                  op, input.first, loopOp->inTensor(opInIndex)->id);
            } else {
              throw error("[DecomposeLoopHelper::hookUpAfterLoop] LoopOp "
                          "internally produced "
                          "tensors ({}) cannot be unrolled.",
                          originalInputId);
            }
          } else if (input.second->tensorType() == TensorType::Const) {

            // Is a constant
            TensorId newConstId;
            if (originalInputId.find(reservedConstValuePrefix()) !=
                std::string::npos) {
              newConstId = removeScope(op->getGraph(), originalInputId);
            } else {
              newConstId = ir.createIntermediateTensorId(
                  removeScope(op->getGraph(), originalInputId));
            }
            newConstId = addScope(graph, newConstId);
            if (!graph.getTensors().getConstIds().contains(newConstId)) {
              graph.getTensors().addConstInit(
                  newConstId,
                  input.second->info,
                  input.second->tensorData()->data());
            }
            clones[op][j]->connectInTensorLike(
                backupOp, input.first, newConstId);
          } else {
            // Not an implicit loop input or constant
            auto inTensorIdIter = afterLoopTensorIterMap.find(
                std::make_pair(originalInputId, apparentIteration));
            if (inTensorIdIter == afterLoopTensorIterMap.end()) {
              throw error("[DecomposeLoops] No tensor for tensor {} "
                          "apparentIteration {} required by {}",
                          originalInputId,
                          apparentIteration,
                          clones[op][j]->debugName());
            }
            TensorId inTensorId = inTensorIdIter->second;
            clones[op][j]->connectInTensorLike(
                backupOp, input.first, inTensorId);
          }
        }
        clones[op][j]->setup();
      }
    }
  }
}

void DecomposeLoops::DecomposeLoopHelper::
    removeTopoConsAcrossApparentIterations(
        const std::map<int, std::vector<Op *>> &apparentIterationMap) {
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
}

void DecomposeLoops::DecomposeLoopHelper::fixTopoConsBeforeLoop(
    const std::map<int, std::vector<Op *>> &beforeLoopBins) {
  // 1.) Constraints before the loop
  if (model.getTopoConLevelBefore() == DecomposeTopoConLevel::Full) {
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
}

void DecomposeLoops::DecomposeLoopHelper::fixTopoConsInsideLoop(
    const std::map<int, std::vector<Op *>> &insideLoopBins) {
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
}

void DecomposeLoops::DecomposeLoopHelper::fixTopoConsAfterLoop(
    const std::map<int, std::vector<Op *>> &afterLoopBins) {
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
}

void DecomposeLoops::DecomposeLoopHelper::fixTopoCons() {
  int unrollFactor = model.getUnrollFactor();

  // Add and remove topocons
  std::map<int, std::vector<Op *>> beforeLoopBins;
  std::map<int, std::vector<Op *>> insideLoopBins;
  std::map<int, std::vector<Op *>> afterLoopBins;

  std::map<int, std::vector<Op *>> apparentIterationMap;

  for (size_t i = 0; i < schedule.size(); ++i) {
    Op *op                 = schedule.at(i);
    Op *backupOp           = getBackupOp(op);
    auto apparentIteration = getApparentIteration(backupOp, -1);
    apparentIterationMap[apparentIteration].push_back(op);
    auto pos = model.typeToPosition(opToDecomposeLoopOpType.at(backupOp),
                                    apparentIteration);
    insideLoopBins[pos].push_back(op);
    for (int j = 0; j < unrollFactor; ++j) {
      Op *cloneOp            = clones[schedule.at(i)][j];
      auto apparentIteration = getApparentIteration(backupOp, j);
      apparentIterationMap[apparentIteration].push_back(op);
      auto pos = model.typeToPosition(opToDecomposeLoopOpType.at(backupOp),
                                      apparentIteration);
      if (isBeforeLoop(backupOp, j)) {
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
    logging::trace(
        "[DecomposeLoopHelper::fixTopoCons] {} bins: {}", name, ss.str());
  };

  if (logging::shouldLog(logging::Module::transform, logging::Level::Trace)) {
    logBins("before loop", beforeLoopBins);
    logBins("inside loop", insideLoopBins);
    logBins("after loop", afterLoopBins);
  }

  removeTopoConsAcrossApparentIterations(apparentIterationMap);
  fixTopoConsBeforeLoop(beforeLoopBins);
  fixTopoConsInsideLoop(insideLoopBins);
  fixTopoConsAfterLoop(afterLoopBins);
}

void DecomposeLoops::DecomposeLoopHelper::promoteAliases(
    InIndex backupOpInIndex) {
  int unrollFactor = model.getUnrollFactor();

  for (auto backupOpOutIndex = LoopOp::getFirstOutputOutIndex();
       backupOpOutIndex < backupLoopOp->output->n();
       ++backupOpOutIndex) {
    auto regions = backupLoopOp->aliases(backupOpInIndex, backupOpOutIndex);
    if (std::any_of(
            regions.begin(), regions.end(), [](const view::Region &region) {
              return !region.isEmpty();
            })) {

      auto backupSgInIndex =
          backupLoopOp->opInToSubgraphInIndex(backupOpInIndex);
      auto backupSgInTensorId =
          backupLoopOp->getCalledGraph().getInputId(backupSgInIndex);
      auto originalSgInTensorId = backupMaps.tensorIdMap.at(backupSgInTensorId);

      auto backupSgOutIndex =
          backupLoopOp->opOutToSubgraphOutIndex(backupOpOutIndex);
      auto backupSgOutTensorId =
          backupLoopOp->getCalledGraph().getOutputId(backupSgOutIndex);
      auto originalSgOutTensorId =
          backupMaps.tensorIdMap.at(backupSgOutTensorId);

      logging::trace("[DecomposeLoopHelper::promoteAliases] Backup LoopOp "
                     "aliased input {}:{} -> output {}:{} ",
                     backupOpInIndex,
                     backupLoopOp->inId(backupOpInIndex),
                     backupOpOutIndex,
                     backupLoopOp->outId(backupOpOutIndex));

      AliasesMap aliasesMap(backupLoopOp->getGraph());
      Aliases &aliases = aliasesMap.getAliases(backupLoopOp->getGraph());
      auto fwdAliasRegions =
          aliases.getChainsFromTo(backupLoopOp->inTensor(backupOpInIndex),
                                  backupLoopOp->outTensor(backupOpOutIndex));
      auto bwdAliasRegions =
          aliases.getChainsFromTo(backupLoopOp->outTensor(backupOpOutIndex),
                                  backupLoopOp->inTensor(backupOpInIndex));

      for (LoopIteration apparentIteration = -1;
           apparentIteration < unrollFactor + 1;
           ++apparentIteration) {
        logging::trace("[DecomposeLoopHelper::promoteAliases] Trying to find "
                       "alias chain {}->{} apparent iteration {}",
                       originalSgInTensorId,
                       originalSgOutTensorId,
                       apparentIteration);

        auto itIn =
            loopTensorIterMap.find({originalSgInTensorId, apparentIteration});
        auto itOut =
            loopTensorIterMap.find({originalSgOutTensorId, apparentIteration});

        if (itIn != loopTensorIterMap.end() &&
            itOut != loopTensorIterMap.end()) {
          auto sgInIndex = loopOp->getCalledGraph().getInputIndex(itIn->second);
          auto opInIndex = loopOp->subgraphInToOpInIndex(sgInIndex);
          auto opInTensor = loopOp->inTensor(opInIndex);

          auto sgOutIndex =
              loopOp->getCalledGraph().getOutputIndex(itOut->second);
          auto opOutIndex  = loopOp->subgraphOutToOpOutIndex(sgOutIndex);
          auto opOutTensor = loopOp->outTensor(opOutIndex);

          loopOp->addAlias(
              opInIndex, opOutIndex, fwdAliasRegions, bwdAliasRegions);
          logging::trace("[DecomposeLoopHelper::promoteAliases] Promoting "
                         "alias {}:{}->{}:{}",
                         opInIndex,
                         opInTensor->id,
                         opOutIndex,
                         opOutTensor->id);
        }
      }
    }
  }
}

void DecomposeLoops::DecomposeLoopHelper::promoteModifies(
    InIndex backupOpInIndex) {
  int unrollFactor = model.getUnrollFactor();

  auto backupSgInIndex = backupLoopOp->opInToSubgraphInIndex(backupOpInIndex);
  auto backupSgInTensorId =
      backupLoopOp->getCalledGraph().getInputId(backupSgInIndex);
  auto originalTensorId = backupMaps.tensorIdMap.at(backupSgInTensorId);

  if (backupLoopOp->modifiesIndex(backupOpInIndex)) {
    auto regions = backupLoopOp->modifies(backupOpInIndex);

    logging::trace("[DecomposeLoopHelper::promoteModifies] Backup LoopOp "
                   "modified input index {} regions {}",
                   backupOpInIndex,
                   regions);

    for (LoopIteration apparentIteration = -1;
         apparentIteration < unrollFactor + 1;
         ++apparentIteration) {
      auto it = loopTensorIterMap.find({originalTensorId, apparentIteration});
      if (it != loopTensorIterMap.end()) {
        auto sgInTensor = loopOp->getCalledGraph().getTensor(it->second);
        if (sgInTensor->isExplicitLoopInput()) {
          auto sgInIndex  = loopOp->getCalledGraph().getInputIndex(it->second);
          auto opInIndex  = loopOp->subgraphInToOpInIndex(sgInIndex);
          auto opInTensor = loopOp->inTensor(opInIndex);
          loopOp->addModified(opInIndex, regions);
          logging::trace("[DecomposeLoopHelper::promoteModifies] Promoting "
                         "modified input {}:{}->{} regions {}",
                         opInIndex,
                         opInTensor->id,
                         sgInTensor->id,
                         regions);
        }
      }
    }
  }
}

void DecomposeLoops::DecomposeLoopHelper::updateLoop() {
  int unrollFactor = model.getUnrollFactor();

  // Update trip count
  loopOp->setTripCountValue(loopOp->getTripCountValue() - unrollFactor);
  loopOp->setup();

  // Update loop aliases & modifies
  // Promote modified explicit loop inputs

  for (auto backupOpInIndex = LoopOp::getFirstInputInIndex();
       backupOpInIndex < backupLoopOp->getNumExplicitInputs();
       ++backupOpInIndex) {

    auto backupSgInIndex = backupLoopOp->opInToSubgraphInIndex(backupOpInIndex);
    auto backupSgInTensorId =
        backupLoopOp->getCalledGraph().getInputId(backupSgInIndex);
    auto backupOpInTensorId = backupLoopOp->inId(backupOpInIndex);

    logging::trace("[DecomposeLoopHelper::updateLoop] Checking aliases and "
                   "modified for backup LoopOp "
                   "input {}:{}->{}",
                   backupOpInIndex,
                   backupOpInTensorId,
                   backupSgInTensorId);

    promoteAliases(backupOpInIndex);
    promoteModifies(backupOpInIndex);
  }

  // Log the result
  logging::transform::trace(
      "[DecomposeLoopHelper::updateLoop] Summary; Decomposed LoopOp: {}",
      loopOp->debugName());

  for (int j = 0; j < unrollFactor; ++j) {
    for (size_t i = 0; i < schedule.size(); ++i) {
      Op *op       = schedule.at(i);
      Op *backupOp = getBackupOp(op);
      if (isBeforeLoop(backupOp, j)) {
        logging::transform::trace(
            "[DecomposeLoopHelper::updateLoop] Summary; Op before loop: {}",
            clones[op][j]->debugName());
      }
    }
  }
  for (size_t i = 0; i < schedule.size(); ++i) {
    Op *op = schedule.at(i);
    logging::transform::trace(
        "[DecomposeLoopHelper::updateLoop] Summary; Op inside loop: {}",
        op->debugName());
  }
  for (int j = 0; j < unrollFactor; ++j) {
    for (size_t i = 0; i < schedule.size(); ++i) {
      Op *op       = schedule.at(i);
      Op *backupOp = getBackupOp(op);
      if (!isBeforeLoop(backupOp, j)) {
        logging::transform::trace(
            "[DecomposeLoopHelper::updateLoop] Summary; Op after loop: {}",
            clones[op][j]->debugName());
      }
    }
  }
}

void DecomposeLoops::DecomposeLoopHelper::cleanup() {
  // Cleanup
  backupLoopOp->disconnectAllInputs();
  backupLoopOp->disconnectAllOutputs();
  backupLoopOp->getGraph().eraseOp(backupLoopOp->id);
  ir.removeGraph(backupGraphId);
}

bool DecomposeLoops::apply(Graph &graph) const {
  auto schedule = graph.getOpSchedule({}, RequireOptimalSchedule::No);

  // Decompose all loops
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
