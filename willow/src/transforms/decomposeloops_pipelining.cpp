// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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
#include <popart/op/ipucopy.hpp>
#include <popart/op/loop.hpp>

#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/decomposeloops.hpp>
#include <popart/util.hpp>

namespace popart {

DecomposeLoopOpPipelineType::DecomposeLoopOpPipelineType(
    PipelineStage ps_,
    DecomposeLoopOpTypeEnum type_,
    bool pipelineIpuCopy_,
    bool computeLike_)
    : DecomposeLoopOpType(), ps(ps_), type(type_),
      pipelineIpuCopy(pipelineIpuCopy_), computeLike(computeLike_) {}

bool DecomposeLoopOpPipelineType::
operator<(const DecomposeLoopOpType &other) const {
  if (auto castOther =
          dynamic_cast<const DecomposeLoopOpPipelineType *>(&other)) {
    return std::make_tuple(this->ps,
                           static_cast<int>(this->type),
                           static_cast<int>(this->pipelineIpuCopy),
                           static_cast<int>(this->computeLike)) <
           std::make_tuple(castOther->ps,
                           static_cast<int>(castOther->type),
                           static_cast<int>(castOther->pipelineIpuCopy),
                           static_cast<int>(castOther->computeLike));
  }
  return false;
}

bool DecomposeLoopOpPipelineType::
operator==(const DecomposeLoopOpType &other) const {
  if (auto castOther =
          dynamic_cast<const DecomposeLoopOpPipelineType *>(&other)) {
    return *this == *castOther;
  }
  return false;
}

bool DecomposeLoopOpPipelineType::
operator!=(const DecomposeLoopOpType &other) const {
  if (auto castOther =
          dynamic_cast<const DecomposeLoopOpPipelineType *>(&other)) {
    return *this != *castOther;
  }
  return true;
}

bool DecomposeLoopOpPipelineType::
operator==(const DecomposeLoopOpPipelineType &other) const {
  return std::make_tuple(
             this->ps, this->type, this->pipelineIpuCopy, this->computeLike) ==
         std::make_tuple(
             other.ps, other.type, other.pipelineIpuCopy, other.computeLike);
}

bool DecomposeLoopOpPipelineType::
operator!=(const DecomposeLoopOpPipelineType &other) const {
  return !(*this == other);
}

std::ostream &DecomposeLoopOpPipelineType::output(std::ostream &os) const {
  os << "(PipelineStage: " << ps << ", type: " << type
     << ", isPipelineIpuCopy: " << pipelineIpuCopy
     << ", isComputeLike: " << computeLike << ")";
  return os;
}

DecomposeLoopPipelineModel::DecomposeLoopPipelineModel(
    int minStage_,
    int maxStage_,
    int numStages_,
    DecomposeTopoConLevel topoConLevelBefore_,
    DecomposeTopoConLevel topoConLevelLoop_,
    DecomposeTopoConLevel topoConLevelAfter_,
    const std::set<ExchangeStrategy> &computeLikeExchangeStrategies_)
    : DecomposeLoopModel(topoConLevelBefore_,
                         topoConLevelLoop_,
                         topoConLevelAfter_,
                         computeLikeExchangeStrategies_),
      minStage(minStage_), maxStage(maxStage_), numStages(numStages_) {}

int DecomposeLoopPipelineModel::getUnrollFactor() const {
  // If maxStage == minStage, unroll once where all pipeline stages less than
  // minStage will be pulled before the loop.
  // See DecomposeLoopPipelineModel header for the general unroll logic.
  return maxStage - minStage + 1;
}

int DecomposeLoopPipelineModel::typeToPosition(DecomposeLoopOpTypeWrapper type,
                                               LoopIteration iteration) const {
  auto utype = unwrap(type);

  auto et = utype.getType();
  auto ps = utype.getPipelineStage();
  auto pc = utype.isPipelineIpuCopy();
  auto cl = utype.isComputeLike();

  auto getTypeOffset = [&et, &pc, &cl]() {
    // Numbers derived from table in DecomposeLoopPipelineModel comment
    switch (et) {
    case DecomposeLoopOpTypeEnum::AuxiliaryBefore:
      if (cl) {
        return 10;
      } else {
        return 2;
      }
    case DecomposeLoopOpTypeEnum::IoBeforeCompute:
      if (cl) {
        return 11;
      } else {
        return 3;
      }
    case DecomposeLoopOpTypeEnum::IoToCompute:
      return 7;
    case DecomposeLoopOpTypeEnum::Compute:
      if (pc) {
        return 19;
      } else {
        return 14;
      }
    case DecomposeLoopOpTypeEnum::ComputeToIo:
      return 18;
    case DecomposeLoopOpTypeEnum::IoAfterCompute:
      if (cl) {
        return 15;
      } else {
        return 23;
      }
    case DecomposeLoopOpTypeEnum::AuxiliaryAfter:
      if (cl) {
        return 16;
      } else {
        return 26;
      }
    case DecomposeLoopOpTypeEnum::N:
      throw internal_error("Unsupported DecomposeLoopOpTypeEnum value {}", et);
    }
    throw internal_error("DecomposeLoopPipelineModel::typeToPosition::"
                         "getTypeOffset Unhandled case.");
  };

  // The type of Op determines the base position offset
  auto typeOffset = getTypeOffset();

  // All of the possible positions (see table), defines how many positions two
  // ops of the same type are apart
  int cycleStride = 10;

  // Each iteration and each pipeline stage shift the positions by cycleStride
  return typeOffset + (ps + iteration) * cycleStride;
}

std::pair<int, int>
DecomposeLoopPipelineModel::getAdjustedPipelineStageAndIterationsBeforeLoop(
    DecomposeLoopOpPipelineType type) const {
  // Offset the pipelineStage by the Op type to enable overlapped IO
  int adjustedPs = type.getPipelineStage();

  std::set<DecomposeLoopOpPipelineType> setBefore{
      DecomposeLoopOpPipelineType::auxBefore(adjustedPs),
      DecomposeLoopOpPipelineType::ioBefore(adjustedPs),
      DecomposeLoopOpPipelineType::ioToCompute(adjustedPs)};

  std::set<DecomposeLoopOpPipelineType> setAfter{
      DecomposeLoopOpPipelineType::ioAfter(adjustedPs),
      DecomposeLoopOpPipelineType::auxAfter(adjustedPs)};

  if (setBefore.find(type) != setBefore.end()) {
    // Treat Ops of these types like the previous PipelineStage
    adjustedPs -= 1;
  } else if (setAfter.find(type) != setAfter.end()) {
    // Treat Ops of these types like the next PipelineStage
    adjustedPs += 1;
  } else {
    // Treat Ops of these types like the current PipelineStage
    adjustedPs += 0;
  }

  int iterationsBeforeLoop = 0;
  for (int i = minStage; i <= maxStage; ++i) {
    if (adjustedPs < i) {
      iterationsBeforeLoop++;
    }
  }

  return {adjustedPs, iterationsBeforeLoop};
}

LoopIteration DecomposeLoopPipelineModel::getApparentIteration(
    DecomposeLoopOpTypeWrapper type,
    int unrollIndex) const {
  auto utype = unwrap(type);

  auto adjustedPair = getAdjustedPipelineStageAndIterationsBeforeLoop(utype);
  int iterationsBeforeLoop = adjustedPair.second;

  if (unrollIndex == -1) {
    // Inside the loop
    return iterationsBeforeLoop;
  } else {
    // Before or after the loop
    return unrollIndex + ((unrollIndex < iterationsBeforeLoop) ? 0 : 1);
  }
}

bool DecomposeLoopPipelineModel::isBeforeLoop(DecomposeLoopOpTypeWrapper type,
                                              int unrollIndex) const {
  auto utype = unwrap(type);

  if (unrollIndex == -1) {
    // Inside the loop
    return false;
  } else {
    // Before or after the loop
    auto adjustedPair = getAdjustedPipelineStageAndIterationsBeforeLoop(utype);
    int iterationsBeforeLoop = adjustedPair.second;

    // Check how many iterations of this PipelineStage and Op type combination
    // are expected before the loop
    return unrollIndex < iterationsBeforeLoop;
  }
}

namespace {
/**
 * Helper class to organize functions required to classify operations
 */
class DecomposeLoopOpTypePipelineHelper {
public:
  DecomposeLoopOpTypePipelineHelper(const DecomposeLoopPipelineModel &model_,
                                    Op *op_,
                                    bool allowSeeding_,
                                    bool allowDelaying_)
      : model(model_), op(op_), ps(op->getPipelineStage()),
        tileSet(op->settings.tileSet), allowSeeding(allowSeeding_),
        allowDelaying(allowDelaying_) {
    auxBefore = DecomposeLoopOpPipelineType::auxBefore(ps);
    auxBeforeComputeLike =
        DecomposeLoopOpPipelineType::auxBeforeComputeLike(ps);
    ioBefore            = DecomposeLoopOpPipelineType::ioBefore(ps);
    ioBeforeComputeLike = DecomposeLoopOpPipelineType::ioBeforeComputeLike(ps);
    ioToCompute         = DecomposeLoopOpPipelineType::ioToCompute(ps);
    compute             = DecomposeLoopOpPipelineType::compute(ps);
    computePipelineIpuCopy =
        DecomposeLoopOpPipelineType::computePipelineIpuCopy(ps);
    computeToIO         = DecomposeLoopOpPipelineType::computeToIO(ps);
    ioAfter             = DecomposeLoopOpPipelineType::ioAfter(ps);
    ioAfterComputeLike  = DecomposeLoopOpPipelineType::ioAfterComputeLike(ps);
    auxAfter            = DecomposeLoopOpPipelineType::auxAfter(ps);
    auxAfterComputeLike = DecomposeLoopOpPipelineType::auxAfterComputeLike(ps);

    isIoOp           = DecomposeLoops::isIOOp(op);
    isIoTileCopy     = op->isConvertibleTo<IoTileCopyOp>();
    isOnIOTiles      = tileSet == TileSet::IO;
    isOnComputeTiles = tileSet == TileSet::Compute;
    computeLike      = (DecomposeLoops::isComputeLikeIOOp(
                      model_.getComputeLikeExchangeStrategies(), op)) ||
                  (!isIoTileCopy && isOnComputeTiles);

    isPipelineIpuCopy = op->isPipelineIpuCopyOp();
  };

  /**
   * Register any type of operation occuring before the current operation.
   * \param opToDecomposeLoopOpType Existing Op to type mapping.
   * \param bop                     Operation, which occurs before the Op
   *                                associated with this class, to register.
   */
  void registerBefore(const std::map<Op *, DecomposeLoopOpTypeWrapper, POpCmp>
                          &opToDecomposeLoopOpType,
                      Op *bop) {
    if (bop->hasPipelineStage()) {
      if (bop->getPipelineStage() == ps) {
        auto it = opToDecomposeLoopOpType.find(bop);
        if (it != opToDecomposeLoopOpType.end()) {
          beforeTypes.insert(model.unwrap(it->second));
        }
      } else {
        fromOtherPipelineStage = true;
      }
    }
  };

  /**
   * Register any type of operation occuring after the current operation.
   * \param opToDecomposeLoopOpType Existing Op to type mapping.
   * \param aop                     Operation, which occurs after the Op
   *                                associated with this class, to register.
   */
  void registerAfter(const std::map<Op *, DecomposeLoopOpTypeWrapper, POpCmp>
                         &opToDecomposeLoopOpType,
                     Op *aop) {
    if (aop->hasPipelineStage()) {
      if (aop->getPipelineStage() == ps) {
        auto it = opToDecomposeLoopOpType.find(aop);
        if (it != opToDecomposeLoopOpType.end()) {
          afterTypes.insert(model.unwrap(it->second));
        }
      } else {
        toOtherPipelineStage = true;
      }
    }
  };

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
  bool anyTypes(std::set<DecomposeLoopOpPipelineType> s0,
                std::set<DecomposeLoopOpPipelineType> s1) {
    return std::any_of(
        s1.begin(), s1.end(), [&s0](const DecomposeLoopOpPipelineType &t1) {
          return std::find(s0.begin(), s0.end(), t1) != s0.end();
        });
  }

  /**
   * Return if any types in s1 occur before the current Op.
   * \param s1 Set of types to check.
   * \return   True if any types in s1 occur in beforeTypes, false otherwise.
   */
  bool anyBefore(std::set<DecomposeLoopOpPipelineType> s1) {
    return anyTypes(beforeTypes, s1);
  }

  /**
   * Return if any types in s1 occur after the current Op.
   * \param s1 Set of types to check.
   * \return   True if any types in s1 occur in afterTypes, false otherwise.
   */
  bool anyAfter(std::set<DecomposeLoopOpPipelineType> s1) {
    return anyTypes(afterTypes, s1);
  }

  DecomposeLoopOpTypeWrapper ioTileCopyOpRule() {
    if (isIoTileCopy) {
      if (isOnComputeTiles) {
        if (anyBefore({compute,
                       computePipelineIpuCopy,
                       computeToIO,
                       ioAfter,
                       ioAfterComputeLike,
                       auxAfter,
                       auxAfterComputeLike})) {
          return auxAfter;
        }
        if (computeLike) {
          return compute;
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
    // fromOtherPipelineStage cannot run earlier
    if (!isIoOp && !fromOtherPipelineStage) {
      if ((anyAfter({ioBefore})) || anyAfter({auxBefore})) {
        return auxBefore;
      }

      if ((anyAfter({ioBeforeComputeLike})) ||
          anyAfter({auxBeforeComputeLike})) {
        return auxBeforeComputeLike;
      }
    }
    return {};
  }

  DecomposeLoopOpTypeWrapper opsAfterEverythingRule() {
    // toOtherPipelineStage cannot run later
    if (!isIoOp && !toOtherPipelineStage) {
      if ((!isIoOp && anyBefore({ioAfter})) || anyBefore({auxAfter})) {
        return auxAfter;
      }

      if ((!isIoOp && anyBefore({ioAfterComputeLike})) ||
          anyBefore({auxAfterComputeLike})) {
        return auxAfterComputeLike;
      }
    }
    return {};
  }

  DecomposeLoopOpTypeWrapper ioOperationRule() {
    if (isIoOp) {
      if (computeLike) {
        if (fromOtherPipelineStage ||
            anyBefore(
                {compute, computePipelineIpuCopy, ioToCompute, computeToIO}) ||
            (allowDelaying && !anyAfter({auxBefore,
                                         auxBeforeComputeLike,
                                         ioBefore,
                                         ioBeforeComputeLike,
                                         ioToCompute,
                                         compute}))) {
          return ioAfterComputeLike;
        } else {
          return ioBeforeComputeLike;
        }
      } else {
        if (fromOtherPipelineStage ||
            anyBefore(
                {compute, computePipelineIpuCopy, ioToCompute, computeToIO}) ||
            (allowDelaying && !anyAfter({auxBefore,
                                         auxBeforeComputeLike,
                                         ioBefore,
                                         ioBeforeComputeLike,
                                         ioToCompute,
                                         compute,
                                         computePipelineIpuCopy,
                                         ioAfterComputeLike,
                                         auxAfterComputeLike,
                                         computeToIO}))) {
          return ioAfter;
        } else {
          return ioBefore;
        }
      }
    }
    return {};
  }

  DecomposeLoopOpTypeWrapper computeOperationRule() {
    if (!isIoTileCopy && !isIoOp) {
      if (fromOtherPipelineStage || toOtherPipelineStage ||
          anyBefore({ioToCompute,
                     ioBefore,
                     ioBeforeComputeLike,
                     compute,
                     computePipelineIpuCopy})) {
        if (anyBefore({computePipelineIpuCopy})) {
          return auxAfterComputeLike;
        }
        if (isPipelineIpuCopy) {
          return computePipelineIpuCopy;
        } else {
          return compute;
        }
      }
    }
    return {};
  }

  DecomposeLoopOpTypeWrapper seedRule() {
    if (allowSeeding) {
      if (isPipelineIpuCopy) {
        return computePipelineIpuCopy;
      }
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
          // If there is no dependency, run as early as possible
          return auxBefore;
        }
      }
      if (isOnComputeTiles) {
        if (!afterTypes.empty()) {
          auto type = *afterTypes.begin();
          if (type == ioBefore || type == ioToCompute) {
            // Op can occur before any IO or compute
            type = auxBefore;
          }
          if (type == ioBeforeComputeLike) {
            // Op can occur before any IO or compute
            type = auxBeforeComputeLike;
          }
          return type;
        }
        if (!beforeTypes.empty()) {
          auto type = *beforeTypes.rbegin();
          if (type == ioAfter || type == computeToIO) {
            // Ensuring we do not block IO merging by pushing the Op later
            type = auxAfter;
          }
          if (type == ioAfterComputeLike) {
            // Op can occur before any IO or compute
            type = auxAfterComputeLike;
          }
          return type;
        }
        if (allowDelaying) {
          // If there is no dependency, run as compute
          return compute;
        }
      }
    }
    return {};
  }

  const DecomposeLoopPipelineModel &model;

  Op *op;
  PipelineStage ps;
  TileSet tileSet;

  bool allowSeeding;
  bool allowDelaying;

  // If the Op has an input from another PipelineStage
  bool fromOtherPipelineStage = false;

  // If the Op has an output to another PipelineStage
  bool toOtherPipelineStage = false;

  std::set<DecomposeLoopOpPipelineType> beforeTypes;
  std::set<DecomposeLoopOpPipelineType> afterTypes;

  // Op properties
  bool isIoOp;
  bool isIoTileCopy;
  bool isOnIOTiles;
  bool isOnComputeTiles;
  bool computeLike;
  bool isPipelineIpuCopy;

  // Short-hands of valid types (specific to one pipeline stage)
  DecomposeLoopOpPipelineType auxBefore;
  DecomposeLoopOpPipelineType auxBeforeComputeLike;
  DecomposeLoopOpPipelineType ioBefore;
  DecomposeLoopOpPipelineType ioBeforeComputeLike;
  DecomposeLoopOpPipelineType ioToCompute;
  DecomposeLoopOpPipelineType compute;
  DecomposeLoopOpPipelineType computePipelineIpuCopy;
  DecomposeLoopOpPipelineType computeToIO;
  DecomposeLoopOpPipelineType ioAfter;
  DecomposeLoopOpPipelineType ioAfterComputeLike;
  DecomposeLoopOpPipelineType auxAfter;
  DecomposeLoopOpPipelineType auxAfterComputeLike;
};
} // namespace

DecomposeLoopOpTypeWrapper DecomposeLoopPipelineModel::getDecomposeLoopOpType(
    const std::map<Op *, DecomposeLoopOpTypeWrapper, POpCmp>
        &opToDecomposeLoopOpType,
    Op *op,
    bool allowSeeding,
    bool allowDelaying) const {

  DecomposeLoopOpTypePipelineHelper helper(
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

std::set<DecomposeLoopOpTypeWrapper>
DecomposeLoopPipelineModel::getDecomposeLoopOpTypesToCheck() const {

  std::set<DecomposeLoopOpTypeWrapper> typeSet;

  for (PipelineStage ps = 0; ps < numStages; ++ps) {
    auto auxBefore = DecomposeLoopOpPipelineType::auxBefore(ps);
    auto auxBeforeComputeLike =
        DecomposeLoopOpPipelineType::auxBeforeComputeLike(ps);
    auto ioBefore = DecomposeLoopOpPipelineType::ioBefore(ps);
    auto ioBeforeComputeLike =
        DecomposeLoopOpPipelineType::ioBeforeComputeLike(ps);
    auto ioToCompute = DecomposeLoopOpPipelineType::ioToCompute(ps);
    auto compute     = DecomposeLoopOpPipelineType::compute(ps);
    auto computePipelineIpuCopy =
        DecomposeLoopOpPipelineType::computePipelineIpuCopy(ps);
    auto computeToIO = DecomposeLoopOpPipelineType::computeToIO(ps);
    auto ioAfter     = DecomposeLoopOpPipelineType::ioAfter(ps);
    auto ioAfterComputeLike =
        DecomposeLoopOpPipelineType::ioAfterComputeLike(ps);
    auto auxAfter = DecomposeLoopOpPipelineType::auxAfter(ps);
    auto auxAfterComputeLike =
        DecomposeLoopOpPipelineType::auxAfterComputeLike(ps);

    // For the unrolling to be valid, every type of operation has to exist
    // within the loop body exactly once.

    // This invariable condition (1.) and 2.)) results in the consequence
    // that certain types of operations can only exist
    // if the unroll factor (defined through minStage and maxStage, which
    // signify the lowest and highest pipeline stage to pull in front of the
    // loop during unrolling) accounts for them.

    // 1.) Types only allowed to exist if the overlapped IO is to be unrolled
    if (minStage <= ps) {
      typeSet.insert(auxBefore);
      typeSet.insert(ioBefore);
      typeSet.insert(ioToCompute);
    }

    typeSet.insert(auxBeforeComputeLike);
    typeSet.insert(ioBeforeComputeLike);
    typeSet.insert(compute);
    typeSet.insert(computePipelineIpuCopy);
    typeSet.insert(computeToIO);
    typeSet.insert(ioAfterComputeLike);
    typeSet.insert(auxAfterComputeLike);

    // 2.) Types only allowed to exist if the overlapped IO is to be unrolled
    if (maxStage > ps) {
      typeSet.insert(ioAfter);
      typeSet.insert(auxAfter);
    }
  }

  return typeSet;
}

int DecomposeLoopPipelineModel::getTypeGroup(
    DecomposeLoopOpTypeWrapper type) const {
  auto utype = unwrap(type);
  // Group by pipeline stage
  return utype.getPipelineStage();
}

} // namespace popart
