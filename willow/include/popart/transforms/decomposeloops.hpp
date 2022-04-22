// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DECOMPOSELOOPS_HPP
#define GUARD_NEURALNET_DECOMPOSELOOPS_HPP

#include <cstddef>
#include <iosfwd>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <popart/pointercomparators.hpp>
#include <popart/transforms/transform.hpp>
#include <popart/util.hpp>

#include "popart/graphid.hpp"
#include "popart/names.hpp"
#include "popart/op/exchange/exchange.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {
class Graph;
class Ir;
class LoopOp;
class Op;
class Tensor;

using LoopIteration = int;
using LoopTensorMap = std::map<std::pair<TensorId, LoopIteration>, TensorId>;

/**
 * Enum classifying any \c Op into 7 types according to their function in an
 * overlapped, skewed unroll of a \c LoopOp subgraph
 * (see \ref DecomposeLoops class for definition of skewed unrolling).
 *
 * `before`: Describes the operations scheduled before the unrolled LoopOp.
 * `loop`: Describes the operations scheduled inside the LoopOp after unrolling.
 * `after`: Describes the operations scheduled after the unrolled LoopOp.
 *
 * Overview over all possible base types:
 * \ref DecomposeLoopOpTypeEnum::AuxiliaryBefore
 *      Type that describes any operation that runs on either IO or compute
 *      tiles that have to occur (due to data dependencies) before any of the
 *      other types.
 *
 * \ref DecomposeLoopOpTypeEnum::IoBeforeCompute
 *      Type that describes any operation that does IO on IO tiles, and is
 *      desirable to overlap with computation that occurs after execution of
 *      this type.
 *      This includes `HostLoadOp`, `HostStoreOp`, `RemoteLoadOp`,
 *      `RemoteStoreOp`, `MultiExchangeOp`.
 *
 * \ref DecomposeLoopOpTypeEnum::IoToCompute
 *      Type that describes any operation that copies data from IO tiles to
 *      compute tiles. This includes `IoTileCopyOp`.
 *
 * \ref DecomposeLoopOpTypeEnum::Compute
 *      Type that describes any operation that runs on compute tiles, or
 *      operations that run on IO tiles but should be treated as `::Compute`.
 *
 * \ref DecomposeLoopOpTypeEnum::ComputeToIo
 *      Type that describes any operation that copies data from compute tiles to
 *      IO tiles. This includes `IoTileCopyOp`.
 *
 * \ref DecomposeLoopOpTypeEnum::IoAfterCompute
 *      Type that describes any operation that does IO on IO tiles, and is
 *      desirable to overlap with computation that occurs before.
 *      This includes `HostLoadOp`, `HostStoreOp`, `RemoteLoadOp`,
 *      `RemoteStoreOp`, `MultiExchangeOp`.
 *
 * \ref DecomposeLoopOpTypeEnum::AuxiliaryAfter
 *      Type that describes any operation that runs on either IO or compute
 *      tiles that have to occur (due to data dependencies) after any of the
 *      other types.
 *
 * Note that classification can deviate if data or topological constraint
 * dependencies do not allow the typical classification of a certain operation.
 *
 * Overlap between the operations marked below occurs opportunistically,
 * by the IO operations executing on `IO tiles`, and the compute operations
 * executing on `compute tiles`. If IO is arranged before compute, without any
 * data dependencies between them, there is potential for parallel execution.
 *
 * For further IO overlap documentation, see:
 * https://docs.graphcore.ai/projects/popart-user-guide/en/latest/overlap_io.html
 *
 */
enum class DecomposeLoopOpTypeEnum {
  //                   // Schedule (number denotes apparent loop iteration
  //                   // count):
  //                   // (example shown is IO overlapped unrolling)
  //                   // before  | loop  | after
  AuxiliaryBefore = 0, // 0..1....|2......|......
  IoBeforeCompute,     // .0..1...|..2....|......
  IoToCompute,         // ..0...1.|.....2.|......
  Compute,             // .....0..|...1...|.2....
  ComputeToIo,         // .......0|......1|...2..
  IoAfterCompute,      // ........|.0.....|1...2.
  AuxiliaryAfter,      // ........|....0..|..1..2
  // Overlap:          //     ^^    ^^^    ^^
  N // Number of enums
};

/**
 * Abstract base class for classifying an \c Op.
 * Classifying operations is required so the decomposer can do skewed unrolling
 * (see \ref DecomposeLoops class for definition of skewed unrolling)
 * by figuring out how many iterations of each operation type need to occur
 * before and after the loop.
 */
class DecomposeLoopOpType {
public:
  DecomposeLoopOpType() {}

  /**
   * Sort Op type arbitrarily, required for maps and sets of op types.
   * \param other Other instance of DecomposeLoopOpType
   * \return      True if this < other
   */
  virtual bool operator<(const DecomposeLoopOpType &other) const = 0;

  virtual std::ostream &output(std::ostream &) const = 0;
  virtual ~DecomposeLoopOpType() {}

  virtual bool operator==(const DecomposeLoopOpType &other) const = 0;
  virtual bool operator!=(const DecomposeLoopOpType &other) const = 0;
};

/**
 * Class to classify operations for IO overlapped unrolling.
 */
class DecomposeLoopOpIOOverlapType : public DecomposeLoopOpType {
public:
  /**
   * Create type for IO overlap. This type only wraps DecomposeLoopOpTypeEnum.
   * \param type DecomposeLoopOpTypeEnum
   */
  DecomposeLoopOpIOOverlapType(DecomposeLoopOpTypeEnum type);
  bool operator<(const DecomposeLoopOpType &other) const override;
  bool operator==(const DecomposeLoopOpType &other) const override;
  bool operator!=(const DecomposeLoopOpType &other) const override;

  DecomposeLoopOpTypeEnum getType() const { return type; }

  std::ostream &output(std::ostream &) const override;

private:
  DecomposeLoopOpTypeEnum type;
};

/**
 * Class to classify operations for pipeline unrolling
 * (see \ref DecomposeLoopPipelineModel).
 */
class DecomposeLoopOpPipelineType : public DecomposeLoopOpType {
public:
  /**
   * Default constructor required for data structures, creates invalid type.
   */
  DecomposeLoopOpPipelineType()
      : ps(unusedPipelineStage), type(DecomposeLoopOpTypeEnum::AuxiliaryBefore),
        pipelineIpuCopy(false), computeLike(false) {}

  /**
   * Create Op type for pipelining.
   * \param ps              The pipeline stage of the Op
   * \param type            The enum type assigned to the Op.
   * \param pipelineIpuCopy True if the operation is an IpuCopyOp between
   *                        pipeline stages.
   * \param computeLike     True if the operation is an IO operation treated
   *                        like compute.
   */
  DecomposeLoopOpPipelineType(PipelineStage ps,
                              DecomposeLoopOpTypeEnum type,
                              bool pipelineIpuCopy,
                              bool computeLike);

  bool operator<(const DecomposeLoopOpType &other) const override;
  bool operator==(const DecomposeLoopOpType &other) const override;
  bool operator!=(const DecomposeLoopOpType &other) const override;
  bool operator==(const DecomposeLoopOpPipelineType &other) const;
  bool operator!=(const DecomposeLoopOpPipelineType &other) const;

  /**
   * Get the pipeline stage.
   * \return PipelineStage associated with the type.
   */
  PipelineStage getPipelineStage() const { return ps; }

  /**
   * Get the enum type.
   * \return DecomposeLoopOpTypeEnum associated with the type.
   */
  DecomposeLoopOpTypeEnum getType() const { return type; }

  /**
   * Check if the type is a copy between pipeline stages.
   * \return True if the operation is an IpuCopyOp between pipeline stages.
   */
  bool isPipelineIpuCopy() const { return pipelineIpuCopy; }

  /**
   * Check if the operation is an IO operation treated like a compute operation
   * (as in, classified, unrolled and scheduled as
   * `DecomposeLoopOpTypeEnum::Compute`) (instead of an IO operation that should
   * overlap with compute (classified `DecomposeLoopOpTypeEnum::IoBeforeCompute`
   * or `DecomposeLoopOpTypeEnum::IoAfterCompute`)).
   *
   * \return True if the operation is an IO operation treated like a compute
   *         operation.
   */
  bool isComputeLike() const { return computeLike; }

  std::ostream &output(std::ostream &) const override;

  // Short-hands of all valid types.
  static DecomposeLoopOpPipelineType auxBefore(PipelineStage ps) {
    return DecomposeLoopOpPipelineType(
        ps, DecomposeLoopOpTypeEnum::AuxiliaryBefore, false, false);
  }

  static DecomposeLoopOpPipelineType auxBeforeComputeLike(PipelineStage ps) {
    return DecomposeLoopOpPipelineType(
        ps, DecomposeLoopOpTypeEnum::AuxiliaryBefore, false, true);
  }

  static DecomposeLoopOpPipelineType ioBefore(PipelineStage ps) {
    return DecomposeLoopOpPipelineType(
        ps, DecomposeLoopOpTypeEnum::IoBeforeCompute, false, false);
  }

  static DecomposeLoopOpPipelineType ioBeforeComputeLike(PipelineStage ps) {
    return DecomposeLoopOpPipelineType(
        ps, DecomposeLoopOpTypeEnum::IoBeforeCompute, false, true);
  }

  static DecomposeLoopOpPipelineType ioToCompute(PipelineStage ps) {
    return DecomposeLoopOpPipelineType(
        ps, DecomposeLoopOpTypeEnum::IoToCompute, false, false);
  }

  static DecomposeLoopOpPipelineType compute(PipelineStage ps) {
    return DecomposeLoopOpPipelineType(
        ps, DecomposeLoopOpTypeEnum::Compute, false, false);
  }

  static DecomposeLoopOpPipelineType computePipelineIpuCopy(PipelineStage ps) {
    return DecomposeLoopOpPipelineType(
        ps, DecomposeLoopOpTypeEnum::Compute, true, false);
  }

  static DecomposeLoopOpPipelineType computeToIO(PipelineStage ps) {
    return DecomposeLoopOpPipelineType(
        ps, DecomposeLoopOpTypeEnum::ComputeToIo, false, false);
  }

  static DecomposeLoopOpPipelineType ioAfter(PipelineStage ps) {
    return DecomposeLoopOpPipelineType(
        ps, DecomposeLoopOpTypeEnum::IoAfterCompute, false, false);
  }

  static DecomposeLoopOpPipelineType ioAfterComputeLike(PipelineStage ps) {
    return DecomposeLoopOpPipelineType(
        ps, DecomposeLoopOpTypeEnum::IoAfterCompute, false, true);
  }

  static DecomposeLoopOpPipelineType auxAfter(PipelineStage ps) {
    return DecomposeLoopOpPipelineType(
        ps, DecomposeLoopOpTypeEnum::AuxiliaryAfter, false, false);
  }

  static DecomposeLoopOpPipelineType auxAfterComputeLike(PipelineStage ps) {
    return DecomposeLoopOpPipelineType(
        ps, DecomposeLoopOpTypeEnum::AuxiliaryAfter, false, true);
  }

private:
  /// The pipeline stage the Op belongs to
  PipelineStage ps;
  /// The Op classification by function type
  DecomposeLoopOpTypeEnum type;
  /// If the Op is an Ipu copy between pipeline stages
  bool pipelineIpuCopy;
  /// If the Op is to be treated like compute (no overlapped IO)
  bool computeLike;
};

/**
 * Interface class wrapper for classifying operations into types.
 * Hides implementation details and allows data structures of the
 * DecomposeLoopOpType subclasses.
 */
class DecomposeLoopOpTypeWrapper {
public:
  DecomposeLoopOpTypeWrapper() : type(nullptr) {}

  template <typename T>
  DecomposeLoopOpTypeWrapper(T t) : type(std::make_shared<T>(t)) {}

  DecomposeLoopOpTypeWrapper(const DecomposeLoopOpTypeWrapper &wrapper) =
      default;

  DecomposeLoopOpTypeWrapper &
  operator=(const DecomposeLoopOpTypeWrapper &wrapper) {
    type = wrapper.type;
    return *this;
  }

  template <typename T> const T *getType() const {
    return static_cast<const T *>(type.get());
  }

  bool operator<(const DecomposeLoopOpTypeWrapper &other) const {
    return *type < *other.type;
  }

  bool hasValue() { return type.get() != nullptr; }

  bool operator==(const DecomposeLoopOpTypeWrapper &other) const {
    if (type == nullptr && other.type != nullptr) {
      return false;
    } else if (type != nullptr && other.type == nullptr) {
      return false;
    } else if (type == nullptr && other.type == nullptr) {
      return true;
    }
    return *type == *other.type;
  }

  bool operator!=(const DecomposeLoopOpTypeWrapper &other) const {
    if (type == nullptr && other.type != nullptr) {
      return true;
    } else if (type != nullptr && other.type == nullptr) {
      return true;
    } else if (type == nullptr && other.type == nullptr) {
      return false;
    }
    return *type != *other.type;
  }

private:
  std::shared_ptr<const DecomposeLoopOpType> type;
};

/**
 * Enum describing extent to which topological schedule constraints (topocons)
 * should be applied after decomposition.
 * (see \ref topocons.hpp).
 */
enum class DecomposeTopoConLevel {
  /// Do not restrict schedule with topocons.
  None,
  /// Fully restrict schedule with topocons. Each operation type (see \ref
  /// DecomposeLoopOpTypeEnum) will receive topological constraints against all
  /// other types, thereby cementing the schedule as described in i.e. \ref
  /// DecomposeLoopOpTypeEnum.
  Full,
  /// Number of levels
  N
};

std::ostream &operator<<(std::ostream &os, const DecomposeLoopOpTypeWrapper &);
std::ostream &operator<<(std::ostream &os, const DecomposeLoopOpType &);
std::ostream &operator<<(std::ostream &os, const DecomposeLoopOpTypeEnum &);

/**
 * The \c DecomposeLoopModel interface describes an abstract way to:
 * - Determine how many iterations need to be unrolled
 * - Classify the operations into types in accordance with a heuristic model
 *   (each decomposition model can describe it's own Op type with the
 *   \c DecomposeLoopOpType interface).
 * - Assign a schedule position to each classified \c Op type based on the
 *   apparent iteration.
 * - Check if an \c Op type and \c unrollIndex should be before or after the
 *   LoopOp.
 * - Assign an apparent iteration to each classified \c Op type based on the
 *   \c unrollIndex. The apparent iteration is the original loop iteration
 *   covered by an unrolled operation.
 */
class DecomposeLoopModel {
public:
  virtual std::string getName() const { return "DecomposeLoopModel"; }

  /**
   * DecomposeLoopModel instance
   * Defaults to \c DecomposeTopoConLevel::None
   * Defaults to \c ExchangeStrategy::JustInTime
   */
  DecomposeLoopModel();

  /**
   * DecomposeLoopModel instance
   * \param topoConLevelBefore             Level of schedule restriction to
   *                                       apply on Ops before the \c LoopOp
   * \param topoConLevelLoop               Level of schedule restriction
   *                                       to apply on Ops inside the \c LoopOp
   * \param topoConLevelAfter              Level of schedule restriction to
   *                                       apply on Ops after the \c LoopOp
   * \param computeLikeExchangeStrategies  Exchange strategies that
   *                                       should not be considered for
   *                                       overlapped IO
   */
  DecomposeLoopModel(
      DecomposeTopoConLevel topoConLevelBefore,
      DecomposeTopoConLevel topoConLevelLoop,
      DecomposeTopoConLevel topoConLevelAfter,
      const std::set<ExchangeStrategy> &computeLikeExchangeStrategies);

  virtual ~DecomposeLoopModel() {}

  /**
   * How many loop iterations should be unrolled for this model. The `LoopOp`
   * iteration count will be reduced by the value of `getUnrollFactor`.
   * \return Number of iterations to unroll (before and after the loop
   *         combined).
   */
  virtual int getUnrollFactor() const = 0;

  /**
   * Given an Op type, returns the schedule position, which is used for conflict
   * checking of \c Op classifications and topological constraints.
   * \param type      \c Op type as classified by this model
   * \param iteration The apparent loop iteration (see \ref DecomposeLoops).
   * \return          schedule position
   */
  virtual int typeToPosition(DecomposeLoopOpTypeWrapper type,
                             LoopIteration iteration) const = 0;

  /**
   * Get the group of the type. The grouping is used for getModelString, such
   * that each group is placed in a separate section. Sections are visualized
   * by horizontal dividers. Typically, the types are grouped into sections
   * according to e.g. the pipeline stage.
   * \param type \c Op type as classified by this model
   * \return     Integer identifier of the group
   */
  virtual int getTypeGroup(DecomposeLoopOpTypeWrapper type) const;

  /**
   * Given an Op type and unroll index, get the apparent iteration.
   * The apparent iteration signifies which iteration of the loop (before
   * unrolling) the operation (after unrolling) computes (see \ref
   * DecomposeLoops).
   *
   * \param type        \c Op type as classified by this model
   * \param unrollIndex The \c unrollIndex of the \c Op
   * \return            The apparent iteration of this \c Op type and unroll
   *                    index
   */
  virtual LoopIteration getApparentIteration(DecomposeLoopOpTypeWrapper type,
                                             int unrollIndex) const = 0;

  /**
   * Given an Op type and unroll index, returns if the operation occurs
   * before the loop
   * \param type        \c Op type as classified by this model
   * \param unrollIndex The \c unrollIndex of the \c Op
   * \return            True if the \c Op occurs before the \a Loop after
   *                    unrolling
   */
  virtual bool isBeforeLoop(DecomposeLoopOpTypeWrapper type,
                            int unrollIndex) const = 0;

  /**
   * Returns true if \a A cannot be scheduled before \a B
   * \param iterA The apparent iteration of the \c Op expected to occur before B
   * \param iterB The apparent iteration of the \c Op expected to occur after A
   * \param typeA The type of the \c Op expected to occur before B
   * \param typeB The type of the \c Op expected to occur after A
   * \return      True if \c Op A cannot be scheduled before \c Op B
   */
  bool hasDependencyConflict(LoopIteration iterA,
                             LoopIteration iterB,
                             DecomposeLoopOpTypeWrapper typeA,
                             DecomposeLoopOpTypeWrapper typeB) const;

  /**
   * \return The topological constraint level to be applied to \a Ops unrolled
   *         before the loop (see \ref DecomposeTopoConLevel).
   */
  DecomposeTopoConLevel getTopoConLevelBefore() const {
    return topoConLevelBefore;
  }

  /**
   * \return The topological constraint level to be applied to \a Ops staying
   *         within the loop (see \ref DecomposeTopoConLevel).
   */
  DecomposeTopoConLevel getTopoConLevelLoop() const { return topoConLevelLoop; }

  /**
   * \return The topological constraint level to be applied to \a Ops unrolled
   *         after the loop (see \ref DecomposeTopoConLevel).
   */
  DecomposeTopoConLevel getTopoConLevelAfter() const {
    return topoConLevelAfter;
  }

  /**
   * `Compute like exchange strategies` describe an \c ExchangeStrategy that
   * does not need to be considered for overlapping IO and compute. If an IO
   * operation has a strategy assigned that is returned by this function, then
   * the operation can be treated as `Compute` rather than `IO`.
   *
   * Typically, `ExchangeStrategy::JustInTime` does not require overlapped IO.
   * (see \ref ExchangeStrategy).
   *
   * \return Exchange strategies that can be treated as
   *         \c DecomposeLoopOpTypeEnum::Compute instead of IO
   */
  std::set<ExchangeStrategy> getComputeLikeExchangeStrategies() const {
    return computeLikeExchangeStrategies;
  }

  /**
   * Classifies Ops into different types that are relevant to skewed unrolling
   * (see \ref DecomposeLoops) by means of fix-point iteration. Ops are
   * re-assigned an \c OpType based on the other operations until none changes
   * it's type anymore.
   *
   * If the assignment does not change anymore, the fix-point iteration is
   * relaxed twice:
   * - allowSeeding
   * - allowDelaying
   * (see \ref getDecomposeLoopOpType).
   *
   * \param subgraph \c LoopOp subgraph
   * \return         Mapping from every \c Op to the appropriate classification
   *                 type (see e.g. \ref DecomposeLoopOpTypeEnum).
   */
  virtual std::map<Op *, DecomposeLoopOpTypeWrapper, POpCmp>
  classifyOperations(Graph &subgraph) const;

  /**
   * Classify the OpType of an Op. This corresponds to a single step of the
   * fix-point iteration (see \ref classifyOperations).
   * \param opToDecomposeLoopOpType      The existing mapping of \c Op to type
   * \param op            Op to classify
   *
   * \param allowSeeding  Allow choosing an \c OpType even if there is nothing
   *                      to infer the correct type from. Should be used when
   *                      the assigned types do not change anymore, but some
   *                      operations still don't have a type associated with
   *                      them.
   *
   * \param allowDelaying Allow choosing an \c OpType that can be scheduled
   *                      later, if it is beneficial to overlapping pipeline
   *                      stages or IO and compute
   *                      (such as changing a AuxiliaryBefore to a Compute).
   *                      Should be used when every Op has a type assigned in
   *                      order to optimize the classification.
   *
   * \return              The type chosen for the Op, or no type if none can be
   *                      determined.
   */
  DecomposeLoopOpTypeWrapper virtual getDecomposeLoopOpType(
      const std::map<Op *, DecomposeLoopOpTypeWrapper, POpCmp>
          &opToDecomposeLoopOpType,
      Op *op,
      bool allowSeeding,
      bool allowDelaying) const = 0;

  /**
   * Helper to get all valid \c Op types for the model.
   * \return Set of all \c Op types that can be used for
   *         \c getModelReadableString
   */
  virtual std::set<DecomposeLoopOpTypeWrapper>
  getDecomposeLoopOpTypesToCheck() const = 0;

  /**
   * Generates a string representation of the model that is human readable
   */
  std::string getModelString();

private:
  DecomposeTopoConLevel topoConLevelBefore;
  DecomposeTopoConLevel topoConLevelLoop;
  DecomposeTopoConLevel topoConLevelAfter;
  std::set<ExchangeStrategy> computeLikeExchangeStrategies;
};

std::ostream &operator<<(std::ostream &os, const DecomposeLoopModel &);

/**
 * Base class for \a IO based (i.e. separating compute and IO related operations
 * by means of \ref DecomposeLoopOpTypeEnum) loop decomposition.
 */
class DecomposeLoopIOModel : public DecomposeLoopModel {
public:
  DecomposeLoopIOModel() {}
  /**
   * DecomposeLoopIOModel instance.
   * \param topoConLevelBefore             Level of schedule restriction to
   *                                       apply on Ops before the \c LoopOp
   * \param topoConLevelLoop               Level of schedule restriction
   *                                       to apply on Ops inside the \c LoopOp
   * \param topoConLevelAfter              Level of schedule restriction to
   *                                       apply on Ops after the \c LoopOp
   * \param computeLikeExchangeStrategies  Exchange strategies that
   *                                       should not be considered for
   *                                       overlapped IO
   */
  DecomposeLoopIOModel(
      DecomposeTopoConLevel topoConLevelBefore,
      DecomposeTopoConLevel topoConLevelLoop,
      DecomposeTopoConLevel topoConLevelAfter,
      const std::set<ExchangeStrategy> &computeLikeExchangeStrategies);

  DecomposeLoopOpTypeEnum unwrap(const DecomposeLoopOpTypeWrapper &w) const {
    return w.getType<DecomposeLoopOpIOOverlapType>()->getType();
  }

  std::set<DecomposeLoopOpTypeWrapper>
  getDecomposeLoopOpTypesToCheck() const override;

  DecomposeLoopOpTypeWrapper virtual getDecomposeLoopOpType(
      const std::map<Op *, DecomposeLoopOpTypeWrapper, POpCmp>
          &opToDecomposeLoopOpType,
      Op *op,
      bool allowSeeding,
      bool allowDelaying) const override;
};

/**
 * Decomposing model that simply unrolls iterations before and after the loop.
 */
class DecomposeLoopUnrollModel : public DecomposeLoopIOModel {
public:
  DecomposeLoopUnrollModel();

  /**
   * DecomposeLoopUnrollModel instance.
   * \param unrollBefore                   Number of iterations to unroll before
   *                                       the loop.
   * \param unrollAfter                    Number of iterations to unroll after
   *                                       the loop.
   * \param topoConLevelBefore             Level of schedule restriction to
   *                                       apply on Ops before the \c LoopOp
   * \param topoConLevelLoop               Level of schedule restriction
   *                                       to apply on Ops inside the \c LoopOp
   * \param topoConLevelAfter              Level of schedule restriction to
   *                                       apply on Ops after the \c LoopOp
   * \param computeLikeExchangeStrategies  Exchange strategies that
   *                                       should not be considered for
   *                                       overlapped IO
   */
  DecomposeLoopUnrollModel(
      int unrollBefore,
      int unrollAfter,
      DecomposeTopoConLevel topoConLevelBefore,
      DecomposeTopoConLevel topoConLevelLoop,
      DecomposeTopoConLevel topoConLevelAfter,
      const std::set<ExchangeStrategy> &computeLikeExchangeStrategies);

  std::string getName() const final override {
    return "DecomposeLoopUnrollModel";
  }

  int getUnrollFactor() const override { return unrollBefore + unrollAfter; }

  int typeToPosition(DecomposeLoopOpTypeWrapper type,
                     LoopIteration iteration) const override;
  LoopIteration getApparentIteration(DecomposeLoopOpTypeWrapper type,
                                     int unrollIndex) const override;
  bool isBeforeLoop(DecomposeLoopOpTypeWrapper type,
                    int unrollIndex) const override;

private:
  int unrollBefore;
  int unrollAfter;
};

/**
 * Decomposing model that does skewed unroll (see \ref DecomposeLoops) of two
 * iterations to produce IO/Compute overlap.
 *
 * \code
 *                       Schedule (number denotes apparent iteration count):
 *                       (example shown is IO overlapped unrolling)
 *                       before  | loop | after
 * AuxiliaryBefore       0..1....|2......|......
 * IoBeforeCompute       .0..1...|..2....|......
 * IoToCompute           ..0...1.|.....2.|......
 * Compute               .....0..|...1...|.2....
 * ComputeToIo,          .......0|......1|...2..
 * IoAfterCompute,       ........|.0.....|1...2.
 * AuxiliaryAfter        ........|....0..|..1..2
 * Overlap:                  ^^    ^^^    ^^
 * \endcode
 * (see \ref DecomposeLoopOpTypeEnum for detailed description)
 *
 * Legend:
 * - The vertical axis lists the Op classification types.
 * - The horizontal axis corresponds to the schedule.
 * - Numbers denote apparent iteration.
 *   A dot (.) means no operation of the type is executed at that schedule
 *   position.
 * - Vertical bars (|) mark the start and end of the loop body.
 *
 */
class DecomposeLoopOverlapModel : public DecomposeLoopIOModel {
public:
  std::string getName() const final override {
    return "DecomposeLoopOverlapModel";
  }

  DecomposeLoopOverlapModel();

  /**
   * DecomposeLoopUnrollModel instance.
   * \param topoConLevelBefore             Level of schedule restriction to
   *                                       apply on Ops before the \c LoopOp.
   * \param topoConLevelLoop               Level of schedule restriction
   *                                       to apply on Ops inside the \c LoopOp.
   * \param topoConLevelAfter              Level of schedule restriction to
   *                                       apply on Ops after the \c LoopOp.
   * \param computeLikeExchangeStrategies  Exchange strategies that
   *                                       should not be considered for
   *                                       overlapped IO.
   */
  DecomposeLoopOverlapModel(
      DecomposeTopoConLevel topoConLevelBefore,
      DecomposeTopoConLevel topoConLevelLoop,
      DecomposeTopoConLevel topoConLevelAfter,
      const std::set<ExchangeStrategy> &computeLikeExchangeStrategies);

  /**
   * For overlapped IO, only unrolling two iterations is sufficient:
   * - One iteration unrolled before the LoopOp
   * - One iteration unrolled after the LoopOp
   * (see \ref DecomposeLoopOpTypeEnum).
   * \return Number of iterations to unroll.
   */
  int getUnrollFactor() const override { return 2; }

  int typeToPosition(DecomposeLoopOpTypeWrapper type,
                     LoopIteration iteration) const override;
  LoopIteration getApparentIteration(DecomposeLoopOpTypeWrapper type,
                                     int unrollIndex) const override;
  bool isBeforeLoop(DecomposeLoopOpTypeWrapper type,
                    int unrollIndex) const override;
};

// clang-format off
/**
 * Decomposing model that does skewed unroll of pipeline stages to produce
 * pipeline stage overlap, with optional IO overlap (see \ref OverlapIO or
 * please see the section on overlapped IO in the PopART user guide).
 *
 * Example 1:
 * ~~~~~~~~~~
 *
 * A loop (represented by {}) with 5 pipeline stages:
 *
 * \code
 * {ps0 - ps1 - ps2 - ps3 - ps4}
 * \endcode
 *
 * would change into
 *
 * first decompose unroll:
 *
 * \code
 * ps0 - { ps1 - ps2 - ps3 - ps4 }
 *       { ps0                   } - ps1 - ps2 - ps3 - ps4
 * \endcode
 *
 * second decompose unroll:
 *
 * \code
 * ps0 - ps1 - { ps2 - ps3 - ps4 }
 *       ps0 - { ps1             } - ps2 - ps3 - ps4
 *             { ps0             } - ps1 - ps2 - ps3 - ps4
 * \endcode
 *
 * third decompose unroll:
 *
 * \code
 *
 * ps0 - ps1 - ps2 - { ps3 - ps4 }
 *       ps0 - ps1 - { ps2       } - ps3 - ps4
 *             ps0 - { ps1       } - ps2 - ps3 - ps4
 *                   { ps0       } - ps1 - ps2 - ps3 - ps4
 * \endcode
 *
 * fourth decompose unroll:
 *
 * \code
 * ps0 - ps1 - ps2 - ps3 - { ps4 }
 *       ps0 - ps1 - ps2 - { ps3 } - ps4
 *             ps0 - ps1 - { ps2 } - ps3 - ps4
 *                   ps0 - { ps1 } - ps2 - ps3 - ps4
 *                         { ps0 } - ps1 - ps2 - ps3 - ps4
 * \endcode
 *
 * Legend:
 * - The x-axis represents the sequential execution of pipeline stages.
 * - The y-axis represents parallel execution of pipeline stages
 *   (parts that can overlap).
 * - {} represents the LoopOp
 * - ps0 - ps4 are the distinct pipeline stages.
 *
 * Note that in the actual algorithm, all unroll steps are done at once.
 * The step-by-ste unrolling is for illustrative purposes only.
 *
 * Example 2:
 * ~~~~~~~~~~
 *
 * Arrangement with overlapped IO for 2 pipeline stages (see \ref OverlapIO
 * please see the section on overlapped IO in the PopART user guide):
 * - Overlapped input IO of \c PipelineStage S is handled like
 *   \c PipelineStage S-1
 *
 * - Overlapped output IO of \c PipelineStage S is handled like
 *   \c PipelineStage S+1
 *
 * - Consequently, up to two new "PipelineStages" and unrolling steps
 *   are added if overlapped IO is used.
 *   The first one (psb) is only added if the first pipeline stage (ps0) does
 *   overlapped input IO (IoBeforeCompute).
 *   The last one (psa) is only added if the last pipeline stage (ps1) does
 *   overlapped output IO (IoAfterCompute).
 *
 * \code
 *              0     1     2       3       4     5     6
 *
 *         0   psb - ps0 - ps1 - { psa }
 *         1         psb - ps0 - { ps1 } - psa
 *         2               psb - { ps0 } - ps1 - psa
 *         3                     { psb } - ps0 - ps1 - psa
 *
 * Legend:
 * - psb     PipelineStage before stage 0 (for handling IO only). Note, this is
 *           an abstract concept. In practice (see below) psa would be
 *           represented by one or more IoBeforeCompute operations!
 * - psa     PipelineStage after stage 1 (for handling IO only) Note, this is
 *           an abstract concept. In practice (see below) psb would be
 *           represented by one or more IoAfterCompute operations!
 * - x-axis  Sequential executions / data paths (0, ..., 6 sequential steps)
 * - y-axis  Parallel / overlapping executions (0, ..., 3 parallel paths)
 *
 *  isPipelineIpuCopy   isComputeLike
 *                  \   /                        cycleStride
 *                   | |                         |----------|
 *                   A B       0   1          2          3          4          5        6
 * AuxiliaryBefore   0 0  ps0  |0..|..1.......|..2.......|..3.......|..........|........|..|
 * IoBeforeCompute   0 0  ps0  |.0.|...1......|...2......|...3......|..........|........|..|
 * IoToCompute       0 0  ps0  |..0|.......1..|.......2..|.......3..|..........|........|..|
 * AuxiliaryBefore   0 1  ps0  |...|0.........|1.........|2.........|3.........|........|..|
 * IoBeforeCompute   0 1  ps0  |...|.0........|.1........|.2........|.3........|........|..|
 * Compute           0 0  ps0  |...|....0.....|....1.....|....2.....|....3.....|........|..|
 * IoAfterCompute    0 1  ps0  |...|.....0....|.....1....|.....2....|.....3....|........|..|
 * AuxiliaryAfter    0 1  ps0  |...|......0...|......1...|......2...|......3...|........|..|
 * ComputeToIo       0 0  ps0  |...|........0.|........1.|........2.|........3.|........|..|
 * Compute           1 0  ps0  |...|.........0|.........1|.........2|.........3|........|..|
 * IoAfterCompute    0 0  ps0  |...|..........|...0......|...1......|...2......|..3.....|..|
 * AuxiliaryAfter    0 0  ps0  |...|..........|......0...|......1...|......2...|.....3..|..|
 * ----------------------------+---+----------+----------+----------+----------+--------+--+
 * AuxiliaryBefore   0 0  ps1  |...|..0.......|..1.......|..2.......|..3.......|........|..|
 * IoBeforeCompute   0 0  ps1  |...|...0......|...1......|...2......|...3......|........|..|
 * IoToCompute       0 0  ps1  |...|.......0..|.......1..|.......2..|.......3..|........|..|
 * AuxiliaryBefore   0 1  ps1  |...|..........|0.........|1.........|2.........|3.......|..|
 * IoBeforeCompute   0 1  ps1  |...|..........|.0........|.1........|.2........|.3......|..|
 * Compute           0 0  ps1  |...|..........|....0.....|....1.....|....2.....|...3....|..|
 * IoAfterCompute    0 1  ps1  |...|..........|.....0....|.....1....|.....2....|....3...|..|
 * AuxiliaryAfter    0 1  ps1  |...|..........|......0...|......1...|......2...|.....3..|..|
 * ComputeToIo       0 0  ps1  |...|..........|........0.|........1.|........2.|......3.|..|
 * Compute           1 0  ps1  |...|..........|.........0|.........1|.........2|.......3|..|
 * IoAfterCompute    0 0  ps1  |...|..........|..........|...0......|...1......|..2.....|3.|
 * AuxiliaryAfter    0 0  ps1  |...|..........|..........|......0...|......1...|.....2..|.3|
 *                             ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^~~~~~~~~~~~~~~~~~~~~~~~
 *                             Ramp up (fill)               LoopOp         Ramp down (flush)
 *
 *
 * A: if true (1), the Op is an IpuCopyOp between pipelines
 * B: if true (1), the Op is a compute-like operation
 *    (IO operation on IO tiles that can be unrolled and scheduled like a
 *     compute operation due to not being required for overlapped IO)
 *
 * Legend:
 * - The vertical axis lists the Op classification types.
 * - The horizontal axis corresponds to the schedule.
 * - Numbers in the chart denote apparent loop iteration of the operation(type).
 *   A dot (.) means no operation of the type is executed at that schedule
 *   position.
 * - Vertical bars (|) mark the start and end of the loop body, and the separate
 *   pipeline ramp up/down stages before and after the loop.
 * - Cycle stride: Defines how many positions two ops of the same type are
 *   apart.
 *
 * (this table is used to derive typeToPosition, getApparentIteration and
 * isBeforeLoop)
 *
 * (see \ref DecomposeLoopOpTypeEnum for details)
 *
 * \endcode
 *
 */
// clang-format on
class DecomposeLoopPipelineModel : public DecomposeLoopModel {
public:
  /**
   * DecomposeLoopPipelineModel instance.
   * The model will unroll the loop maxStage - minStage + 1 times
   * (\ref getUnrollFactor).
   * \param minStage The lower limit for unrolling pipeline stages
   * \param maxStage The upper limit for unrolling pipeline stages
   * \param numStages The number of total pipeline stages
   * \param topoConLevelBefore             Level of schedule restriction to
   *                                       apply on Ops before the \c LoopOp.
   * \param topoConLevelLoop               Level of schedule restriction
   *                                       to apply on Ops inside the \c LoopOp.
   * \param topoConLevelAfter              Level of schedule restriction to
   *                                       apply on Ops after the \c LoopOp.
   * \param computeLikeExchangeStrategies  Exchange strategies that
   *                                       should not be considered for
   *                                       overlapped IO.
   */
  DecomposeLoopPipelineModel(
      int minStage,
      int maxStage,
      int numStages,
      DecomposeTopoConLevel topoConLevelBefore,
      DecomposeTopoConLevel topoConLevelLoop,
      DecomposeTopoConLevel topoConLevelAfter,
      const std::set<ExchangeStrategy> &computeLikeExchangeStrategies);

  int getUnrollFactor() const override;

  int typeToPosition(DecomposeLoopOpTypeWrapper type,
                     LoopIteration iteration) const override;
  LoopIteration getApparentIteration(DecomposeLoopOpTypeWrapper type,
                                     int unrollIndex) const override;
  bool isBeforeLoop(DecomposeLoopOpTypeWrapper type,
                    int unrollIndex) const override;

  DecomposeLoopOpPipelineType
  unwrap(const DecomposeLoopOpTypeWrapper &w) const {
    return *w.getType<DecomposeLoopOpPipelineType>();
  }

  std::set<DecomposeLoopOpTypeWrapper>
  getDecomposeLoopOpTypesToCheck() const override;

  int getTypeGroup(DecomposeLoopOpTypeWrapper type) const override;

  DecomposeLoopOpTypeWrapper virtual getDecomposeLoopOpType(
      const std::map<Op *, DecomposeLoopOpTypeWrapper, POpCmp>
          &opToDecomposeLoopOpType,
      Op *op,
      bool allowSeeding,
      bool allowDelaying) const override;

private:
  std::pair<int, int> getAdjustedPipelineStageAndIterationsBeforeLoop(
      DecomposeLoopOpPipelineType type) const;

  int minStage;
  int maxStage;
  int numStages;
};

/**
 * Transform that generically decomposes/unrolls loop iterations to:
 * - Unroll \c LoopOp iterations in general
 * - Arrange \a IO \c Ops to enable overlap between IO and compute tiles
 * - Arrange \c Ops \c PipelineStages to enable overlap between
 *   \c PipelineStages
 *
 * If we want to unroll a loop by a factor of 2, each \c Op that existed in the
 * loop needs 3 instances, denoted as 0, 1 and 2, one per apparent iteration.
 * If we want to unroll such that iterations can partially overlap
 * (IO and compute overlap),
 * we can't generally, for all operations, place 0 before the loop,
 * 2 after loop and 1 during the loop (see skewed unrolling below), because
 * this would not lead to overlap between either pipeline stages or IO and
 * compute operations.
 *
 * Rather, we classify Ops (see \ref DecomposeLoopOpTypeEnum), according to
 * their data, topological dependencies and the tile set they are running on,
 * into one of the categories.
 * The available categories depend on the DecomposeLoopModel implementation.
 * We can then shuffle the operations to before, during and after the loop
 * accordingly.
 * Note that every operation is cloned 2 extra times (for an unroll factor of
 * 2), but the original operation in the loop remains.
 *
 * However, the "apparent iteration" (iteration that the Op instance corresponds
 * to in the LoopOp before unrolling) has changed.
 *
 * The number of apparent iterations in total is always the unroll factor
 * (counting all iterations before and after the loop) plus one iteration for
 * the loop itself:
 *
 * num_apparent_iterations = unroll_factor + 1
 *
 * In loop iteration n, the Ops (depending on classification) now correspond to
 * iterations i (0), i+1 (1) and i+2 (2) respectively.
 * The Ops unrolled before the loop process iterations 0 (0) and 1 (1)
 * The Ops unrolled after the loop process iterations n-1 (1) and n (2)
 * (where (0) (1) and (2) correspond to the cloned operations)
 *
 * As an example for \a apparent \a iteration:
 * Before unrolling, there is an operation in a loop (denoted as {}):
 *    { Op }
 *
 * If we unroll by a factor of 2, the operation is cloned into the parent graph
 * twice,and there are different possible arrangements, depending on how we skew
 * the unrolling:
 *
 *    a.) { Op } - Op0 - Op1
 *
 *      In this case:
 *      Op  - unrollIndex  -1 - apparent iteration 0 - before loop: no
 *      Op0 - unrollIndex   0 - apparent iteration 1 - before loop: no
 *      Op1 - unrollIndex   1 - apparent iteration 2 - before loop: no
 *
 *      (use case example: if Op is a HostStoreOp that should do overlapped IO
 *       with compute (such as a MatMulOp))
 *
 *    b.) Op0 - { Op } - Op1
 *
 *      In this case:
 *      Op  - unrollIndex  -1 - apparent iteration 1 - before loop: no
 *      Op0 - unrollIndex   0 - apparent iteration 0 - before loop: yes
 *      Op1 - unrollIndex   1 - apparent iteration 2 - before loop: no
 *
 *      (use case example: if Op is a MatMulOp that should do overlapped compute
 *       with IO (such as HostloadOp and HostStoreOp))
 *
 *    c.) Op0 - Op1 - { Op }
 *
 *      In this case:
 *      Op  - unrollIndex  -1 - apparent iteration 2 - before loop: no
 *      Op0 - unrollIndex   0 - apparent iteration 0 - before loop: yes
 *      Op1 - unrollIndex   1 - apparent iteration 1 - before loop: yes
 *
 *      (use case example: if Op is a HostLoadOp that should do overlapped IO
 *       with compute (such as a MatMulOp))
 *
 * Use case example:
 *
 * \code
 *  HostLoadOp0 HostLoadOp1 { HostLoadOp  }
 *              MatMulOp0   { MatMulOp    } MatMulOp1
 *                          { HostStoreOp } HostStoreOp0 HostStoreOp1
 *              ^^^^^^^^^^^   ^^^^^^^^^^^   ^^^^^^^^^^^^
 *              overlap       overlap       overlap
 * \endcode
 *
 * { } denotes the LoopOp
 *
 * Where the data dependencies are:
 * HostLoadOp0 -> MatMulOp0 -> HostStoreOp
 * HostLoadOp1 -> MatMulOp  -> HostStoreOp0
 * HostLoadOp  -> MatMulOp1 -> HostStoreOp1

 *
 * This skew is controlled by the decomposition model (see \ref
 DecomposeLoopOpTypeEnum for details). If the model is unrolling
 * pipeline stages, for example, each stage will be skewed differently
 * (see \ref DecomposeLoopPipelineModel).
 */
class DecomposeLoops : public Transform {
public:
  static std::size_t id();

  DecomposeLoops() : Transform() {}
  virtual ~DecomposeLoops() override {}

  /**
   * Decomposes all \c LoopOps in the \c graph using the standard model
   * of loop decomposition  (which is \c DecomposeLoopOverlapModel())
   * \param graph Graph containing the \c LoopOp to decompose
   * \return true If apply is successful. An error will be thrown if not.
   */
  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "DecomposeLoops"; }

  /**
   * Decompose a loop with a custom \c DecomposeLoopModel
   * \param graph \c graph containing the \c LoopOp to decompose
   * \param loopOp \c LoopOp to decompose
   * \param model \c DecomposeLoopModel to apply
   */
  void decomposeLoop(Graph &graph,
                     LoopOp *loopOp,
                     const DecomposeLoopModel &model) const;

  /**
   * Check if an \c Op should be classified as compute. The condition is that
   * the operation is on compute tiles.
   * \param op Op to check
   * \return true if it is a Compute \c Op
   */
  static bool isComputeOp(Op *op);

  /**
   * Checks if an \c Op is an IO operation. The condition is that the operation
   * is one of `HostLoadOp`, `HostStoreOp`, `RemoteLoadOp`, `RemoteStoreOp`,
   * `MultiExchangeOp`.
   * \param op Op to check
   * \return true if it is an IO \c Op
   */
  static bool isIOOp(Op *op);

  /**
   * Checks if an \c Op is classified as IO, and executes on IO tiles, but
   * should  still be handled like a \a compute operation
   * (as in, classified, unrolled and scheduled as
   * `DecomposeLoopOpTypeEnum::Compute`) (instead of an IO operation that should
   * overlap with compute (classified `DecomposeLoopOpTypeEnum::IoBeforeCompute`
   * or `DecomposeLoopOpTypeEnum::IoAfterCompute`)).
   *
   * Operations should be handled like compute instead of IO operations when
   * they are not required to overlap with compute.
   *
   * \param computeLikeStrategies ExchangeStrategy that should be considered as
   *                              \a compute
   * \param op                    Op to check
   * \return                      True if it is a compute \c Op
   */
  static bool
  isComputeLikeIOOp(std::set<ExchangeStrategy> computeLikeStrategies, Op *op);

private:
  class DecomposeLoopHelper {
  public:
    DecomposeLoopHelper(Ir &ir_,
                        Graph &graph_,
                        const DecomposeLoopModel &model_,
                        LoopOp *loopOp,
                        Graph &subgraph_);

    // Functions performing the decomposition

    /**
     * Adjust loop so it becomes decomposable.
     * Situations that need adjusting:
     * - If the LoopOp outputs a constant tensor
     * - If the LoopOp outputs an input tensor
     * In these cases, there is no way to ascertain which apparent iteration
     * the tensor belongs to. An identity operation
     * (which will then belong to an iteration) is inserted as a separator
     * to ensure decomposing is possible.
     */
    void adjustLoop();

    /**
     * Create a backup of the original subgraph and LoopOp for lookup purposes.
     */
    void createBackupStructure();

    /**
     * Fetch schedule of the loop subgraph and classify all operations according
     * to the DecomposeLoopModel.
     */
    void prepare();

    /**
     * Clone the loop subgraph operations for each unrolled iteration.
     */
    void clone();

    /**
     * Create the initial input tensors for the unrolled operations.
     */
    void hookUpBeforeLoopInitialize();

    /**
     * Iterate through all unrolled operations that occur before the loop and
     * connect their output tensors.
     */
    void hookUpBeforeLoopOutputs();

    /**
     * Iterate through all unrolled operations that occur before the loop and
     * connect their input tensors.
     */
    void hookUpBeforeLoopInputs();

    /**
     * Calls:
     * - \ref hookUpBeforeLoopInitialize()
     * - \ref hookUpBeforeLoopOutputs()
     * - \ref hookUpBeforeLoopInputs()
     */
    void hookUpBeforeLoop();

    /**
     * Create a new LoopOp output TensorId, if required, or return the original
     * LoopOp output TensorId, if the iterationForOutput matches the last loop
     * iteration (before unrolling).
     * \param originalInputId     The original Op input id before unrolling,
     *                            which is used to derive the tensor name.
     * \param backupTensor        The backup tensor associated with the
     *                            originalInputId, which is used to determine
     *                            if the tensor is a loop body subgraph input
     *                            tensor.
     * \param iterationForOutput  The apparent loop iteration associated with
     *                            the LoopOp output.
     * \return                    TensorId for the new LoopOp output.
     */
    TensorId getThreadThroughLoopOutputId(TensorId originalInputId,
                                          Tensor *backupTensor,
                                          LoopIteration iterationForOutput);

    /**
     * Connect loop input -> loop output (thread through loop) until current
     * (apparent) iteration is inside the loop.
     *
     * The case addressed here is when the graph before unrolling is, for
     * example:
     * {P-C}
     *
     * (one producer (P) and one consumer (C) inside a loop)
     *
     * And after unrolling:
     * 1.) There are one or more producers before the loop after
     *     unrolling (P0, P1).
     * 2.) One consumer is inside the loop (C0)
     * 3.) Zero or more consumers (C1, C2) are after the loop
     *
     * C0 and P2 correspond to the same Op instances as used before
     * unrolling, so C0 == C and P2 == P. P0, P1, C1, C2 are cloned
     * from P and C.
     *
     * P0----{-C0 }
     *    P1-{----}-C1
     *       { P2-}----C2
     *
     * This needs to loop-carry the tensors correctly:
     * - P1 output is fed back to P0 input of the loop
     * - P2 output is fed back to P1 input of the loop
     * - P1 and P2 output are outputs of the LoopOp when it terminates
     *
     * \param originalInputId   The loop subgraph tensor for which to create
     *                          through a replacement tensor.
     * \param apparentIteration The iteration at which to stop threading through
     *                          the LoopOp.
     */
    void threadThroughLoop(TensorId originalInputId,
                           LoopIteration apparentIteration);

    /**
     * Iterate through all operations inside the loop body, disconnect their
     * original outputs and reconnect their new outputs.
     */
    void hookUpInLoopOutputs();

    /**
     * Iterate through all operations inside the loop body, disconnect their
     * original inputs and reconnect their new inputs.
     */
    void hookUpInLoopInputs();

    /**
     * Calls:
     * - \ref hookUpInLoopOutputs()
     * - \ref hookUpInLoopInputs()
     */
    void hookUpInLoop();

    /**
     * Iterate through all unrolled operations that occur after the loop and
     * connect their output tensors.
     */
    void hookUpAfterLoopOutputs();

    /**
     * Iterate through all unrolled operations that occur after the loop and
     * connect their input tensors.
     */
    void hookUpAfterLoopInputs();

    /**
     * Calls:
     * - \ref hookUpAfterLoopOutputs()
     * - \ref hookUpAfterLoopInputs()
     */
    void hookUpAfterLoop();

    /**
     * Add additional topological constraints to optimize the schedule for
     * operation overlap.
     */
    void fixTopoCons();

    /**
     * Remove topo cons that are no longer required
     */
    void removeTopoConsAcrossApparentIterations(
        const std::map<int, std::vector<Op *>> &apparentIterationMap);

    /**
     * Constrain topo cons before the LoopOp.
     */
    void fixTopoConsBeforeLoop(
        const std::map<int, std::vector<Op *>> &beforeLoopBins);

    /**
     * Constrain topo cons inside the LoopOp.
     */
    void fixTopoConsInsideLoop(
        const std::map<int, std::vector<Op *>> &insideLoopBins);

    /**
     * Constrain topo cons after the LoopOp.
     */
    void
    fixTopoConsAfterLoop(const std::map<int, std::vector<Op *>> &afterLoopBins);

    /**
     * Promote the aliased (which inputs the loop aliases to it's outputs)
     * properties of the loop.
     * \param backupOpInIndex Original input index for which to promote aliases,
     *                        if they exist.
     */
    void promoteAliases(InIndex backupOpInIndex);

    /**
     * Promote the modifies (which inputs the loop inplace modifies)
     * properties of the loop.
     * \param backupOpInIndex Original input index for which to promote
     *                        modifies, if they exist.
     */
    void promoteModifies(InIndex backupOpInIndex);

    /**
     * Reduce the loop iteration count by the unroll factor.
     * Promote the aliased (which inputs the loop aliases to it's outputs) and
     * modifies (which inputs the loop inplace modifies) properties of the loop.
     */
    void updateLoop();

    /**
     * Remove the backup of the original subgraph and LoopOp that were used for
     * lookup purposes.
     */
    void cleanup();

    // Helper functions
    Graph &getBackupGraph() const;
    Op *getBackupOp(Op *op) const;

    LoopIteration getApparentIteration(Op *op, int unrollIndex) const;
    bool isBeforeLoop(Op *op, int unrollIndex) const;
    bool isLastBeforeLoop(Op *op, int unrollIndex) const;

    bool addTopoCon(Graph &graph, Op *before, Op *after, bool tied) const;

    Ir &ir;

    // Parent graph to decompose into
    Graph &graph;

    // Model used for decomposition
    const DecomposeLoopModel &model;

    // LoopOp to decompose
    LoopOp *loopOp;

    // Subgraph to decompose from
    Graph &subgraph;

    // Schedule of the subgraph
    std::vector<Op *> schedule;

    // Backup of the original subgraph for lookup purposes
    GraphId backupGraphId;
    ClonedGraphMaps backupMaps;
    LoopOp *backupLoopOp;
    std::vector<Op *> backupSchedule;

    // Op type lookup
    std::map<DecomposeLoopOpTypeWrapper, std::vector<Op *>> opsByType;
    std::map<Op *, DecomposeLoopOpTypeWrapper, POpCmp> opToDecomposeLoopOpType;

    // Clone operations and map from clones to originals
    std::map<Op *, std::vector<Op *>> clones;
    std::map<Op *, Op *> originals;

    // Map of tensors and loop iterations
    LoopTensorMap beforeLoopTensorIterMap;
    LoopTensorMap loopTensorIterMap;
    LoopTensorMap afterLoopTensorIterMap;
  };
};

} // namespace popart

#endif
