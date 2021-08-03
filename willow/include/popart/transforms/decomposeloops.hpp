// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DECOMPOSELOOPS_HPP
#define GUARD_NEURALNET_DECOMPOSELOOPS_HPP

#include <popart/op/loop.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

using LoopIteration = int;
using LoopTensorMap = std::map<std::pair<TensorId, LoopIteration>, TensorId>;

// If we want to unroll a loop by a factor of 2, each Op that existed in the
// loop needs 3 instances: 0, 1 and 2
// If we want to unroll such that iterations can partially overlap
// (IO and compute overlap),
// we can't place 0 before loop, 2 after loop and 1 during the loop.
// Rather, we classify Ops, according to their data & topological dependencies
// and tile set they are running on, into one of 7 categories.
// We can then shuffle the operations to before, during and after the loop
// accordingly.
// Note that every operation is cloned 2 extra times, but the original
// operation in the loop remain. However, the "apparent iteration" has changed.
// In loop iteration n, the Ops (depending on classification) now process
// iterations i (0), i+1 (1) and i+2 (2) respectively.
// The Ops unrolled before the loop process iterations 0 (0) and 1 (1)
// The Ops unrolled after the loop process iterations n-1 (1) and n (2)
enum class DecomposeLoopOpType {
  //                   // Schedule (number denotes apparent iteration count):
  //                   // before  | loop  | after
  //                   // 00000000 0011111 111112
  //                   // 01234567 8901234 567890
  AuxiliaryBefore = 0, // 0..1....|2......|......
  IoBeforeCompute,     // .0..1...|..2....|......
  IoToCompute,         // ..0...1.|.....2.|......
  Compute,             // .....0..|...1...|.2....
  ComputeToIo,         // .......0|......1|...2..
  IoAfterCompute,      // ........|.0.....|1...2.
  AuxiliaryAfter,      // ........|....0..|..1..2
  // Overlap:          //     ^^    ^^^    ^^
  N
};

// Extent to which topo cons should be applied after decomposition
enum class DecomposeTopoConLevel {
  None, // Do not restrict schedule with topocons
  Full, // Fully restrict schedule with topocons
  N     // Number of levels
};

std::ostream &operator<<(std::ostream &os, const DecomposeLoopOpType &);

class DecomposeLoopModel {
public:
  DecomposeLoopModel();
  DecomposeLoopModel(
      DecomposeTopoConLevel topoConLevelBefore_,
      DecomposeTopoConLevel topoConLevelLoop_,
      DecomposeTopoConLevel topoConLevelAfter_,
      const std::set<ExchangeStrategy> &computeLikeExchangeStrategies_);

  virtual ~DecomposeLoopModel() {}

  // Given an Op type, returns the schedule position
  virtual int typeToPosition(DecomposeLoopOpType type,
                             LoopIteration iteration) const = 0;

  // Given an Op type and unroll index, get the apparent iteration
  virtual LoopIteration getApparentIteration(DecomposeLoopOpType type,
                                             int unrollIndex) const = 0;

  // Given an Op type and unroll index, returns if the operation occurs
  // before the loop
  virtual bool isBeforeLoop(DecomposeLoopOpType type,
                            int unrollIndex) const = 0;

  // Retruns true if "from" cannot be scheduled before "to"
  bool hasDependencyConflict(LoopIteration iterFrom,
                             LoopIteration iterTo,
                             DecomposeLoopOpType typeFrom,
                             DecomposeLoopOpType typeTo) const;

  DecomposeTopoConLevel getTopoConLevelBefore() const {
    return topoConLevelBefore;
  }

  DecomposeTopoConLevel getTopoConLevelLoop() const { return topoConLevelLoop; }

  DecomposeTopoConLevel getTopoConLevelAfter() const {
    return topoConLevelAfter;
  }

  std::set<ExchangeStrategy> getComputeLikeExchangeStrategies() const {
    return computeLikeExchangeStrategies;
  }

private:
  DecomposeTopoConLevel topoConLevelBefore;
  DecomposeTopoConLevel topoConLevelLoop;
  DecomposeTopoConLevel topoConLevelAfter;
  std::set<ExchangeStrategy> computeLikeExchangeStrategies;
};

class DecomposeLoopUnrollModel : public DecomposeLoopModel {
public:
  DecomposeLoopUnrollModel() {}
  int typeToPosition(DecomposeLoopOpType type,
                     LoopIteration iteration) const override;
  LoopIteration getApparentIteration(DecomposeLoopOpType type,
                                     int unrollIndex) const override;
  bool isBeforeLoop(DecomposeLoopOpType type, int unrollIndex) const override;
};

class DecomposeLoopOverlapModel : public DecomposeLoopModel {
public:
  DecomposeLoopOverlapModel() {}
  DecomposeLoopOverlapModel(
      DecomposeTopoConLevel topoConLevelBefore_,
      DecomposeTopoConLevel topoConLevelLoop_,
      DecomposeTopoConLevel topoConLevelAfter_,
      const std::set<ExchangeStrategy> &computeLikeExchangeStrategies_);
  int typeToPosition(DecomposeLoopOpType type,
                     LoopIteration iteration) const override;
  LoopIteration getApparentIteration(DecomposeLoopOpType type,
                                     int unrollIndex) const override;
  bool isBeforeLoop(DecomposeLoopOpType type, int unrollIndex) const override;
};

// DecomposeLoops:
// Transform that decomposes/unrolls loop iterations to enable overlap between
// IO and compute tiles
class DecomposeLoops : public Transform {
public:
  static std::size_t id();

  DecomposeLoops() : Transform() {}
  virtual ~DecomposeLoops() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "DecomposeLoops"; }

  void decomposeLoop(Graph &graph,
                     LoopOp *loopOp,
                     const DecomposeLoopModel &model) const;

private:
  bool
  addTopoConConditionally(Graph &graph, Op *before, Op *after, bool tied) const;

  DecomposeLoopOpType
  getType(const DecomposeLoopModel &model,
          const std::map<Op *, DecomposeLoopOpType> &opToType,
          Op *op,
          std::set<DecomposeLoopOpType> prevTypes) const;
};

} // namespace popart

#endif
