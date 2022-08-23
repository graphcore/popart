// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_OPTIMIZERDECOMPOSE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_OPTIMIZERDECOMPOSE_HPP_

#include <string>
#include <utility>
#include <vector>
#include <popart/op/accumulate.hpp>
#include <popart/optimizervalue.hpp>
#include <popart/variablesettings.hpp>

#include "popart/datatype.hpp"
#include "popart/names.hpp"
#include "popart/patterns/pattern.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
class Graph;
class Op;

class OptimizerDecompose : public PreAliasPattern {
protected:
  TensorInfo addStateTensor(Graph &graph,
                            const TensorId &tensorId,
                            const Shape &shape,
                            const DataType &type,
                            const VariableSettings &varset,
                            float initValue = 0.0) const;

  /// If the variable we are implementing the optimiser step for
  /// has a replica grouping, the info we pass here
  /// is the replica shape, not the host shape (so without the groups dim).
  template <typename T>
  void addStateTensor(Graph &graph,
                      const TensorId &tensorId,
                      const TensorInfo info,
                      const VariableSettings &varset,
                      float initValue = 0.0) const;

  // Add accl Op and updated tensor
  // Runs after gradient accumulation (if enabled) has completed
  std::pair<Op *, TensorId> accl(Graph &graph,
                                 Op *combo,
                                 TensorId acclId,
                                 TensorId gradIntoAcclId,
                                 AccumulationType type,
                                 OptimizerValue value,
                                 TensorId valueTensorId,
                                 std::string acclName,
                                 bool gradAccum) const;

  // Gradient accumulation
  TensorId gradAccum(Graph &graph,
                     Op *combo,
                     TensorId weightId,
                     TensorId accumId,
                     TensorId gradIntoAccumId,
                     bool accumReduce,
                     TensorId outputId = "") const;

  // Reset all values of accumulutor with TensorID
  // Transfers the name and properties from Op, combo and schedules the op to
  // take place once beforeOps have run (and after gradient accumulation has
  // taken place.)
  Op *zeroAccumulator(Graph &graph,
                      Op *combo,
                      std::vector<Op *> beforeOps,
                      TensorId accumId) const;

  // Gradient reduction
  TensorId gradReduce(Graph &graph,
                      Op *combo,
                      TensorId weightId,
                      TensorId weightGradId,
                      TensorId outputId = "") const;

  // Gradient casting
  // Runs after gradient accumulation (if enabled) has completed
  TensorId gradCast(Graph &graph,
                    Op *combo,
                    TensorId gradIntoAcclId,
                    bool gradAccum) const;

  // Gradient unscaling
  // Runs after gradient accumulation (if enabled) has completed
  TensorId gradUnscale(Graph &graph,
                       Op *combo,
                       const OptimizerValue &gs,
                       TensorId gsId,
                       TensorId gradIntoAcclId,
                       bool gradAccum) const;

  // L2 regularization
  // Runs after gradient accumulation (if enabled) has completed
  TensorId regularizeL2(Graph &graph,
                        Op *combo,
                        const OptimizerValue &wd,
                        TensorId wdId,
                        TensorId weightId,
                        TensorId gradIntoAcclId,
                        bool gradAccum) const;

  bool runningMeanReduction(Graph &graph) const;

private:
  // Returns the counter id based off attributes of combo.
  // Currently will have 1 counter per ipu, per pipelineStage.
  TensorId getCounterId(Op *combo) const;

  std::pair<Op *, TensorId>
  counterIncrement(Graph &graph, Op *combo, TensorId counterId) const;
  std::pair<Op *, TensorId>
  counterReset(Graph &graph, Op *combo, TensorId counterId) const;

  // Add counter increment and zero if they don't exist.
  // return increment operation and counter tensor.
  std::pair<Op *, TensorId> findOrCreateRunningMeanCounter(Graph &graph,
                                                           Op *combo) const;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_OPTIMIZERDECOMPOSE_HPP_
