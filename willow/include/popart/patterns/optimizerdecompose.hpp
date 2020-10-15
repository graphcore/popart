// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_OPTIMIZERDECOMPOSE_PATTERN_HPP
#define GUARD_NEURALNET_OPTIMIZERDECOMPOSE_PATTERN_HPP

#include <popart/onnxutil.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/optimizervalue.hpp>
#include <popart/patterns/patterns.hpp>

namespace popart {

class OptimizerDecompose : public PreAliasPattern {
protected:
  TensorInfo addStateTensor(Graph &graph,
                            const TensorId &tensorId,
                            const Shape &shape,
                            const DataType &type) const;

  template <typename T>
  void addStateTensor(Graph &graph,
                      const TensorId &tensorId,
                      const TensorInfo info) const;

  // Store optimizer state tensor
  void storeTensor(Ir &ir, TensorId id) const;

  // ADd accl Op and updated tensor
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
                     TensorId accumId,
                     TensorId gradIntoAccumId,
                     bool accumReduce,
                     bool gradAccum) const;

  // Update gradient accumulator
  void accumUpdate(Graph &graph,
                   Op *combo,
                   std::vector<Op *> beforeOps,
                   TensorId accumId) const;

  // Gradient reduction
  TensorId gradReduce(Graph &graph, Op *combo, TensorId weightGradId) const;

  // Gradient casting
  TensorId gradCast(Graph &graph,
                    Op *combo,
                    TensorId gradIntoAcclId,
                    bool gradAccum) const;
  // Gradient unscaling
  TensorId gradUnscale(Graph &graph,
                       Op *combo,
                       const OptimizerValue &gs,
                       TensorId gsId,
                       TensorId gradIntoAcclId,
                       bool gradAccum) const;

  // L2 regularization
  TensorId regularizeL2(Graph &graph,
                        Op *combo,
                        const OptimizerValue &wd,
                        TensorId wdId,
                        TensorId weightId,
                        TensorId gradIntoAcclId,
                        bool gradAccum) const;
};

} // namespace popart

#endif
