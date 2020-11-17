// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LOOP_HPP
#define GUARD_NEURALNET_LOOP_HPP

#include <popart/op.hpp>
#include <popart/op/subgraph.hpp>

namespace popart {

// Loop operation construct
//
// Loop op and body inputs:
// 0    trip count
// 1    termination condition <--.
// 2    explicit input <---------|--.
// ..   ..             ..        |  |
// M    explicit input <---------|--|--.
// M+1  implicit input           |  |  |
// ..                            |  |  |
// N    implicit input           |  |  |
//                               |  |  | Loop carried dependencies
// Loop body outputs:            |  |  |
// 0     termination condition --'  |  |
// 1     output --------------------+  |
// ..           ..                  |  |
// M-1   output --------------------|--+
//                                  |  |
// Loop op outputs:                 |  |
// 0     output <-------------------'  |
// ..                                  |
// M-2   output <----------------------'
//
// Explicit inputs are loop-carried inputs that are copied into the loop body,
// copied from loop body output to loop body input after each iteration,
// and copied out of the loop into the parent graph as loop outputs on
// loop termination.
//
// Implicit inputs are only copied into the loop body once, and are implicit
// because in the ONNX specification, the loop body can consume tensors from
// the parent graph without it being a subgraph input to the loop body.
// In the PopART IR, we add implicit inputs to the LoopOp and loop body
// subgraph, due to subgraphs not being allowed to consume parent graph tensors
// directly.
// This also allows loop body subgraphs to be reused at multiple call sites.
//
//
// Note:
// The trip count and termination condition tensors are optional as Op inputs
// The condition tensor input/output in the subgraph needs to exist in any case

class LoopOp : public SubgraphOp {
public:
  LoopOp(const OperatorIdentifier &, const Op::Settings &, Graph &callee_);

  LoopOp(const OperatorIdentifier &,
         const Op::Settings &,
         Graph &callee_,
         std::vector<std::pair<TensorId, TensorInfo>> opInputs_,
         std::vector<TensorId> implicitTensors_);

  void setup() final;
  void appendAttributes(OpSerialiserBase &) const override;
  void connectInTensor(InIndex inIndex, TensorId tensorId) final;
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  std::vector<TensorId> getInputsForGraph(const Graph &graph) const final;
  std::unique_ptr<Op> clone() const override;
  std::vector<const Graph *> getCalledGraphs() const final;
  std::vector<TensorId> implicitInputTensors() const;

  Graph &getCalledGraph() const override;

  int getTripCountValue() const { return tripCountValue; }
  void setTripCountValue(int value) { tripCountValue = value; }
  int numExplicitInputs() const;
  int numImplicitInputs() const;

  InIndex subgraphInToOpInIndex(InIndex index) const override;
  InIndex opInToSubgraphInIndex(InIndex index) const override;

  OutIndex subgraphOutToOpOutIndex(OutIndex index) const override;
  OutIndex opOutToSubgraphOutIndex(OutIndex index) const override;

  // The input specifying the maximum number of loop iterations
  static InIndex getMaximumTripCountInIndex() { return 0; }

  // Termination condition: if true, loop keeps going, if false, loop stops
  static InIndex getTerminationConditionInIndex() { return 1; }

  // The first regular, user-defined loop input
  static InIndex getFirstInputInIndex() { return 2; }

  void
  addLoopInput(InIndex index, TensorId tensorId, TensorId subgraphTensorId);

  void
  addLoopOutput(OutIndex index, TensorId tensorId, TensorId subgraphTensorId);

private:
  std::reference_wrapper<Graph> callee;
  int tripCountValue;
};

} // namespace popart

#endif
