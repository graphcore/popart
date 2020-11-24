// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GRAPH_HPP
#define GUARD_NEURALNET_GRAPH_HPP

#include <map>
#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include <popart/graphid.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/scheduler_requireoptimal.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>

namespace popart {

class BackwardPassCreator;

enum class RequireOptimalSchedule; /*
  Yes = true,
  No = false
*/

class Graph {
  friend class BackwardPassCreator;

public:
  Graph(Ir &, const GraphId &);

  Graph()              = delete;
  Graph(const Graph &) = delete;

  const std::map<OpId, std::unique_ptr<Op>> &getOps() const;
  std::map<OpId, std::unique_ptr<Op>> &getOps();
  std::vector<OpId> getOpIds() const;

  static const int64_t NoVGraph;

  // Obtain a set of all vitual graphs Id used across ops
  const std::set<int64_t> getAllVirtualGraphIds() const;

  // Obtain counts for each vitual graphs Id used across ops
  const std::map<int64_t, int> getVirtualGraphCounts() const;

  Op *getOp(OpId opId) const;

  const Tensors &getTensors() const;
  Tensors &getTensors();

  const Ir &getIr() const { return ir; }
  Ir &getIr() { return ir; }

  const TensorId &getLoss() const { return loss; }

  void setLoss(const TensorId &loss_) { loss = loss_; }

  void constructFromOnnxGraph(const ONNX_NAMESPACE::GraphProto &onnx_graph);
  Op *growFromNode(const Node &node);
  Graph &getBackwardsGraph(const GraphId &bwdId);

  // moves ownership of created Op into the Graph,
  // and returns the Op's OpId
  OpId moveIntoGraph(std::unique_ptr<Op> op);

  // Create an op in this graph.
  template <typename OP, typename... Args> Op *createOp(Args &&... args);

  std::vector<const Graph *> getCalledGraphs() const;

  // called from growFromNode and many other places where Ops created
  // T requires functions input(int) and input_size()
  template <typename T> void connectInputs(const T &inContainer, OpId opId);

  // T requires functions output(int) and output_size()
  template <typename T> void connectOutputs(const T &outContainer, OpId opId);

  void connectInputsFromInputMapWrapper(const InputMapWrapper &in, OpId id);
  void connectOutputsFromOutputMapWrapper(const OutputMapWrapper &, OpId opId);

  void eraseOp(OpId id);

  // The variable update ops must be final consumers of the
  // input variable tensor. This function imposes these constraints
  void setVarUpdateConstraints();

  // All other ops producing tensors consumed by the bwd conv
  // must happen before the flipweights
  void setConvFlipWeightConstraints();

  // Essentially Kahn's algorithm (1962),
  // https://en.wikipedia.org/wiki/Topological_sorting
  // with additional constrains imposed through the input paramater.
  //
  // Returns the schedule of all the ops in the graph.
  //
  // Parameters:
  //   `const OpsBeforeKey &`: Extra topological constraints.
  //   `RequireOptimalSchedule requireOptimalSchedule`:
  //         Whether the true optimal schedule is required, which could be very
  //         expensive to compute; or whether merely any valid topological
  //         traversal is required.
  //         Note, the schedule is cached, but there may still be a cache miss
  //         if the graph has changed, or if an optimal schedule is required but
  //         the cached one is not optimal.
  //
  // Returns:
  //   `std::vector<Op *>`: The ops in schedule order.
  std::vector<Op *>
  getOpSchedule(const OpsBeforeKey &,
                RequireOptimalSchedule requireOptimalSchedule) const;

  // Freeze the schedule into a total order
  void freezeSchedule(const OpsBeforeKey &gCons);

  // Do all the Ops with all their dependencies form a DAG?
  bool isSchedulable(const OpsBeforeKey &,
                     bool respectExecutionPhases = false) const;

  // There are ops in the graph with the recompute attribute, derived
  // from user-specified onnx node attribute
  bool hasUserRecomputeOps() const;

  // For every Op "op" in topoOps, there is a set of Ops "ops"
  // defined as the union of
  // 1) "op" and
  // 2)  all Ops appearing before "op" which
  // have output tensors for which there are Ops appearing after
  // "op" in topoOps which will consume them.
  // Note : if topoOps is just the forward pass, the grad-op
  // consumers of a tensor do not appear in "ops". This agrees
  // with the definition.
  std::vector<std::set<Op *>>
  getLiveSets(const std::vector<Op *> &topoOps) const;

  const std::vector<TensorId> &getInputIds() const { return graph_inputs; }

  /// Add a graph input at a specific index in the list
  /// \param index Force the input to be at the specified index in the graph.
  ///              Overwrites any existing input at the index.
  /// \param id tensor name to create and connect
  /// \param info tensor info
  void
  addInput(const InIndex &index, const TensorId &id, const TensorInfo &info);

  /// Add a graph input to the end of the list
  /// \param id tensor name to create and connect
  /// \param info tensor info
  void addInput(const TensorId &id, const TensorInfo &info);

  // Mark an existing tensor as a graph input.
  void markAsInput(const TensorId &);
  // Add new graph input with auto generated name
  TensorId addInput(const TensorInfo &);
  TensorId getInputId(InIndex idx) const { return graph_inputs.at(idx); }

  const std::vector<TensorId> &getOutputIds() const { return graph_outputs; }
  // Mark an existing tensor as a graph output.

  /// Mark a graph tensor as graph output at a specific index in the list
  /// \param index Force the output to be at the specified index in the graph.
  ///              Overwrites any existing output at the index.
  /// \param id tensor in the graph to mark as output
  void markAsOutput(const OutIndex &index, const TensorId &id);

  /// Mark a graph tensor as graph output at the end of the list
  /// \param id tensor in the graph to mark as output
  void markAsOutput(const TensorId &id);

  void removeOutput(const TensorId &);
  TensorId getOutputId(OutIndex idx) const { return graph_outputs.at(idx); }

  TensorId addScope(const TensorId &) const;
  TensorId removeScope(const TensorId &) const;
  Scope getScope() const;

  // For grad-graphs, matching input indices to
  // corresponding IN/OUT/GRADOUT indices of
  // corresponding non-grad-graph.
  const std::vector<GradInOutMapper> &gradInputInfo() const {
    return gradInInfo;
  }

  // Returns the call sites to this graph in any order.
  std::vector<Op *> getCallSiteOps() const;

  // Returns the call sites to this graph in IR scheduler order.
  // At most num call sites are returned if num > 0.
  std::vector<Op *> getCallSiteOps(size_t num) const;

  // Computes the "edge map" of Op-wise dependencies of this graph, including
  // those described in `this->topoCons`.
  //
  // Returns:
  //   `edges`: std::map<OpId, std::unordered_set<OpId>>
  //            Map from every OpId to its consumer OpIds. If there is a
  //            dependency from a to b in the graph, then b will be in edges[a].
  //            If a has no dependents in the graph, then a will map to the
  //            empty set (so all OpIds are always in the mapping).
  //
  // Design note: For the type of container used for the edge map, we need a
  // container that is:
  //   1. Orderable (by key), as it is very useful for some callers.
  //   2. Can handle non-contiguous, non-starting-at-zero OpIds.
  //   3. Fast lookup. Modification not a concern.
  //   4. Multiple values per key, which are unique and (explicitly) unordered.
  //
  // `std::multimap<OpId, OpId>`, since C++11, does enforce an ordering on the
  // values - their insertion order - thus we must use a
  // `std::map<OpId, unordered_set<OpId>>`.
  //
  // We do not know that this method is a performance bottleneck, so do not need
  // to consider the performance benefits vs design implications of using a more
  // lightweight container, like `std::vector`.
  std::map<OpId, std::unordered_set<OpId>> getEdgeMap() const;

  const std::string getGraphId() const { return id.str(); }

private:
  std::vector<Op *>
  growGradOps(Op *nonGradOp, const std::map<TensorId, TensorId> &gradTensorMap);

public:
  std::unique_ptr<TopoCons> topoCons;
  const GraphId id;

private:
  std::unique_ptr<Tensors> up_tensors;
  std::map<OpId, std::unique_ptr<Op>> ops;
  std::vector<TensorId> graph_inputs;
  std::vector<TensorId> graph_outputs;
  std::unique_ptr<Scheduler> scheduler;
  std::vector<GradInOutMapper> gradInInfo;

  Ir &ir;
  TensorId loss;

  // Get the virtual graph Id from an op (NoVGraph if not set)
  static int64_t getVirtualGraphId(const Op &op);
};

template <typename T>
void Graph::connectInputs(const T &inContainer, OpId opId) {
  Op *op              = ops[opId].get();
  auto optionalInputs = op->optionalInputs();
  for (int inIndex = 0; inIndex < inContainer.input_size(); ++inIndex) {
    auto &inName = inContainer.input(inIndex);
    if (inName == TensorId()) {
      if (optionalInputs.find(inIndex) == optionalInputs.end()) {
        throw error(
            "No input found for input {} of {}, but input is not optional",
            inIndex,
            op->debugName());
      }
    } else {
      auto scopedName = getTensors().find(inName, op->getScope());
      // default: connects tensor <-> op, in both directions.
      // Note that this is a virtual function, and so specific Ops
      // may to do something different to the default here.
      op->connectInTensor(inIndex, scopedName);
    }
  }
}

template <typename OP, typename... Args> Op *Graph::createOp(Args &&... args) {
  auto ptr  = std::unique_ptr<Op>(new OP(std::forward<Args>(args)...));
  auto opId = moveIntoGraph(std::move(ptr));
  return getOp(opId);
}

template <typename T>
void Graph::connectOutputs(const T &outContainer, OpId opId) {
  for (int outIndex = 0; outIndex < outContainer.output_size(); ++outIndex) {
    auto &outName = outContainer.output(outIndex);
    if (outName == TensorId()) {
      // no output at this position
    } else {
      // ONNX specifies that a tensor is the output of at most 1 node.
      // here we create the Output (activation or gradient) Tensor and
      // connect it to the Op.
      ops[opId]->createAndConnectOutTensor(outIndex, outName);
    }
  }
}

} // namespace popart

#endif
