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

namespace onnxpasses {
class IOnnxToOnnx;
}

enum class RequireOptimalSchedule; /*
  Yes = true,
  No = false
*/

class Graph {
  friend class BackwardPassCreator;

public:
  Graph(Ir &, const GraphId &);
  ~Graph();

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
  template <typename OP, typename... Args> OP *createOp(Args &&... args);

  // Create an op in this graph, connecting inputs and outputs
  template <typename OP, typename... Args>
  OP *createConnectedOp(const std::map<InIndex, TensorId> &in,
                        const std::map<OutIndex, TensorId> &out,
                        Args &&... args);

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
  /// Get the index of the graph input with a specific id. If the id is not
  /// a valid input id then a error will be raised.
  /// \param id Tensor name to find the index for.
  /// \return The input index for the specified id, if it exists.
  InIndex getInputIndex(TensorId id) const;

  /// Add a graph input at a specific index in the list
  /// \param index Force the input to be at the specified index in the graph.
  /// \param id Tensor name to create and connect
  /// \param info Tensor info
  /// \param overwrite Overwrites any existing input at the index if true,
  ///                  otherwise, moves all other inputs by one position
  void addInput(const InIndex &index,
                const TensorId &id,
                const TensorInfo &info,
                bool overwrite);

  /// Add a graph input to the end of the list
  /// \param id Tensor name to create and connect
  /// \param info Tensor info
  void addInput(const TensorId &id, const TensorInfo &info);

  // Mark an existing tensor as a graph input.
  void markAsInput(const TensorId &);
  // Add new graph input with auto generated name
  TensorId addInput(const TensorInfo &);
  TensorId getInputId(InIndex idx) const { return graph_inputs.at(idx); }
  bool hasInputId(const TensorId &id) const;

  void removeInput(const TensorId &);
  void removeInput(const InIndex &);

  const std::vector<TensorId> &getOutputIds() const { return graph_outputs; }
  OutIndex getOutputIndex(TensorId id) const;

  // Mark an existing tensor as a graph output.

  /// Mark a graph tensor as graph output at a specific index in the list
  /// \param index Force the output to be at the specified index in the graph.
  ///              Overwrites any existing output at the index.
  /// \param id Tensor in the graph to mark as output
  /// \param overwrite Overwrites any existing output at the index if true,
  ///                  otherwise, moves all other outputs by one position
  void markAsOutput(const OutIndex &index, const TensorId &id, bool overwrite);

  /// Mark a graph tensor as graph output at the end of the list
  /// \param id Tensor in the graph to mark as output
  void markAsOutput(const TensorId &id);

  void removeOutput(const TensorId &);
  void removeOutput(const OutIndex &);
  TensorId getOutputId(OutIndex idx) const { return graph_outputs.at(idx); }
  bool hasOutputId(const TensorId &id) const;

  TensorId addScope(const TensorId &) const;
  TensorId removeScope(const TensorId &) const;
  Scope getScope() const;

  // For grad-graphs, matching input indices to
  // corresponding IN/OUT/GRADOUT indices of
  // corresponding non-grad-graph.
  const std::vector<GradInOutMapper> &gradInputInfo() const {
    return gradInInfo;
  }

  // For grad-graphs, mapping from output indices to
  // corresponding IN indices of non-grad graph.
  std::map<OutIndex, InIndex> gradOutputInfo() const { return gradOutInfo; }

  /// Replace oldId with newId on any consumers.
  /// Both tensors need to exist.
  /// \param oldId Tensor to disconenct from consumers & graph outputs
  /// \param newId Tensor to connect from consimers & graph outputs
  void replaceTensor(const TensorId &oldId, const TensorId &newId);

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

  const std::string &getGraphId() const { return id.str(); }
  std::string getGraphString() const;

  // Copy all contents from another graph to this graph
  void copyFrom(const Graph &other);

private:
  std::vector<Op *>
  growGradOps(Op *nonGradOp, const std::map<TensorId, TensorId> &gradTensorMap);

public:
  std::unique_ptr<TopoCons> topoCons;
  const GraphId id;

  /**
   * Set the object which will perform the ONNX -> ONNX transformation, which
   * happens early on in the Graph constructor. The default object, which is
   * used if this method is not called, is an instance of the
   * onnxpasses::Canonnxalizer class, which performs a set of required
   * transformations, such as decomposing ASinh into more basic Nodes.
   * */
  void setOnnxToOnnx(std::unique_ptr<onnxpasses::IOnnxToOnnx>);

private:
  std::unique_ptr<Tensors> up_tensors;
  std::map<OpId, std::unique_ptr<Op>> ops;
  std::vector<TensorId> graph_inputs;
  std::vector<TensorId> graph_outputs;
  std::unique_ptr<Scheduler> scheduler;
  std::vector<GradInOutMapper> gradInInfo;
  std::map<OutIndex, InIndex> gradOutInfo;
  std::unique_ptr<onnxpasses::IOnnxToOnnx> onnxToOnnx;

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

template <typename OP, typename... Args> OP *Graph::createOp(Args &&... args) {
  auto ptr  = std::unique_ptr<Op>(new OP(std::forward<Args>(args)...));
  auto opId = moveIntoGraph(std::move(ptr));
  return static_cast<OP *>(getOp(opId));
}

template <typename OP, typename... Args>
OP *Graph::createConnectedOp(const std::map<InIndex, TensorId> &in,
                             const std::map<OutIndex, TensorId> &out,
                             Args &&... args) {
  OP *op = createOp<OP>(std::forward<Args>(args)...);

  for (auto &input : in) {
    op->connectInTensor(input.first, input.second);
  }

  for (auto &output : out) {
    if (getTensors().contains(output.second)) {
      Tensor *t = getTensors().get(output.second);
      if (t->hasProducer()) {
        t->getProducer()->disconnectOutTensor(t);
      }
      op->connectInTensor(output.first, output.second);
    } else {
      op->createAndConnectOutTensor(output.first, output.second);
    }
  }

  op->setup();

  return op;
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
