#ifndef GUARD_NEURALNET_GRAPH_HPP
#define GUARD_NEURALNET_GRAPH_HPP

#include <map>

#include <popart/graphid.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>

namespace popart {

class BackwardPassCreator;

class Graph {
  friend class BackwardPassCreator;

public:
  Graph(Ir &, const GraphId &);

  Graph()              = delete;
  Graph(const Graph &) = delete;

  const std::map<OpId, std::unique_ptr<Op>> &getOps() const;
  std::map<OpId, std::unique_ptr<Op>> &getOps();

  Op *getOp(OpId opId);

  const Tensors &getTensors() const;
  Tensors &getTensors();

  const Ir &getIr() const { return ir; }
  Ir &getIr() { return ir; }

  void constructFromOnnxGraph(const onnx::GraphProto &onnx_graph);
  Op *growFromNode(const Node &node);
  Graph &getBackwardsGraph(const GraphId &bwdId);

  // moves ownership of created Op into the Graph,
  // and returns the Op's OpId
  OpId moveIntoGraph(std::unique_ptr<Op> op);

  std::vector<const Graph *> getCalledGraphs() const;

  // called from growFromNode and many other places where Ops created
  // T requires functions input(int) and input_size()
  template <typename T> void connectInputs(const T &inContainer, OpId opId);

  // T requires functions output(int) and output_size()
  template <typename T> void connectOutputs(const T &outContainer, OpId opId);

  void connectInputsFromInputMapWrapper(const InputMapWrapper &in, OpId id);
  void connectOutputsFromOutputMapWrapper(const OutputMapWrapper &, OpId opId);

  std::unique_ptr<Op> addOp(const Node &node);

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
  // Ops which are ready to be inserted have an insertion "priority",
  // set elsewhere.
  std::vector<Op *> getOpSchedule(const OpsBeforeKey &) const;

  // Do all the Ops with all their dependencies form a DAG?
  bool isSchedulable(const OpsBeforeKey &,
                     bool respectPingPongPhases = false) const;

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
  void addInput(const TensorId &, const TensorInfo &);
  // Mark an existing tensor as a graph input.
  void markAsInput(const TensorId &);
  // Add new graph input with auto generated name
  TensorId addInput(const TensorInfo &);
  TensorId getInputId(InIndex idx) const { return graph_inputs.at(idx); }

  const std::vector<TensorId> &getOutputIds() const { return graph_outputs; }
  // Mark an existing tensor as a graph output.
  void markAsOutput(const TensorId &);
  void removeOutput(const TensorId &);
  TensorId getOutputId(OutIndex idx) const { return graph_outputs.at(idx); }

  void markAsZeroCopy(const TensorId &);
  bool isMarkedAsZeroCopy(const TensorId &) const;

  TensorId addScope(const TensorId &) const;
  TensorId removeScope(const TensorId &) const;
  Scope getScope() const;

  // For grad-graphs, matching input indices to
  // corresponding IN/OUT/GRADOUT indices of
  // corresponding non-grad-graph.
  const std::vector<GradInOutMapper> &gradInputInfo() const {
    return gradInInfo;
  }

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
  std::vector<TensorId> zero_copy;
  std::unique_ptr<Scheduler> scheduler;
  std::vector<GradInOutMapper> gradInInfo;

  Ir &ir;
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
