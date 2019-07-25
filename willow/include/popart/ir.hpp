#ifndef GUARD_NEURALNET_WILLOWIR_HPP
#define GUARD_NEURALNET_WILLOWIR_HPP

#include <map>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/names.hpp>
#include <popart/opidentifier.hpp>
#include <popart/optionflags.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/tensorindex.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

// helper class used during backwards pass construction.
// This class helps to decouple the non-grad op from a
// grad op (previously grad ops kept a pointer to a non-grad
// op, which was dangerous as optimisations remove ops)
class GradNonGradPair {
public:
  Op *grad;
  Op *nongrad;
  GradNonGradPair(Op *g_, Op *ng_);
  GradNonGradPair();
};

// The gradient of a tensor is the sum of 1 or several tensors,
// 1 for each of the nodes which consumed it. This class is for
// tracking/counting these as they come in down edges in backwards
// part of the training compute graph.
class TensorGradRegistry {
public:
  using TMap = std::map<TensorId, std::vector<Tensor *>>;
  // Register tensor "edgeGrad" as being a
  // gradient of "nonGrad" w.r.t. loss along a single edge
  void insert(Tensor *nonGrad, Tensor *edgeGrad);

  // Decrease the number of edges expected to be registered
  // for a non-grad tensor.
  void decrementNumberExpectedEdges(Tensor *nonGrad);

  int getNumberExpectedEdges(Tensor *nonGrad);

  // Return the non-gradient tensors which have ALL their
  // required gradients registered, and are thus ready to
  // have their edge gradients summed to
  // obtain the final gradient.
  // Note that this is NOT a const pop member function
  TMap popComplete();

  // stores all non-grad tensors which have some, but not all of
  // their edges already having gradients registered
  TMap partial;
  // stores all non-grad tensors which have had all of their
  // edges provide gradients. When popCompleted() is called,
  // this map is returned,
  TMap complete;

private:
  // the number of edges expected to register gradients for a non-grad tensor.
  std::map<TensorId, int> expectedNumEdges;

  void tryMakeComplete(Tensor *nonGrad);
};

class OpGradRegistry {
public:
  // register that the output of nonGrad Op at OutIndex index
  // has had its gradient tensor computed
  void insert(Op *nonGrad, int index);
  std::vector<Op *> popComplete();

private:
  // For a non-grad-op, which of its outputs (by index)
  // have had a gradient computed
  std::map<OpId, std::set<int>> partial;
  // When all required gradient inputs are in,
  // move the key of partial from partial to complete
  std::vector<Op *> complete;
};

// Ir Constructor inputs
class IrBundle {
public:
  IrBundle(const onnx::ModelProto &modelProto,
           const InputShapeInfo &inputShapeInfo,
           const DataFlow &dataFlow,
           const std::vector<Loss *> &losses,
           const Optimizer *optimizer,
           DeviceInfo &deviceInfo,
           const SessionOptions &userOptions,
           const Patterns &patterns);

  const onnx::ModelProto &modelProto;
  const InputShapeInfo &inputShapeInfo;
  const DataFlow &dataFlow;
  const std::vector<Loss *> &losses;
  const Optimizer *optimizer;
  DeviceInfo &deviceInfo;
  const SessionOptions &userOptions;
  const Patterns &patterns;
};

// FFS : Use a factory method to create the IR class and return a pointer
//       Then the constructor can be private.
// ie. static std::unique_ptr<Ir> create(const IrBundle&);

class Ir {

public:
  enum class ExecutionMode { INFERENCE, EVALUATION, TRAINING };

  Ir();
  ~Ir();

  // Set the onnxModel.
  // A note on constant tensors: The outputs of ONNX Constant Operators
  // will always be treated as constants, so left unchanged if in training mode
  // Weights for training should always therefore rather appear in the ONNX
  // initializer list, and in the ONNX input list.
  void setOnnxModel(const onnx::ModelProto &model);

  // Set the dataflow
  void setDataFlow(const DataFlow &df);

  // Set the user options
  void setUserOptions(const SessionOptions &flags);

  // Set the input shape information
  void setInputShapeInfo(const InputShapeInfo &info);

  // Set the optimizer and add optimizer tensors
  // FFS could this be combined with updateOptimizer?
  void setOptimizer(const Optimizer *o);

  // Set the device info
  void setDeviceInfo(DeviceInfo &);

  const DeviceInfo *getDeviceInfo();

  // Set the optimization patterns
  void setPatterns(const Patterns &p);

  // Remove from the IR any tensors which are unconnected, i.e.
  // the have no producers or consumers
  void removeIsolatedTensors();

  // Set which execution mode we are using
  void setExecutionMode(const ExecutionMode &mode);

  // Convenience methods to query the mode of the model.
  // Onnx refers to Inference as testing.
  bool isTraining() { return executionMode == ExecutionMode::TRAINING; }
  bool isTesting() { return executionMode == ExecutionMode::INFERENCE; }
  bool isEvaluation() { return executionMode == ExecutionMode::EVALUATION; }

  // Set the losses, will clear any previous losses
  void setLosses(const std::vector<Loss *> &l);

  // Log the IR in a human readable format.
  void logIr();

  // Prepare the IR based on the IrBundle configuration
  void prepare(const IrBundle &);

  // Reset the weights with data from an ONNX model
  void resetWeights(const onnx::ModelProto &modelProto);

  void updateOptimizer(const Optimizer *);
  // take training steps
  onnx::ModelProto step(int n);
  // if the tensor is returned to user (passes call to DataFlow).
  bool isAnchored(const TensorId &) const;
  void append(std::stringstream &) const;
  std::vector<std::unique_ptr<Loss>> losses;

  // The tensors specific to the optimization. Learning rate(s), momentum(s) etc
  std::vector<Tensor *> optimizerTensors() const;

  // The input data tensors. label(s), image(s), etc. This does not include
  // optimizer stream tensors (they are not data)
  std::vector<Tensor *> dataStreamTensors() const;

  std::vector<Op *> opsOfType(const OperatorIdentifier &opid);
  bool isConsumedByOpOfType(TensorId tid, const OperatorIdentifier &opid);

  // Simple recursive depth first search
  std::vector<const Graph *> getGraphSchedule() const;

  // Essentially Kahn's algorithm (1962),
  // https://en.wikipedia.org/wiki/Topological_sorting
  // with additional constrains imposed through the input paramater.
  // Ops which are ready to be inserted have an insertion "priority",
  // set elsewhere.
  std::vector<Op *> getOpSchedule(const OpsBeforeKey &) const;

  // Do all the Ops with all their dependencies form a DAG?
  bool isSchedulable(const OpsBeforeKey &) const;

public:
  OpId getOpsCounter() const;
  OpId getAndIncrOpsCounter();
  TensorId getFinalLossId() const;
  // The OpId if the Op which sums all loss values from the LossOps
  OpId getFinalLossOpId() const;
  // if check is in userOptions.dotChecks, then write the .dot file
  // in userOptions.logDir
  void dotCheckpoint(DotCheck check) const;

  const onnx::ModelProto &getModel() const;

  const SessionOptions &getSessionOptions() const { return userOptions; }

  std::vector<TensorId> getTensorIds(TensorType) const;
  Tensor *getTensor(const TensorId &) const;
  bool containsTensor(const TensorId &) const;
  std::vector<TensorId> getGraphInputIds() const;

  const Graph &getMainGraph() const;
  Graph &getMainGraph();

  Graph &getGraph(const GraphId &) const;
  bool hasGraph(const GraphId &) const;

  Graph &createGraph(const GraphId &);

  std::map<OpId, std::unique_ptr<Op>> &getMainGraphOps();
  const std::map<OpId, std::unique_ptr<Op>> &getMainGraphOps() const;

  Tensors &getMainGraphTensors();
  const Tensors &getMainGraphTensors() const;

  // Accessors for the dataFlow
  const DataFlow &getDataFlow() const { return dataFlow; }

  const std::set<Op *> &getTrainTargetOps() { return trainTargetOps; }

  bool addToTrainTargetOps(Op *op);

  // modify a Graph using a graph transformation
  // (public for unit testing only)
  void applyTransform(std::size_t transformId, Graph &graph);

  // run after creating the backwards pass, checks that
  // the user provided anchor tensors actually exist.
  // the user may have not used the correct gradient
  // tensor naming convention for example, this will
  // be caught here.
  void validateAnchors() const;

  ExecutionMode getExecutionMode() const;

  // Can the IR be used for inference.
  bool canInfer() const;

  // Can the IR be used for evaluation.
  // This is true when there are losses to compute.
  bool canEvaluate() const;

  // Can the IR be used for training.
  // This is true when there are losses and an optimizer.
  bool canTrain() const;

  // returns true if there are initializers in the onnx model
  bool containsInitialisers();

  // Convert the ONNX graph into the forwards pass of the IR
  void constructForwards();

  // Convert an ONNX graph into IR
  Graph &constructFromOnnxGraph(const onnx::GraphProto &graph,
                                const Scope &scope);

  void foldConstants(Graph &);

  // Construct the backwards pass of the IR by doing an autograd of the forward
  // pass
  void constructBackwards();

  // Register the input tensors of the ONNX graph,
  // and the inputs to the losses. For the ONNX input tensors,
  // determines which are Stream and which are Variable
  void registerInputTensors();

  // Consider the number of out edges a Vertex (Op/Tensor) has which lead to the
  // final loss Tensor is used in constructing the backwards pass. This function
  // sets this number for all Vertices. Out edges go to consumers for Tensors,
  // and to output Tensors for Ops.
  void setNEdgesToLoss();

  // For all vertices set the phase, and whether or not
  // there is a path to vertex in whose phase is BWD.
  void updateVertices();

  // modify the Ir using all the registered pre-alias patterns
  void applyPreAliasPatterns(Graph &);

  void applyUpdateInplacePrioritiesForIpu();

  void applyInplacePattern(Graph &);

  // confirm that the names of the Const tensors
  // from the user (constTensors) are in the onnx Model
  // Can be run after the forward pass of Ir has been
  // constructed
  void confirmConstIds() const;

  // confirm that no tensors in input(), nodes() or preRunKnowledge()
  // use reserved naming conventions. A note on design: The decision
  // to NOT add an independent dimension to TensorId, used exclusively
  // by automatically named tensors, was that when printing TensorIds
  // there would still be the possibility of conflict (i.e. projection
  // to single string might result in conflict).
  void confirmNoReservedIds() const;

  // starting from losses, construct the individual loss ops
  // as well as an op which sums them to get the final op
  void growFinalLoss();

  // Return the default opset version for a domain
  int getDefaultOpsetVersion(const std::string &domain) const;

  // Return the opset version in use for a domain
  int getOpSetVersionFromModel(const std::string &domain) const;

  bool autoRecomputationEnabled() const {
    return userOptions.autoRecomputation != RecomputationType::None;
  }

  uint32_t getAndIncrementDropoutSeedModifier();

private:
  // Accessors for the tensors
  const Tensors &getTensors() const;
  Tensors &getTensors();

  // modify the Ir using with pattern matching
  // Returns true if a change to the Ir was made.
  bool applyPreAliasPattern(const PreAliasPattern *, Graph &);

  // gradients are named automatically. To prevent them
  // getting names already taken by non-gradient tensors,
  // we check that a reserved pattern is not present.
  void confirmNonReservedId(const TensorId &tenId) const;

  void growGradientVarUpdateOp(const TensorId &varId);

  void growCopyVarUpdateOp(const TensorId &varId, const TensorId &from);

  // Common code for the growGradient... and growCopy...
  void growVarUpdateOpInternal(OpId opId);

  // Get the best virtual graph Id based on the graph Ids of producers of ts
  // to minimise graph<->graph communication
  boost::optional<int64_t>
  getVirtualGraphIdFromTensorProducers(std::vector<Tensor *> ts);

  Op *growGradSumOp(Tensor *target, const std::vector<Tensor *> &toSum);

  std::vector<Op *> growGradOps(Op *forwardOp);

  // for each of the losses described by loss,
  // create a grad-op. Return a vector of {gradop, lossop} pairs
  std::vector<GradNonGradPair> growLossGradients();

  // Verify the connectivity of the graph
  void verifyConnectivity() const;
  void verifyOpInputConnectivity(const Graph &graph) const;
  void verifyOpOutputConnectivity(const Graph &graph) const;
  void verifyTensorProducerConnectivity() const;
  void verifyTensorConsumerConnectivity() const;
  void verifyTensorIds() const;
  void verifyVirtualGraphIds(bool postAutoVirtualGraphTransform) const;
  void verifyVertexAttributesOnlyInMain() const;

  // Verify ConstExpr folding has removed input tensors
  // as expected
  void verifyConstExprFolding();
  bool isCandidateForConstExprFolding(const Tensor &tensor) const;
  std::set<Tensor *> getRootInputsToOp(Op *op);

private:
  DataFlow dataFlow;

  std::unique_ptr<onnx::ModelProto> onnxModel;

  // learning rate, momentum, etc.
  // Optimizer needed to construct backwards pass:
  // if momentum the Ir is different
  std::unique_ptr<Optimizer> optimizer;
  DeviceInfo *deviceInfo = nullptr;
  SessionOptions userOptions;
  InputShapeInfo inputShapeInfo;

  // The set of patterns to apply after constructing
  // forwards and backwards passes
  Patterns patterns;

  // create an Op from a Node
  std::unique_ptr<Op> addOp(const Node &, const Scope &);

  std::map<GraphId, std::unique_ptr<Graph>> graphs;

  // total number of ops ever created
  OpId opsCounter{100};

  // Map of transform Id to enable flag
  std::map<std::size_t, bool> transformEnableMap;

  // Map of ops and their root inputs
  std::map<OpId, std::set<Tensor *>> opAndRootInputs;

  // The update ops which must be run during a training pass
  std::set<Op *> trainTargetOps;

  OpId finalLossOpId{-1000};

  ExecutionMode executionMode = ExecutionMode::TRAINING;

  bool isPrepared = false;

  // enable/disable a transform stage
  void enableTransform(std::size_t transformId, bool enable);

  uint32_t dropoutSeedModifier = 0;

public:
  // A "dummy" Op used to ensure that anchor tensors
  // will be copied out of sub-graphs, even if they
  // have no consumers external to the sub-graph.
  Op &getSubgraphAnchorPlaceholder();
};

} // namespace popart

namespace std {
template <> struct hash<popart::Ir> {
  std::size_t operator()(const popart::Ir &ir) const {
    std::stringstream ss;
    ir.append(ss);
    return std::hash<std::string>{}(ss.str());
  }
};
}; // namespace std

#endif
