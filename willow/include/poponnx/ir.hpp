#ifndef GUARD_NEURALNET_WILLOWIR_HPP
#define GUARD_NEURALNET_WILLOWIR_HPP

#include <map>
#include <poponnx/dataflow.hpp>
#include <poponnx/inputshapeinfo.hpp>
#include <poponnx/names.hpp>
#include <poponnx/opidentifier.hpp>
#include <poponnx/optionflags.hpp>
#include <poponnx/patterns/patterns.hpp>
#include <poponnx/tensorindex.hpp>
#include <poponnx/transforms/transform.hpp>

namespace poponnx {

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
  using TMap = std::map<Tensor *, std::vector<Tensor *>>;
  // Register tensor "edgeGrad" as being a
  // gradient of "nonGrad" w.r.t. loss along some edge
  void insert(Tensor *nonGrad, Tensor *edgeGrad);

  // Return the non-gradient tensors which have ALL their
  // required gradients registered, and are thus ready to
  // have their edge gradients summed to
  // obtain the final gradient.
  // Note that this is NOT a const pop member function
  TMap popComplete();

private:
  // stores all non-grad tensors which have some, but not all of
  // their edges already having gradients registered
  TMap partial;
  // stores all non-grad tensors which have had all of their
  // edges provide gradients. When popCompleted() is called,
  // this map is returned,
  TMap complete;
};

class OpGradRegistry {
public:
  // register that the output of nonGrad at index
  // has had its gradient tensor computed
  void insert(Op *nonGrad, int index);
  std::vector<Op *> popComplete();

private:
  // For a non-grad-op, which of its outputs (by index)
  // have had a gradient computed
  std::map<Op *, std::set<int>> partial;
  // When all required gradient inputs are in,
  // move the key of partial from partial to complete
  std::vector<Op *> complete;
};

// Stores string elements in both a std::vector and a std::set
// The elements in the std::vector will be unique
class VectorAndSet {
public:
  VectorAndSet();
  VectorAndSet(const std::vector<std::string> &vals);
  ~VectorAndSet();
  VectorAndSet &operator=(const VectorAndSet &rhs) = default;
  bool contains(std::string) const;
  const std::vector<std::string> &v() const;
  // insert string if not present, otherwise do nothing
  void insert(const std::string &);
  void reset(const std::vector<std::string> &vals);

private:
  std::vector<std::string> v_vals;
  std::set<std::string> m_vals;
};

class Tensors {
public:
  Tensors(Ir &pg);
  ~Tensors() = default;

  Tensor *get(TensorId) const;
  void remove(TensorId);
  bool contains(TensorId) const;

  // create a Variable Tensor
  void addVarInit(const TensorId &, const onnx::TensorProto *);

  // create a Constant Tensor
  void addConstInit(const TensorId &, const onnx::TensorProto *);
  void addConstInit(const TensorId &, const TensorInfo &, const void *);

  // create a Tensor of type Stream
  void addStream(TensorId, const TensorInfo &);
  // create a Tensor of type ActGrad (basically any tensor which is
  // the output of an Op)
  void addActGrad(TensorId);
  std::vector<TensorId> getIds(TensorType) const;
  std::vector<TensorId> getAllTensorIds() const;
  std::vector<TensorId> getNoProducerIds() const;
  const onnx::TensorProto *getOnnxInit(TensorId) const;
  void append(std::stringstream &) const;

  const VectorAndSet &getConstIds() const { return constIds; }
  void insertConstId(const std::string &);
  // remove all Tensors which have no producer and no consumers
  void removeIsolated();

private:
  // Store the Tensors of type Const
  VectorAndSet constIds;

  std::map<TensorId, std::unique_ptr<Tensor>> M;
  // adds to M, but first confirms that TensorId not already in
  void insert(TensorId, std::unique_ptr<Tensor>);

  void addInit(const TensorId &, const onnx::TensorProto *, TensorType);

  Ir &ir;
};

// Ir Constructor inputs
class IrBundle {
public:
  IrBundle(const onnx::ModelProto &modelProto,
           const InputShapeInfo &inputShapeInfo,
           const DataFlow &dataFlow,
           const std::vector<Loss *> &losses,
           const Optimizer *optimizer,
           const SessionOptions &userOptions,
           const Patterns &patterns);

  const onnx::ModelProto &modelProto;
  const InputShapeInfo &inputShapeInfo;
  const DataFlow &dataFlow;
  const std::vector<Loss *> &losses;
  const Optimizer *optimizer;
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

  // Set the optimization patterns
  void setPatterns(const Patterns &p);

  // Remove from the IR any tensors which are unconnected, i.e.
  // the have no producers or consumers
  void removeIsolatedTensors();

  // Set which execution mode we are using
  void setExecutionMode(const ExecutionMode &mode);

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
  bool isAnchored(TensorId);
  void append(std::stringstream &);
  std::vector<std::unique_ptr<Loss>> losses;

  // The tensors specific to the optimization. Learning rate(s), momentum(s) etc
  std::vector<Tensor *> optimizerTensors() const;

  // The input data tensors. label(s), image(s), etc. This does not include
  // optimizer stream tensors (they are not data)
  std::vector<Tensor *> dataStreamTensors() const;

  std::vector<Op *> opsOfType(const OperatorIdentifier &opid);

  // Essentially Kahn's algorithm (1962),
  // https://en.wikipedia.org/wiki/Topological_sorting
  // with additional constrains imposed through the input paramater.
  // Ops which are ready to be inserted have an insertion "priority",
  // set elsewhere.
  std::vector<Op *> getOpSchedule(const OpsBeforeKey &) const;

  // Do all the Ops with all their dependencies form a DAG?
  bool isSchedulable(const OpsBeforeKey &) const;

private:
  std::unique_ptr<Scheduler> scheduler;

public:
  OpId getOpsCounter() const;
  OpId getAndIncrOpsCounter();
  TensorId getFinalLossId() const;
  // The OpId if the Op which sums all loss values from the LossOps
  OpId getFinalLossOpId() const;
  void exportDot(std::string dotfn) const;
  void eraseOp(OpId);
  Op *getOp(OpId);
  const onnx::ModelProto &getModel() const;

  const SessionOptions &getSessionOptions() const { return userOptions; }

  void connectInputsFromInputMapWrapper(const InputMapWrapper &, OpId opId);
  void connectOutputsFromOutputMapWrapper(const OutputMapWrapper &, OpId opId);

  // moves ownership of created Op into the Ir,
  // and returns the Op's OpId
  OpId moveIntoIr(std::unique_ptr<Op> op);

  // Accessors for the tensors
  const Tensors &getTensors() const { return tensors; }
  Tensors &getTensors() { return tensors; }

  const std::map<OpId, std::unique_ptr<Op>> &getOps() const { return ops; }

  // Accessors for the dataFlow
  const DataFlow &getDataFlow() const { return dataFlow; }

  const std::set<Op *> &getTrainTargetOps() { return trainTargetOps; }

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

  // modify the Ir using a graph transformation (public for unit testing only)
  void applyTransform(std::size_t transformId);

  // enable/disable a transform stage (public for unit testing only)
  void enableTransform(std::size_t transformId, bool enable);

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

  // Construct the backwards pass of the IR by doing an autograd of the forward
  // pass
  void constructBackwards();

  // The variable update ops must be final consumers of the
  // input variable tensor. This function imposes these constraints
  void setVarUpdateCons();

  // Register the input tensors of the ONNX graph,
  // and the inputs to the losses. For the ONNX input tensors,
  // determines which are Stream and which are Variable
  void registerInputTensors();

  // The number of paths to the loss is used in
  // constructing the backwards pass. This functions set
  // this number of all Ops and Tensors
  void setNPathsToLoss();

  // For all vertices set the phase, and whether or not
  // there is a path to vertex in whose phase is BWD.
  void updateVertices();

  // modify the Ir using all the registered patterns.
  // Returns true if a change to the Ir was made.
  void applyPatterns(PatternPhase);

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

  // Return the opset version for a domain
  int getOpSetVersionFromModel(const std::string &domain);

private:
  // called from growFromNode and many other places where Ops created
  // T requires functions input(int) and input_size()
  template <typename T> void connectInputs(const T &, OpId opId);

  // T requires functions output(int) and output_size()
  template <typename T> void connectOutputs(const T &, OpId opId);

  // modify the Ir using with pattern matching
  // Returns true if a change to the Ir was made.
  bool applyPattern(const Pattern *);

  // gradients are named automatically. To prevent them
  // getting names already taken by non-gradient tensors,
  // we check that a reserved pattern is not present.
  void confirmNonReservedId(TensorId tenId) const;

  // create an Op from Node (if not Constant Node), wire it to
  // correct input Tensors and create the activation output Tensors
  Op *growFromNode(const Node &);

  Op *growVarUpdateOp(TensorId varId);

  Op *growRecomputeOp(Op *oriOp, const std::set<Op *> &checkpoints);

  Op *growGradSumOp(Tensor *target, const std::vector<Tensor *> &toSum);

  std::vector<Op *> growGradOps(Op *forwardOp);

  // for each of the losses described by loss,
  // create a grad-op. Return a vector of {gradop, lossop} pairs
  std::vector<GradNonGradPair> growLossGradients();

  // Verify the connectivity of the graph
  void verifyConnectivity() const;
  void verifyOpInputConnectivity() const;
  void verifyOpOutputConnectivity() const;
  void verifyTensorProducerConnectivity() const;
  void verifyTensorConsumerConnectivity() const;

private:
  Tensors tensors;
  DataFlow dataFlow;

  std::unique_ptr<onnx::ModelProto> onnxModel;

  // learning rate, momentum, etc.
  // Optimizer needed to construct backwards pass:
  // if momentum the Ir is different
  std::unique_ptr<Optimizer> optimizer;
  SessionOptions userOptions;
  InputShapeInfo inputShapeInfo;

  // The set of patterns to apply after constructing
  // forwards and backwards passes
  Patterns patterns;

  // create an Op from a Node
  std::unique_ptr<Op> addOp(const Node &);
  std::map<OpId, std::unique_ptr<Op>> ops;

  // total number of ops ever created
  OpId opsCounter{100};

  // Map of transform Id to enable flag
  std::map<std::size_t, bool> transformEnableMap;

  // The update ops which must be run during a training pass
  std::set<Op *> trainTargetOps;

  OpId finalLossId{-1000};

  ExecutionMode executionMode = ExecutionMode::TRAINING;

  bool isPrepared = false;

public:
  std::unique_ptr<TopoCons> topoCons;
};

} // namespace poponnx

#endif
