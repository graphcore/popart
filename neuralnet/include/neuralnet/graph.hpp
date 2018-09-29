#ifndef GUARD_NEURALNET_GRAPH_HPP
#define GUARD_NEURALNET_GRAPH_HPP

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
// The protobuf generated ONNX classes
#include <onnx/onnx.pb.h>
#pragma clang diagnostic pop // stop ignoring warnings

#include <map>
#include <neuralnet/attributes.hpp>
#include <neuralnet/names.hpp>
#include <neuralnet/tensorinfo.hpp>
#include <neuralnet/vertex.hpp>

namespace neuralnet {

class Tensor;
class Graph;
class Op;
class Loss;
class Pattern;

// the input tensor of a grad-op has what kind of
// relationship with the corresponding non-grad-op?
// we do not allow an input to a grad-op to NOT be
// directly related to the the corresponding non-grad-op.
enum class GradOpInType { IN = 0, OUT, GRADOUT };

class GradInOutMapper {
public:
  GradInOutMapper(int iGrad_, int iNonGrad_, GradOpInType);
  // input index to a grad-op
  int iGrad;
  // input/output/gradient-of-output index of
  // corresponding non-grad op, where,
  int iNonGrad;
  // input/output/gradient-of-output is
  GradOpInType type;
};

// The gradient of a tensor is the sum of 1 or several tensors,
// 1 for each of the nodes which consumed it. This class is for
// tracking/counting these as they come in down edges.
class TensorGradRegistry {
public:
  using TMap = std::map<Tensor *, std::vector<Tensor *>>;
  // Register tensor "edgeGrad" as being a
  // gradient of "nonGrad" w.r.t. loss along some edge
  void insert(Tensor *nonGrad, Tensor *edgeGrad);

  // Return the non-gradient tensors which have ALL their
  // required gradientsregisted, and are thus ready to
  // have their edge gradients summed to
  // obtain the final gradient.
  // NOT a const member function
  TMap popComplete();

private:
  // stores all non-grad tensors which have some but not all of
  // their edges having provided gradients
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
  // For a non-grad-op, which if its outputs (by index)
  // have had a gradient computed
  std::map<Op *, std::set<int>> partial;
  // When all required gradient inputs are in,
  // move the key of partial from partial to complete
  std::vector<Op *> complete;
};

// Tensors to log every iteration
// Also, frequency at which to return all weights
// TODO(jn) ask David Norman how tensorflow does this.
class Recorder {
public:
  bool isAnchored(TensorId) const;
  Recorder(const std::vector<TensorId> &);
  const std::vector<TensorId> &anchors() const;

private:
  std::set<TensorId> s_anchors;
  const std::vector<TensorId> v_anchors;
};

std::string getNeuralNetDomain();

// where tensor tenId is consumed by Op with OpId opId at
// index index, what should the name of the edge-gradient
// along this edge be? This is purely string manipulation.
TensorId getEdgeGradId(TensorId tenId, OpId opId, int index);

// the name of the tensor of the total gradient
// (loss and regularizers)
TensorId getGradId(TensorId tenId);

// inverse of previous function
TensorId getNonGradId(TensorId tenId);

// get the learning rate tensor's id for Variable tensor
// Of course, the tensor is rank 0
TensorId getLearningRateId();

// needs to be implemented. will manage things like
// weight decay loss etc.
class Regularizer {};

// Learning scheduler
// momentum, learning rates, etc.
class Schedule {};

// What is known about the Graph before it is run.
// This knowledge can sometimes be compiled into the Graph,
// and for certain backends is even required, for example
// Graphcore IPU requires all Stream Tensor shapes.
class PreRunKnowledge {
public:
  PreRunKnowledge() = default;
  void addInfo(TensorId, const TensorInfo &);
  const TensorInfo &getInfo(TensorId) const;
  bool hasInfo(TensorId) const;
  const std::map<TensorId, TensorInfo> &getInfos() const;

  // return all unique TensorIds of tensors with any
  // information stored in this object, be it TensorInfo
  // or actual tensor.
  std::vector<TensorId> getAllTensorIds() const;

private:
  std::map<TensorId, TensorInfo> infos;
  // we will also have a map of actual tensors, these
  // can be used sometimes to compile the graph (slice
  // indices for example)
};

enum class OpType {
  ADD = 0,
  ADDGRAD,
  AVERAGEPOOL,
  AVERAGEPOOLGRAD,
  CONSTANT,
  CONV,
  CONVDATAGRAD,
  CONVWEIGHTSGRAD,
  L1,
  L1GRAD,
  LOGSOFTMAX,
  LOGSOFTMAXGRAD,
  NLL,
  NLLGRAD,
  PAD,
  RELU,
  RELUGRAD,
  SQUEEZE,
  SQUEEZEGRAD,
  SUM,
  VARUPDATE
};

// models inputs and outputs to Ops, inputs/outputs
// enter/leave at certain indices of an Op
// 1 tensor per index, but 1+ index per tensor
class TensorIndexMap {
public:
  void insert(int, Tensor *);
  // the Tensor at index changes. Note that there
  // must already be a Tensor at the index
  void reset(int, Tensor *);
  Tensor *tensor(int);
  const Tensor *tensor(int) const;
  bool hasIndex(int) const;
  const std::vector<int> &indices(Tensor *) const;
  const std::map<Tensor *, std::vector<int>> &indicesMap() const;
  const std::map<int, Tensor *> &tensorMap() const;
  // the number or indices (keys of tensor_map)
  int n() const;
  void append(std::stringstream &, std::string prefix, int max_id_length) const;
  // set the TensorInfo of tensor(index) if hasIndex(index) is true
  void setInfoIfIndex(const TensorInfo &, int index);
  // the returned vector has correct TensorIds at indices in
  // tensor_map and "" at unused indices inbetween
  std::vector<TensorId> getSerialised() const;
  // returns the longest TensorId of all Tensors in indices_map
  int maxIdLength() const;

private:
  std::map<int, Tensor *> tensor_map;
  std::map<Tensor *, std::vector<int>> indices_map;
};

class OpConstructorBundle {
public:
  OpConstructorBundle(std::string op_type,
                      Graph *,
                      Attributes,
                      std::string domain);

  std::string op_type;
  Graph *pgraph;
  Attributes atts;
  std::string domain;
};

class Op : public Vertex {
public:
  Op(const Node &, Graph *);
  Op(const OpConstructorBundle &);

  std::string str() const;

  // create an ActGrad (output) tensor
  // and wire it to this Ops output
  void createAndConnectOutTensor(OutIndex, TensorId);

  void append(std::stringstream &ss) const;
  virtual ~Op();

  // The consumed Tensors
  TensorIndexMap input;

  // The produced Tensors
  TensorIndexMap output;

  // wire a tensor to input: updates input and
  // updates consumers of tensor with id TensorId
  void connectInTensor(InIndex, TensorId);

  // might the input tensors be modified?
  bool mayModify(InIndex) const;

  // all Ops will be performed "as close to" the order of
  // priority (highest to lowest) while still being topo sorted.
  // This is not finalised, might work differently...
  double priority() const;

  // "Relu" or "Conv" etc.
  const std::string &op_type() const;
  const OpType opType;

  // political affiliation of the Op (same as NodeProto)
  const std::string &domain();

  // the graph to which the Op belongs
  Graph *pgraph;

  // The unique identifier of the Op (will always be set in Op::Op)
  OpId id{-1};

  // attributes from the Node, if it was created from one
  const Attributes nAtts;

  // set shape and type parameters,
  // MUST set output TensorInfos for all outputs
  virtual void setup();

  // return a vector of 1 or several gradient Ops: for
  // obtaining the gradient of the inputs of this Op.
  // If this Op is already a gradient Op, throws error
  // TODO : why is this noy constant?
  virtual std::vector<std::unique_ptr<Op>> getGradOps();

  // return a gradient op's non-gradient partner,
  // if relevant and still valid otherwise
  // throws an error.
  virtual Op *getNonGradOp() const;
  // Design choice.
  // as optimisations get complex we might
  // delete a non-grad op corresponding to a grad-op.
  // For this reason, we prefer NOT to store the non-grad
  // Op* directly by default (although we do in several cases)
  // Preferred: store the id and look up the pointer when
  // needed, so that we will get reliable failure via a map look-up.
  // This is the approach with for example L1Op.

  // A grad-op outputs an edge-gradient tensor dT at gradOpOutIndex.
  // dT is the edge-gradient of a tensor T which was the input
  // to grad-op's non-grad partner. At what index was T the input
  // of non-grad-op? If not relevant (non-grad-ops) throw an error
  virtual int getNonGradInIndex(int gradOpOutIndex) const;

  // For grad-ops, matching input indices to
  // corresponding IN/OUT/GRADOUT indices of
  // corresponding non-grad-op.
  // throws an error if not appropriate (non-grad ops)
  virtual const std::vector<GradInOutMapper> &gradInputInfo() const;

  // return the full map corresponding to getNonGradInIndex.
  // throws an error if not appropriate (non-grad)
  virtual const std::map<int, int> &gradOutToNonGradIn() const;

  // for grad-ops, this is the same as output.tensorMap().
  // for other ops, throws an error
  virtual const std::map<int, Tensor *> &gradOutMap() const;

  // for non-grad-op `op', takes in the set of output indices
  // of `op' for which a gradient is available and returns
  // if all the gradients needed to create grad-ops are present
  // currently this will just compare the size of
  // the set passed in with number of paths to final loss
  bool readyToCreateGradients(std::set<int> &) const;

  virtual void imposeTopoCons() {}

private:
  void appendIO(std::stringstream &) const;
  virtual void appendMore(std::stringstream &) const {}
  const std::string *const p_op_type;
  std::string op_domain;

  // design decision : see-sawing between storing a pointer
  // to the Node from which the Op derives (if it does derive
  // from a Node) or not. Deciding not to for now, (1) not much
  // to copy to the Op (2) cleaner

  // design decision : see-sawing between having special classes
  // for NonGradOp and GradOp, deciding not to. The main motivation
  // for HAVING the distinction was that inputs of GradOps would
  // work differently, that instead of listing them all, the
  // non-grad inputs would be implit from the corresponding
  // Op. Also, there could be functions like "getNonGrapOp"
  // which would return the NonGradOp for a GradOp.
  // Motivation for implicit input was the explicit:
  // 1) inefficient.
  // 2) if done in the same dimensions (ie concat the inputs), how to
  //    handle variadic input size? (ie SumOp).

  // rebuttal to
  // 1) not valid (a few more strings?) and also constricts
  //    the grad op to always take all inputs and outputs
  //    from non-grad op
  // 2) not sure what the problem is here. variadic inputs can be
  //    interleaved if they are of the same size
};

class GradOp : public Op {
public:
  GradOp(const OpConstructorBundle &);
  virtual ~GradOp() override = default;
  virtual int getNonGradInIndex(int) const override final;
  virtual const std::map<int, Tensor *> &gradOutMap() const override final;
};

enum class TensorType;

class OpTypes {
public:
  OpTypes();
  const OpType &get(std::string op_type) const;
  const std::string &get(OpType opType) const;

private:
  std::map<std::string, OpType> opTypes_;
  std::map<OpType, std::string> strings_;
};

OpTypes initOpTypes();
const OpTypes &getOpTypes();

class VectorAndSet {
public:
  VectorAndSet(std::vector<std::string> &&vals);
  bool contains(std::string) const;
  const std::vector<std::string> &v() const;
  ~VectorAndSet();

private:
  std::vector<std::string> v_vals;
  std::set<std::string> m_vals;
};

std::string reservedPrefix();

class Tensors {
public:
  Tensors(std::vector<std::string> &&vals1, Graph *pg);
  ~Tensors();
  // Store the Tensors of type Const
  const VectorAndSet constIds;
  Tensor *get(TensorId) const;
  void remove(TensorId);
  bool contains(TensorId) const;
  // create a Tensor, either of type Const or Variable
  void addInit(TensorId, const onnx::TensorProto *);
  // create a Tensor of type Stream
  void addStream(TensorId);
  // create a Tensor of type ActGrad (basically any tensor which is
  // the output of an Op)
  void addActGrad(TensorId);
  std::vector<TensorId> getInitIds() const;
  std::vector<TensorId> getIds(TensorType) const;
  std::vector<TensorId> getNoProducerIds() const;
  const onnx::TensorProto *getOnnxInit(TensorId) const;
  void addNonGradient(TensorId gradId, Tensor *nonGradTensor);
  void append(std::stringstream &) const;

  // return the tensor of which the
  // tensor with TensorId is a COMPLETE gradient
  Tensor *getNonGradientOf(TensorId) const;

private:
  std::map<TensorId, std::unique_ptr<Tensor>> M;
  // adds to M, but first confirms that TensorId not already in
  void insert(TensorId, std::unique_ptr<Tensor>);
  OnnxTensorPtrs init;
  Graph *pgraph;

  // from gradients to non-gradients (if there are any)
  std::map<TensorId, Tensor *> non_gradients_;
};

class Graph {
public:
  Graph(onnx::ModelProto &&,
        PreRunKnowledge &&,
        Recorder &&,
        // strings or something:
        std::vector<std::unique_ptr<Loss>> &&,
        std::vector<std::unique_ptr<Regularizer>> &&,
        // Schedule needed, if momentum the graph is different
        Schedule &&sched,
        // Weights tensors which are not to be updated
        std::vector<std::string> &&cTens,
        std::string logdir_);

  std::string logdir;

  // take training steps
  onnx::ModelProto step(int n);
  // if the tensor is returned to user (Recorder).
  bool isAnchored(TensorId);
  void append(std::stringstream &);
  PreRunKnowledge preRunKnowledge;
  Recorder recorder;
  std::vector<std::unique_ptr<Loss>> losses;
  std::vector<std::unique_ptr<Regularizer>> regularizers;
  Schedule schedule;
  Tensors tensors;
  ~Graph();
  // split ConvOp with bias into two Ops, a ConvOp
  // followed by an x Op
  void splitConvBias();
  std::vector<Op *> opsOfType(OpType);
  void inferTensorInfos();
  // this does not take into priority, simple topological sort
  std::vector<Op *> getTopologicallySorted() const;

  void constructForwards();
  void constructBackwards();
  void prune();
  // for all tensors in the forward graph, set the number of
  // paths to the final loss (needed in the backwards pass)
  void setNPathsToLoss();
  OpId getOpsCounter() const;
  OpId getAndIncrOpsCounter();

  TensorId getFinalLossId() const;
  Op *getFinalLossOp();
  void exportDot(std::string dotfn) const;
  void eraseOp(OpId);
  Op *getOp(OpId);

private:
  // modify the graph using with pattern matching
  void applyPattern(const Pattern *);

  // confirm that the names of the Const tensors
  // from the user (constTensors) are in the onnx Model
  // Can be run after the forward pass of Graph has been
  // constructed
  void confirmConstIds() const;

  // gradients are named automatically. To prevent them
  // getting names already taken by non-gradiet tensors,
  // we check that a reserved pattern is not present.
  void confirmNonReservedId(TensorId tenId) const;

  // cofirm that no tensors in input(), nodes() or preRunKnowlede()
  // use reserved naming conventions. A note on design: The decision
  // to NOT add an independent dimension to TensorId, used exclusively
  // by automatically named tensors, was that when printing TensorIds
  // there would still be the possibility of conflict (i.e. projection
  // to single string might result in conflict).
  void confirmNoReservedIds() const;

  // create an Op from Node (if not Constant Node), wire it to
  // correct input Tensors and create the activation output Tensors
  Op *growFromNode(const Node &);

  Op *growVarUpdateOp(TensorId varId);

  Op *growGradSumOp(Tensor *target, const std::vector<Tensor *> &toSum);

  std::vector<Op *> growGradOps(Op *forwardOp);

  // starting from losses, construct the individual loss ops
  // as well as an op which sums them to get the final op
  void growFinalLoss();

  // for each of the losses described by loss, create a
  // grad-op
  std::vector<Op *> growLossGradients();

  // called from growFromNode and ...
  // T requires functions input(int) and input_size()
  template <typename T> void connectInputs(const T &, OpId opId);

  // T requires functions output(int) and output_size()
  template <typename T> void connectOutputs(const T &, OpId opId);

  const onnx::ModelProto onnxModel;

  // create an Op from a Node
  std::unique_ptr<Op> addOp(const Node &);
  std::map<OpId, std::unique_ptr<Op>> ops;

  // moves ownsership of created Op into the Graph,
  // and returns the Op's OpId (which it already has)
  OpId moveIntoGraph(std::unique_ptr<Op> op);

  // signal that a grad-op has created edge-gradients
  void registerOpGrads(Op *);

  void registerTensorGrad(Tensor *);

  TensorGradRegistry tensor_grad_registry;
  OpGradRegistry op_grad_registry;

  // total number of ops ever created
  OpId opsCounter{100};

  // The update ops which must be run during a training pass
  std::set<Op *> trainTargetOps;

  Op *finalLossOp{nullptr};

  // all in input() of all in node() of the onnx Graph
  void setAllNodeInputsMap();
  std::set<std::string> allNodeInputsMap;
  // only adds an init tensor if it is is allNodeInputsMap;
  void addInitIfUsed(TensorId id, const onnx::TensorProto *t);

  // run after creating the backwards pass, checks that
  // the user provided anchor tensors actually exist.
  // the user may have not used the correct gradient
  // tensor naming convention for example, this will
  // be caught here.
  void validateAnchors() const;
};

} // namespace neuralnet

#endif
