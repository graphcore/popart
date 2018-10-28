#ifndef GUARD_NEURALNET_GRAPH_HPP
#define GUARD_NEURALNET_GRAPH_HPP

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
// The protobuf generated ONNX classes
#include <onnx/onnx_pb.h>
#pragma clang diagnostic pop // stop ignoring warnings

#include <map>
#include <willow/attributes.hpp>
#include <willow/names.hpp>
#include <willow/tensorinfo.hpp>
#include <willow/vertex.hpp>

namespace willow {

class InputMapWrapper;

enum class AnchorReturnType {
  FINAL = 0, // return just the final batch of the step
  SUM,       // return the sum of all samples at the end
             // of the step
  ALL        // return all batches in the step, with writes
             // after each each batch. Exact mirror of input
             // data streams.
};
// Suppose we have an anchor scalar (0-d) tensor,
// Suppose batchesPerStep = 3 and samplesPerBatch = 2.
// Suppose that the 6 samples processed in a step have values
// 1, 2, 1, 0, 1, 3
// Then, under each of the AnchorReturnTypes the returned tensors are,
// FINAL : [1, 3]             (1-d tensor)
// SUM   : 8                  (0-d tensor)
// ALL   : [1, 2, 1, 0, 1, 3] (1-d tensor)

class DataFlow {
public:
  DataFlow(int batchesPerStep,
           int samplesPerBatch,
           const std::vector<TensorId> &,
           AnchorReturnType);

  bool isAnchored(TensorId) const;
  const std::vector<TensorId> &anchors() const;
  int nAnchors() const;
  int samplesPerBatch() const;
  int batchesPerStep() const;
  // TODO : AnchorReturnType can be made
  // tensor specific at no cost
  AnchorReturnType art() const;

private:
  // The number of batches between recording tensors
  int batchesPerStep_;
  // EXACTLY the batch size
  int samplesPerBatch_;
  // Note: there is no communication to the host
  // for batchesPerStep_ * samplesPerBatch_ samples.
  std::set<TensorId> s_anchors;
  const std::vector<TensorId> v_anchors;
  AnchorReturnType art_;
};

// the input tensor of a grad-op has what kind of
// relationship with the corresponding non-grad-op?
// design note: it's not possible for an input to a
// grad-op to NOT be directly related to
// the corresponding non-grad-op.
enum class GradOpInType { IN = 0, OUT, GRADOUT };

class GradInOutMapper {
public:
  GradInOutMapper(int iGrad_, int iNonGrad_, GradOpInType);
  // input index to a grad-op
  int iGrad;
  // input/output/gradient-of-output index to
  // corresponding non-grad op,
  int iNonGrad;
  // where input/output/gradient-of-output is
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

std::string getWillowDomain();

// where tensor tenId is consumed by Op with OpId opId at
// index index, what should the name of the edge-gradient
// along this edge be? This is purely string manipulation.
TensorId getEdgeGradId(TensorId tenId, OpId opId, int index);

// the name of the tensor of the total gradient
// (loss and regularizers)
TensorId getGradId(TensorId tenId);

// inverse of previous function
TensorId getNonGradId(TensorId tenId);

// get a recomputed tensor's name, based on original tensor
TensorId getRecompId(TensorId tenId);

// What is known about the Ir before it is run.
// This knowledge can sometimes be compiled into the Ir,
// and for certain backends is even required, for example
// Graphcore IPU requires all Stream Tensor shapes.
class EarlyInfo {
public:
  EarlyInfo() = default;
  void add(TensorId, const TensorInfo &);
  const TensorInfo &get(TensorId) const;
  bool has(TensorId) const;

  // return all unique TensorIds of tensors with any
  // information stored in this object, be it TensorInfo
  // or actual tensor.
  std::vector<TensorId> getAllTensorIds() const;

private:
  std::map<TensorId, TensorInfo> infos;
  // we will also have a map of actual tensors, these
  // can be used sometimes to compile the Graph (slice
  // indices for example)
};

enum class OpType {
  ADD = 0,
  ADDGRAD,
  AVERAGEPOOL,
  AVERAGEPOOLGRAD,
  CONSTANT,
  CONSTSGDVARUPDATE,
  CONV,
  CONVDATAGRAD,
  CONVWEIGHTSGRAD,
  L1,
  L1GRAD,
  LOGSOFTMAX,
  LOGSOFTMAXGRAD,
  LOGSOFTMAXGRADDIRECT,
  NLL,
  NLLGRAD,
  PAD,
  RELU,
  RELUGRAD,
  SGDVARUPDATE,
  SQUEEZE,
  SQUEEZEGRAD,
  SUM
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
  // The id of the Tensor at an index
  // == tensor(int)->id
  TensorId id(int) const;
  bool hasIndex(int) const;
  const std::vector<int> &indices(Tensor *) const;
  const std::map<Tensor *, std::vector<int>> &indicesMap() const;
  const std::map<int, Tensor *> &tensorMap() const;
  std::map<int, TensorId> tensorIdMap() const;
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
                      Ir *,
                      Attributes,
                      std::string domain);

  std::string op_type;
  Ir *pir;
  Attributes atts;
  std::string domain;
};

class Op : public Vertex {
public:
  Op(const Node &, Ir *);
  Op(const OpConstructorBundle &);
  Op(const Op &);
  Op &operator=(const Op &) = delete;
  // the rule-of-3 says that it's good
  // practise to have an explicit destructor,
  // given that there is an explict copy con.
  // But not really nec. as Vertex has a virtual
  // destructor.
  virtual ~Op() = default;

  std::string str() const;

  // create an ActGrad (output) tensor
  // and wire it to this Ops output
  void createAndConnectOutTensor(OutIndex, TensorId);

  void append(std::stringstream &ss) const;

  // sum of the total memory of all output tensors
  int64_t memOfOutputs() const;
  // We might want a cycle counter too.

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
  // default : 0.0
  double priority{0.0};

  // "Relu" or "Conv" etc.
  const std::string &op_type() const;
  const OpType opType;

  // political affiliation of the Op (same as NodeProto)
  const std::string &domain();

  // the Ir to which the Op belongs
  Ir *pir;

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
  // Why is this not constant? For one, nOps counter increments.
  virtual std::vector<std::unique_ptr<Op>> getGradOps();

  // return a gradient op's non-gradient creator, if relevant.A
  // if not relevant (this is a grad op etc) throws an error
  // Note : the creator might have been optimised out, in which
  // case calling this function has undefined bahaviour.
  virtual Op *getNonGradCreator() const;

  // Design choice.
  // as optimisations get complex we might
  // delete a non-grad op corresponding to a grad-op.
  // For this reason, we prefer NOT to store the non-grad
  // pointer directly by default (although we do in several cases)
  // Prefered: store the id and look up the pointer when
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

  // for non-grad-op `op', takes in the set of output indices
  // of `op' for which a gradient is available and returns
  // if all the gradients needed to create grad-ops are present
  // currently this will just compare the size of
  // the set passed in with number of paths to final loss
  bool readyToCreateGradients(std::set<int> &) const;

  virtual void imposeTopoCons();

  // return a copy of self, similar to
  // cpppatterns.com/patterns/virtual-constructor.html
  // fancy-pants people call it "covariant return type"
  virtual std::unique_ptr<Op> clone() const = 0;

  // note that this is virtual, and will
  // be overwritten by GradOp.
  virtual bool isGradOp() const { return false; }

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
  // no clone for GradOp currently, so will throw an error
  virtual std::unique_ptr<Op> clone() const override final;
  virtual ~GradOp() override = default;
  virtual int getNonGradInIndex(int) const override final;
  virtual bool isGradOp() const override final { return true; }
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
  VectorAndSet(const std::vector<std::string> &vals);
  bool contains(std::string) const;
  const std::vector<std::string> &v() const;
  ~VectorAndSet();

private:
  std::vector<std::string> v_vals;
  std::set<std::string> m_vals;
};

std::string reservedGradientPrefix();
std::string reservedRecomputePrefix();
std::vector<std::string> reservedPrefixes();

class Tensors {
public:
  Tensors(const std::vector<std::string> &vals1, Ir *pg);
  ~Tensors();
  // Store the Tensors of type Const
  const VectorAndSet constIds;
  Tensor *get(TensorId) const;
  void remove(TensorId);
  bool contains(TensorId) const;
  // create a Tensor, either of type Const or Variable
  void addInit(TensorId, const onnx::TensorProto *);
  // create a Tensor of type Stream
  void addStream(TensorId, const TensorInfo &);
  // create a Tensor of type ActGrad (basically any tensor which is
  // the output of an Op)
  void addActGrad(TensorId);
  std::vector<TensorId> getInitIds() const;
  std::vector<TensorId> getIds(TensorType) const;
  std::vector<TensorId> getNoProducerIds() const;
  const onnx::TensorProto *getOnnxInit(TensorId) const;
  void append(std::stringstream &) const;

private:
  std::map<TensorId, std::unique_ptr<Tensor>> M;
  // adds to M, but first confirms that TensorId not already in
  void insert(TensorId, std::unique_ptr<Tensor>);
  Ir *pir;
};

// Ir Constructor inputs
class IrBundle {
public:
  IrBundle(std::string fnModel_,
           const EarlyInfo &,
           const DataFlow &,
           const std::vector<Loss *> &,
           const Optimizer *,
           const std::vector<std::string> &cTens_,
           std::string logdir_,
           const std::vector<std::string> &patternNames_);

  std::string fnModel;
  const EarlyInfo &earlyInfo;
  const DataFlow &dataFlow;
  const std::vector<Loss *> &losses;
  const Optimizer *optimizer;
  // Weights tensors which are not to be updated
  const std::vector<std::string> &cTens;
  std::string logdir;
  const std::vector<std::string> &patternNames;
};

class Ir {
public:
  Ir(const IrBundle &);
  void updateOptimizer(const Optimizer *);
  // take training steps
  onnx::ModelProto step(int n);
  // if the tensor is returned to user (passes call to DataFlow).
  bool isAnchored(TensorId);
  void append(std::stringstream &);
  std::vector<std::unique_ptr<Loss>> losses;
  Tensors tensors;
  // The tensors specific to the optimization. Learning rate(s), momentum(s) etc
  std::vector<Tensor *> optimizerTensors() const;
  // The input data tensors. label(s), image(s), etc. This does not include
  // optimizer stream tensors (they are not data)
  std::vector<Tensor *> dataStreamTensors() const;
  ~Ir();
  // split ConvOp with bias into two Ops, a ConvOp
  // followed by an x Op TODO : move to Patterns
  void splitConvBias();
  std::vector<Op *> opsOfType(OpType);
  void inferTensorInfos();
  // this does not take into priority, simple topological sort
  std::vector<Op *> getTopologicallySorted() const;
  std::vector<Op *> getTopologicallySortedTilLoss() const;
  OpId getOpsCounter() const;
  OpId getAndIncrOpsCounter();
  TensorId getFinalLossId() const;
  Op *getFinalLossOp();
  void exportDot(std::string dotfn) const;
  void eraseOp(OpId);
  Op *getOp(OpId);
  const DataFlow dataFlow;

  // see connectInputs, this is just an instantiation of it
  void connectInputsFromInputMapWrapper(const InputMapWrapper &, OpId opId);

  // moves ownership of created Op into the Ir,
  // and returns the Op's OpId
  OpId moveIntoIr(std::unique_ptr<Op> op);

private:
  // called from growFromNode and many other places where Ops created
  // T requires functions input(int) and input_size()
  template <typename T> void connectInputs(const T &, OpId opId);

  // T requires functions output(int) and output_size()
  template <typename T> void connectOutputs(const T &, OpId opId);

  // learning rate, momentum, etc.
  // Optimizer needed to construct backwards pass:
  // if momentum the Ir is different
  std::unique_ptr<Optimizer> optimizer{nullptr};
  std::string logdir;
  EarlyInfo earlyInfo;

  void constructForwards();
  void constructBackwards();
  // remove nodes an tensors which are not
  // needed to arrive at the target
  void prune();

  void addRecompute();
  // for all tensors in the forward pass, set the number of
  // paths to the final loss (needed in the backwards pass)
  void setNPathsToLoss();

  // modify the Ir using with pattern matching
  void applyPattern(const Pattern *);

  // patterns to apply after constructing forwards and backwards passes
  std::vector<std::unique_ptr<Pattern>> patterns;

  // confirm that the names of the Const tensors
  // from the user (constTensors) are in the onnx Model
  // Can be run after the forward pass of Ir has been
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

  Op *growRecomputeOp(Op *oriOp, const std::set<Op *> &checkpoints);

  Op *growGradSumOp(Tensor *target, const std::vector<Tensor *> &toSum);

  std::vector<Op *> growGradOps(Op *forwardOp);

  // starting from losses, construct the individual loss ops
  // as well as an op which sums them to get the final op
  void growFinalLoss();

  // for each of the losses described by loss, create a
  // grad-op
  std::vector<Op *> growLossGradients();

  onnx::ModelProto onnxModel;

  // create an Op from a Node
  std::unique_ptr<Op> addOp(const Node &);
  std::map<OpId, std::unique_ptr<Op>> ops;

  // total number of ops ever created
  OpId opsCounter{100};

  // The update ops which must be run during a training pass
  std::set<Op *> trainTargetOps;

  Op *finalLossOp{nullptr};

  // all in input() of all in node() of the onnx::Graph
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
};

} // namespace willow

#endif
