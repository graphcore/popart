#ifndef GUARD_NEURALNET_GRAPH_HPP
#define GUARD_NEURALNET_GRAPH_HPP

// The protobuf generated ONNX classes
#include <onnx/onnx_pb.h>

#include <map>
#include <poponnx/attributes.hpp>
#include <poponnx/names.hpp>
#include <poponnx/optionflags.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/vertex.hpp>

namespace willow {

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

class InputMapWrapper;

// An anchor tensor is a tensor which the user wants returned
// after a step is run. Anchors are essentially what tensorflow calls
// "fetches". AnchorReturnType specifies what exactly should be
// returned for a tensor, currently the 3 options are:

enum class AnchorReturnType {
  FINAL = 0, // return just the final batch of the step
  SUM,       // return the sum of all samples at the end
             // of the step
  ALL        // return all batches in the step.
};
// As an example, suppose we have an anchor scalar (0-d) tensor,
// Suppose batchesPerStep = 3 and samplesPerBatch = 2.
// Suppose that the 2*3 = 6 samples processed in a step have values
// 1, 2, 1, 0, 1, 3
// Then, under each of the AnchorReturnTypes the returned tensors are,
// FINAL : [1, 3]             (1-d tensor)
// SUM   : 8                  (0-d tensor)
// ALL   : [1, 2, 1, 0, 1, 3] (1-d tensor)

// Describe what and when the user wants returned.
class DataFlow {
public:
  DataFlow();
  DataFlow(int batchesPerStep,
           int samplesPerBatch,
           const std::vector<TensorId> &,
           AnchorReturnType);

  bool isAnchored(TensorId) const;
  const std::vector<TensorId> &anchors() const;
  int nAnchors() const;
  // batch-size:
  int samplesPerBatch() const;
  int batchesPerStep() const;
  // samplesPerBatch() * batchesPerStep():
  int samplesPerStep() const;
  AnchorReturnType art() const;

private:
  // The number of batches between returning anchors
  // to the user. There is no communication between the
  // user and Willow while batchesPerStep_ batches are processed.
  int batchesPerStep_;
  // EXACTLY the batch-size
  int samplesPerBatch_;
  // There is no communication to the host while
  // batchesPerStep_ * samplesPerBatch_ samples are processed
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
  // "input/output/gradient-of-output" index to
  // corresponding non-grad op,
  int iNonGrad;
  // where "input/output/gradient-of-output" above is
  GradOpInType type;

  bool operator==(const GradInOutMapper &rhs) const;
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

std::string getWillowDomain();

// where tensor tenId is consumed by Op with OpId opId at
// index "index", what should the name of the edge-gradient
// along this edge be? This is pure string manipulation.
TensorId getEdgeGradId(TensorId tenId, OpId opId, int index);

// the name of the tensor which is the
// total gradient of a forward tensor
TensorId getGradId(TensorId tenId);

// inverse of previous function (non-grad name of grad tensor)
TensorId getNonGradId(TensorId tenId);

// get a recomputed tensor's name, based on original tensor
TensorId getRecompId(TensorId tenId);

// What is known about the Ir before it is run.
// This knowledge can sometimes be compiled into the Ir,
// and for certain backends is even required, for example
// the IPU requires all Stream Tensor shapes.
// In the future (TODO T5252) it will also contain indices for slicing
// tensors (I think the LSTM from pytorch might require this)
class EarlyInfo {
public:
  EarlyInfo() = default;
  void add(TensorId, const TensorInfo &);
  const TensorInfo &get(TensorId) const;
  bool has(TensorId) const;

  // return all unique TensorIds of tensors with any
  // information stored in this object, either TensorInfo
  // or an actual tensor
  std::vector<TensorId> getAllTensorIds() const;

private:
  std::map<TensorId, TensorInfo> infos;
  // we will also have a map of actual tensors, these
  // can be used sometimes to compile the Graph (slice
  // indices for example) (TODO T5252)
};

enum class OpType {
  ADD = 0,
  ADDGRAD,
  ADDBIAS,
  ADDBIASDATAGRAD,
  ADDBIASBIASGRAD,
  AVERAGEPOOL,
  AVERAGEPOOLGRAD,
  CONSTANT,
  CONSTSGDVARUPDATE,
  CONV,
  CONVDATAGRAD,
  CONVWEIGHTSGRAD,
  IDENTITY,
  IDENTITYGRAD,
  L1,
  L1GRAD,
  SOFTMAX,
  SOFTMAXGRAD,
  SOFTMAXGRADDIRECT,
  NEGATE,
  NEGATEGRAD,
  NLL,
  NLLGRAD,
  MATMUL,
  MATMULLHSGRAD,
  MATMULRHSGRAD,
  PAD,
  REDUCESUM,
  REDUCESUMGRAD,
  RELU,
  RELUGRAD,
  SGDVARUPDATE,
  SQUEEZE,
  SQUEEZEGRAD,
  SUBTRACT,
  SUBTRACTGRAD,
  SUM
};

// Inputs and outputs to Ops will use this class.
// inputs (outputs) enter (leave) at certain indices
// of an Op. There is 1 tensor per index,
// but 1+ index per tensor.
class TensorIndexMap {
public:
  void insert(int, Tensor *);
  // the Tensor at index changes. Note that there
  // must already be a Tensor at the index (otherwise insert should be used)
  void reset(int, Tensor *);
  // Remove the Tensor index from the tensorMap.
  // If the Tensor is not referred to by any indices, it is removed from the
  // indicesMap.
  void erase(int);
  // get the Tensor at at index
  Tensor *tensor(int);
  const Tensor *tensor(int) const;
  // The id of the Tensor at an index
  // This is just a helper function (same as tensor(int)->id)
  TensorId id(int) const;
  bool hasIndex(int) const;
  const std::vector<int> &indices(Tensor *) const;
  const std::map<Tensor *, std::vector<int>> &indicesMap() const;
  const std::map<int, Tensor *> &tensorMap() const;
  std::map<int, TensorId> tensorIdMap() const;
  // the number or indices. Exactly the number of keys of tensor_map
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
  // Note: copy constructor does NOT copy input and output
  Op(const Op &);
  Op &operator=(const Op &) = delete;
  // A c++ aside: the rule-of-3 says that it's good
  // practise to have an explicit destructor,
  // given that there is an explict copy con.
  // But not really nec. as Vertex has a virtual
  // destructor.
  virtual ~Op() = default;

  std::string str() const;

  // create an ActGrad (output) tensor
  // and wire it to this Op's output
  void createAndConnectOutTensor(OutIndex, TensorId);

  void append(std::stringstream &ss) const;

  // sum of the total memory of all output tensors
  // We might want a cycle counter too for more sophisticated recomputation
  int64_t memOfOutputs() const;

  // The consumed Tensors
  TensorIndexMap input;

  // The produced Tensors
  TensorIndexMap output;

  // wire a tensor to input: updates input and
  // updates consumers of tensor with id TensorId
  void connectInTensor(InIndex, TensorId);

  // might the input tensors be modified?
  bool mayModify(InIndex) const;

  // all Ops will be topologically sorted "as close to" the order of
  // priority (highest to lowest) while still resulting in a valid
  // topological ordering.
  // default : 0.0
  double priority{0.0};

  // "Relu" or "Conv" etc.
  const std::string &op_type() const;
  const OpType opType;

  // political affiliation of the Op
  // same domain as from the NodeProto if constructed from ONNX
  const std::string &domain();

  // the Ir to which the Op belongs
  Ir *pir;

  // The unique identifier of the Op (will always be set in Op::Op)
  OpId id{-1};

  // attributes from the Node, if it was created from ONNX
  const Attributes nAtts;

  // set shape and type parameters,
  // This function MUST set output
  // TensorInfos for all outputs
  virtual void setup();

  // return a vector of 1 or several gradient Ops: for
  // obtaining the gradient of the inputs of this Op.
  // If this Op is already a gradient Op, throws error
  // Why is this not constant? For one, nOps counter increments.
  virtual std::vector<std::unique_ptr<Op>> getGradOps();

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
  // some people call it "covariant return type"
  // Throws error from this class if not implemented
  virtual std::unique_ptr<Op> clone() const;

private:
  void appendIO(std::stringstream &) const;
  virtual void appendMore(std::stringstream &) const {}
  const std::string *const p_op_type;
  std::string op_domain;

  // design decision : see-sawing between storing a pointer
  // to the Node from which the Op derives (if it does derive
  // from a Node) or not. Deciding not to for now, (1) not much
  // to copy to the Op (2) cleaner
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
  IrBundle(const onnx::ModelProto &modelProto,
           const EarlyInfo &earlyInfo,
           const DataFlow &dataFlow,
           const std::vector<Loss *> &losses,
           const Optimizer *optimizer,
           const std::vector<std::string> &cTens,
           const std::string &logdir,
           const SessionOptions &userOptions,
           const std::vector<std::string> &patternNames);

  const onnx::ModelProto &modelProto;
  const EarlyInfo &earlyInfo;
  const DataFlow &dataFlow;
  const std::vector<Loss *> &losses;
  const Optimizer *optimizer;
  // Weights tensors which are not to be updated
  const std::vector<std::string> &cTens;
  std::string logdir;
  const SessionOptions &userOptions;
  const std::vector<std::string> &patternNames;
};

class Ir {
public:
  ~Ir();
  Ir();
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
  // split ConvOp with bias into two Ops, a ConvOp
  // followed by an x Op TODO : move to Patterns (see task T5098)
  void splitConvBias();
  std::vector<Op *> opsOfType(OpType);
  // this does not take into priority, simple topological sort
  std::vector<Op *> getOpSchedule() const;
  std::vector<Op *> getOpScheduleTilLoss() const;
  OpId getOpsCounter() const;
  OpId getAndIncrOpsCounter();
  TensorId getFinalLossId() const;
  Op *getFinalLossOp();
  void exportDot(std::string dotfn) const;
  void eraseOp(OpId);
  Op *getOp(OpId);
  const DataFlow dataFlow;
  onnx::ModelProto getModel() const;

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
  std::unique_ptr<Optimizer> optimizer;
  std::string logdir;
  SessionOptions userOptions;
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

  // for each of the losses described by loss,
  // create a grad-op. Return a vector of {gradop, lossop} pairs
  std::vector<GradNonGradPair> growLossGradients();

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
