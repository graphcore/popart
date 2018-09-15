#ifndef GUARD_NEURALNET_GRAPH_HPP
#define GUARD_NEURALNET_GRAPH_HPP

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
// The protobuf generated ONNX classes
#include <onnx/onnx.pb.h>
#pragma clang diagnostic pop // stop ignoring warnings

#include <map>
#include <neuralnet/names.hpp>
#include <neuralnet/tensorinfo.hpp>

namespace neuralnet {


class Tensor;
class Graph;

// Tensors to log every iteration
// Also, frequency at which to return all weights
// TODO(jn) ask David Norman how tensorflow does this.
class Recorder {};

class Loss {
public:
  Loss() = default;
  virtual ~Loss() = default;

  // return onnx::NodeProto(s) representing this loss
  virtual std::vector<std::unique_ptr<Node>> getNodes() const = 0;
  virtual std::vector<TensorId> getStreamTensorNames() const = 0;

private:
};

// where tensor tenId is consumed by op opId at index index, 
// what is the name of the gradient along this edge?
TensorId getGradId(TensorId tenId, OpId opId, int index);


// the name of the tensor of the total gradient (loss and regularizers)
TensorId getGradId(TensorId tenId);


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
  const std::map<TensorId, TensorInfo> & getInfos() const;
  

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
  AVERAGEPOOL = 0,
  CONSTANT,
  CONV,
  LOGSOFTMAX,
  NEGLOGLIKE,
  PAD,
  RELU,
};


class TensorIndexMap {
public:
  void insert(int, Tensor *);
  // the Tensor at index changes. Note that there
  // must already be a Tensor at the index
  void reset(int, Tensor *);
  Tensor *tensor(int);
  const Tensor *tensor(int) const;
  const std::vector<int> &indices(Tensor *) const;
  const std::map<Tensor *, std::vector<int>> &indicesMap() const;
  // the number or indices (keys of tensor_map)
  int n() const;
  void append(std::stringstream &, std::string prefix) const;

private:
  std::map<int, Tensor *> tensor_map;
  std::map<Tensor *, std::vector<int>> indices_map;
};

class Attributes {
public:
  Attributes(decltype(onnx::NodeProto().attribute()) &);
  Attributes() = default;
  const std::vector<std::string> &getNames() const;
  onnxAttPtr at(std::string name) const;
  void append(std::stringstream &ss) const;
  template <typename T> void setIfPresent(T &, std::string s) const;

private:
  std::map<std::string, onnxAttPtr> att_map;
  std::vector<std::string> names;
};

template <> void Attributes::setIfPresent(int64_t &, std::string s) const;

template <>
void Attributes::setIfPresent(std::vector<int64_t> &, std::string s) const;

template <> void Attributes::setIfPresent(std::string &, std::string s) const;

class Op {
public:
  Op(OpId, const Node &, Graph *);
  // Op(OpId, OpType, std::string domain, Graph *);

  // wire an tensor to input. updates input, and
  // updates consumers of tensor with id TensorId
  void connectInTensor(InIndex, TensorId);

  // create an Activation (output) tensor
  // and wire it to this Ops output
  void createAndConnectOutTensor(OutIndex, TensorId);

  void append(std::stringstream &ss) const;
  virtual ~Op();

  // The consumed Tensors
  TensorIndexMap input;

  // The produced Tensors
  TensorIndexMap output;

  // might the input tensors be modified?
  bool mayModify(InIndex) const;

  // all Ops will be performed in the order of priority (highest to lowest);
  double priority() const;

  OpId id;

  const std::string &op_type() const;
  const OpType opType;

  const std::string & domain();
  Graph *pgraph;

  // Design choice : ALL Ops are created from an Node 
  // which is guaranteed to stay in scope as long 
  // as the Op is in scope.
  const Node *getNode() const;

  // attributes from the Node. Note that there
  // is some duplication between the attributes in 
  // in this Attributes, and those in the Node, but
  // the ones in this Attributes are stored for more efficient
  // access.
  const Attributes nAtts;

  // set shape and type parameters,
  // MUST set output TensorInfos
  virtual void setup();

  virtual Node getGradientPartner() const;

private:
  void appendIO(std::stringstream &) const;
  virtual void appendMore(std::stringstream &) const {}

  // if constructed from an onnx node:
  const Node *ptrNode;
  const std::string *const p_op_type;
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
  Tensor *get(TensorId);
  void remove(TensorId);
  bool contains(TensorId) const;
  // create a Tensor, either of type Const or Variable
  void addInit(TensorId, const onnx::TensorProto *);
  // create a Tensor of type Stream
  void addStream(TensorId);
  // create a Tensor of type Activation
  void addActivation(TensorId);
  std::vector<TensorId> getInitIds() const;
  std::vector<TensorId> getIds(TensorType) const;
  std::vector<TensorId> getNoProducerIds() const;
  const onnx::TensorProto *getOnnxInit(TensorId) const;

private:
  std::map<TensorId, std::unique_ptr<Tensor>> M;
  OnnxTensorPtrs init;
  Graph *pgraph;
};

class Graph {
public:


Graph(onnx::ModelProto &&,
             PreRunKnowledge &&,
             Recorder &&,
             std::unique_ptr<Loss> &&,
             std::vector<std::unique_ptr<Regularizer>> &&,
             // Schedule needed, if momentum the graph is different
             Schedule &&sched,
             // Weights tensors which are not to be updated
             std::vector<std::string> &&cTens);

  // take training steps
  onnx::ModelProto step(int n);
  // if the tensor is returned to user (Recorder).
  bool isLogged(TensorId);
  void append(std::stringstream &);
  PreRunKnowledge preRunKnowledge;
  Recorder recorder;
  std::unique_ptr<Loss> loss;
  std::vector<std::unique_ptr<Regularizer>> regularizers;
  Schedule schedule;
  Tensors tensors;
  // Activation, Gradient, Variable etc
  // TensorTypes tensorTypes;
  ~Graph();
  // split ConvOp with bias into two Ops, a ConvOp
  // followed by an x Op
  void splitConvBias();
  // Padding with edges of width 0 is a nop,
  // remove it unless logging tensors prevents
  void removePadSizeZero();
  // remove []->() where [] is Tensor and () is an Op and []->()
  // forms part of (.)->[]->()->[.]. after this, this section will
  // be (.)->[.]
  void removeNullOp(TensorId name, OpId opId);
  // return pointers to Ops of a certain type
  std::vector<Op *> opsOfType(OpType);
  void inferTensorInfos();
  // this does not take into priority, simple topological sort
  std::vector<Op *> getTopologicallySorted();

  void constructForwards();
  void constructBackwards();
  OpId getAndIncrOpsCounter();


private:

  // confirm that the names of the Const tensors 
  // from the user (constTensors) are in the onnx Model
  // Can be run after the forward pass of Graph has been 
  // constructed
  void confirmConstIds() const;

  // gradients are named automatically. To prevent them 
  // getting names already taken by non-gradiet tensors, 
  // we check that a reserved pattern is not present.
  void confirmNonGradId(TensorId tenId) const;

  // cofirm that no tensors in input(), nodes() or preRunKnowlede() 
  // use reserved naming conventions. A note on design: The decision
  // to NOT add an independent dimension to TensorId, used exclusively
  // by automatically named tensors, was that when printing TensorIds
  // there would still be the possibility of conflict (i.e. projection 
  // to single string might result in conflict). 
  void confirmNoGradIds() const;

  // create an Op from Node (if not Constant Node), wire it to
  // correct input Tensors and create the activation output Tensors
  Op * growFromNode(const Node &);


  // sets Node output and then calls growFromNode. 
  // the reason output is set here is that is might
  // Op *setNodeOutNamesAndGrowFrom(Node &node);

  const onnx::ModelProto onnxModel;

  // Nodes created during the building of the graph, includes
  // Nodes for the backwards pass.
  std::vector<std::unique_ptr<Node>> constructedNodes;

  // create an Op from a Node
  std::unique_ptr<Op> addOp(OpId, const Node &);
  std::map<OpId, std::unique_ptr<Op>> ops;
  OpId opsCounter{0};
};

} // namespace neuralnet

#endif
