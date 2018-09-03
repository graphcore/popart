#ifndef GUARD_NEURALNET_GRAPH_HPP
#define GUARD_NEURALNET_GRAPH_HPP

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <onnx/onnx.pb.h>
#pragma clang diagnostic pop // stop ignoring warnings

#include <map>
#include <neuralnet/names.hpp>
#include <neuralnet/tensorinfo.hpp>

namespace neuralnet {

// Tensors to log every iteration
// Also, frequency at which to return all weights
// TODO(jn) ask David Norman how tensorflow does this.
class Recorder {};

// Learning scheduler
// momentum, learning rates, etc.
class Schedule {};


// What is known about the graph before it is run. 
// This knowledge can be compiled into the Graph, 
// and for certain backends is even required, for example
// Graphcore IPU requires all Stream Tensor shapes.
class PreRunKnowledge {
  public:
  PreRunKnowledge() = default;
  void addInfo(TensorId, const TensorInfo &);
  const TensorInfo & getInfo(TensorId);
  bool hasInfo(TensorId);

  private:
  std::map<TensorId, TensorInfo> infos;
  // we will also have a map of actual tensors, these
  // can be used sometimes to compile the graph (slice 
  // indices for example)

};

enum class OpType {
  AVERAGEPOOL,
  CONSTANT,
  CONV,
  LOGSOFTMAX,
  PAD,
  RELU,
};

class Tensor;
class Graph;

class TensorIndexMap {
public:
  void insert(int, Tensor *);
  // the Tensor at index changes. Note that there
  // must already be a Tensor at the index
  void reset(int, Tensor *);
  Tensor *tensor(int);
  const std::vector<int> &indices(Tensor *);
  void append(std::stringstream &);

private:
  std::map<int, Tensor *> tensor_map;
  std::map<Tensor *, std::vector<int>> indices_map;
};

class Attributes {
public:
  Attributes(decltype(onnx::NodeProto().attribute()) &);
  const std::vector<std::string> &getNames();
  onnxAttPtr at(std::string name);
  void append(std::stringstream &ss);

private:
  std::map<std::string, onnxAttPtr> att_map;
  std::vector<std::string> names;
};

class Op {
public:
  Op(OpId, const Node &, Graph *);

  // wire an tensor to input. updates input, and
  // updates consumers of tensor with id TensorId
  void connectInTensor(InIndex, TensorId);

  // create an Activation (output) tensor
  // and wire it to this Ops output
  void createAndConnectOutTensor(OutIndex, TensorId);

  void append(std::stringstream &ss);
  virtual ~Op();

  // The consumed Tensors
  TensorIndexMap input;

  // The produced Tensors
  TensorIndexMap output;

  // might the input tensors be modified?
  bool mayModify(InIndex);

  // all Ops will be performed in the order of priority (highest to lowest);
  double priority();

  OpId id;

  const std::string op_type;
  const OpType opType;

  std::string domain;
  Graph *pgraph;

  // was this Op created from an onnx node?
  bool fromNode();
  const Node *getNode();

private:
  void appendIO(std::stringstream &);
  virtual void appendMore(std::stringstream &) {}
  const Node *ptrNode;
};

class VectorAndSet {
public:
  VectorAndSet(std::vector<std::string> &&vals);
  bool contains(std::string);
  const std::vector<std::string> &v();

private:
  std::vector<std::string> v_vals;
  std::set<std::string> m_vals;
};

enum class TensorType;
class TensorTypes {
public:
  TensorTypes();
  std::string asString(TensorType);

private:
  std::map<TensorType, std::string> tensor_types_m;
};

class OpTypes {
public:
  OpTypes();
  OpType get(std::string op_type);

private:
  std::unordered_map<std::string, OpType> opTypeMap;
};

class Graph {
public:
  Graph(onnx::ModelProto &&,
        PreRunKnowledge &&,
        Recorder &&,
        // Schedule needed for Graph construction,
        // as if there is momentum the graph is different
        Schedule &&,
        // Weights tensors which are not to be updated
        std::vector<std::string> &&constTensors);

  // take training steps
  onnx::ModelProto step(int n);
  // if the tensor is returned to user (Recorder).
  bool isLogged(TensorId);
  // create a Tensor, either of type Const or Variable
  void addInitTensor(TensorId);
  // create a Tensor of type Stream
  void addStreamTensor(TensorId);
  // create a Tensor of type Activation
  void addActivationTensor(TensorId);
  Tensor *getTensor(TensorId);
  void append(std::stringstream &);
  PreRunKnowledge preRunKnowledge;
  Recorder recorder;
  Schedule schedule;
  // Store the Tensors of type Const
  VectorAndSet constTensorIds;
  OpTypes opTypes;
  TensorTypes tensorTypes;
  ~Graph();
  // run logic checks on the graph
  void validate();
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

private:
  // create an Op from Node (if not Constant Node), wire it to
  // correct input Tensors and create the activation output Tensors
  void growFromNode(const Node &);
  // create an Op from a Node
  std::unique_ptr<Op> addOp(OpId, const Node &);

  const onnx::ModelProto onnxModel;
  std::map<TensorId, std::unique_ptr<Tensor>> tensors;
  std::map<OpId, std::unique_ptr<Op>> ops;
  OpId nOpsGenerated{0};
  OnnxTensors init;
};

} // namespace neuralnet

#endif
