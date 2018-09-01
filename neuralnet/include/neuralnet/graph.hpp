#ifndef GUARD_NEURALNET_GRAPH_HPP
#define GUARD_NEURALNET_GRAPH_HPP

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <onnx/onnx.pb.h>
#pragma clang diagnostic pop // stop ignoring warnings

#include <map>
#include <neuralnet/names.hpp>

namespace neuralnet {

// Tensors to log every iteration
// Also, frequency at which to return all weights
// TODO(jn) ask David Norman how tensorflow does this.
class Recorder {};

// Learning scheduler
// momentum, learning rates, etc.
class Schedule {};

enum class OpType {
  AVERAGEPOOL,
  CONSTANT,
  CONV, 
  PAD,
  RELU,
};


class Tensor;
class Graph;

class TensorIndexMap{
  public:
    void insert(int, Tensor*);
    Tensor * tensor(int);
    const std::vector<int> & indices(Tensor *);
    void append(std::stringstream &);

private:
    std::map<int, Tensor *> tensor_map;
    std::map<Tensor *, std::vector<int>> indices_map;
};


class Attributes{
  public:
    Attributes(decltype(onnx::NodeProto().attribute()) &);
    const std::vector<std::string>  & getNames();
    onnxAttPtr at(std::string name);
    void append(std::stringstream & ss);

  private:
    std::map<std::string, onnxAttPtr> att_map;
    std::vector<std::string> names;
};


class Op {
public:
  Op(OpId, const Node &, Graph *);

  // update input, and update 
  // consumers of tensor with id TensorId
  void connectInTensor(InIndex, TensorId);

  // create an Activation tensor
  // and connect it to this Op
  void createAndConnectOutTensor(OutIndex, TensorId);

  void append(std::stringstream & ss);
  virtual ~Op();

  // The consumed Tensors
  TensorIndexMap input;

  // The produced Tensors
  TensorIndexMap output;

  // might the input tensors be modified?
  bool mayModify(InIndex);

  // all Ops will be performed in the order of priority (highest to lowest);
  double priority();

  OpId opId;

  const  std::string op_type;
  const OpType opType;

  std::string domain;
  Graph * pgraph;

  bool fromNode();
  const Node * getNode();

private:
  void appendIO(std::stringstream &);
  virtual void appendMore(std::stringstream &) {}
  const Node * ptrNode;

};

class VectorAndSet {
  public: 
    VectorAndSet(std::vector<std::string> && vals);
    bool contains(std::string);
    const std::vector<std::string> & v();

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
        Recorder &&,
        // Schedule needed for Graph construction, 
        // as if there is momentum the graph is different
        Schedule &&,
        // Weights tensors which are not to be updated
        std::vector<std::string> &&constTensors);

  onnx::ModelProto step(int n);
  bool isInitTensor(TensorId);
  // if the tensor is returned to user (Recorder). 
  bool isLogged(TensorId);
  void addInitTensor(TensorId);
  void addStreamTensor(TensorId);
  void addActivationTensor(TensorId);
  Tensor * getTensor(TensorId);
  void append(std::stringstream &);
  Recorder recorder;
  Schedule schedule;
  VectorAndSet constTensorIds;
  OpTypes opTypes;
  TensorTypes tensorTypes;
  ~Graph();
  void validate();
  void splitConvBias();
  void removePadSizeZero();
  void remove(TensorId name, OpId opId);
  std::vector<Op *> opsOfType(OpType);

private:
  void growFromNode(const Node &);
  std::unique_ptr<Op> addOp(OpId, const Node &);
  const onnx::ModelProto onnxModel;
  std::map<TensorId, std::unique_ptr<Tensor>> tensors;
  std::map<OpId, std::unique_ptr<Op>> ops;
  OpId nOpsGenerated {0};
  OnnxTensors init;
};

}

#endif
