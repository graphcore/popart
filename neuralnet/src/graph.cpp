// TIMELINE:
// 1) support basic conv nets.

#include <map>
#include <neuralnet/error.hpp>
#include <neuralnet/graph.hpp>
#include <neuralnet/tensor.hpp>
#include <sstream>

#include <neuralnet/averagepool.hpp>
#include <neuralnet/conv.hpp>
#include <neuralnet/relu.hpp>

namespace neuralnet {

Op::~Op()       = default;
Graph::~Graph() = default;

template <class T> void appendSequence(std::stringstream &ss, T t) {
  int index = 0;
  ss << '[';
  for (auto &x : t) {
    if (index != 0) {
      ss << ' ';
    }
    ss << x;
    ++index;
  }
  ss << ']';
}

void TensorIndexMap::insert(int index, Tensor *ptensor) {
  tensor_map[index] = ptensor;
  auto found        = indices_map.find(ptensor);
  if (found == indices_map.end()) {
    indices_map[ptensor] = {index};
  } else {
    indices_map[ptensor].push_back(index);
  }
}

void TensorIndexMap::reset(int index, Tensor *ptensor) {
  auto previous = tensor_map[index];

  tensor_map[index] = ptensor;
  indices_map[ptensor].push_back(index);

  // clean up previous tensor
  std::vector<int> newIndices;
  for (auto &ind : indices_map[previous]) {
    if (ind != index) {
      newIndices.push_back(ind);
    }
  }
  if (newIndices.size() != 0) {
    indices_map[previous] = newIndices;
  } else {
    indices_map.erase(previous);
  }
}

Tensor *TensorIndexMap::tensor(int index) { return tensor_map[index]; }

const std::vector<int> &TensorIndexMap::indices(Tensor *ptensor) {
  return indices_map[ptensor];
}

void Op::connectInTensor(InIndex inIndex, TensorId tenId) {
  Tensor *ptensor = pgraph->getTensor(tenId);
  input.insert(inIndex, ptensor);
  ptensor->consumers.increment(this);
}

void Op::createAndConnectOutTensor(OutIndex outIndex, TensorId tenId) {
  pgraph->addActivationTensor(tenId);
  Tensor *ptensor = pgraph->getTensor(tenId);
  output.insert(outIndex, ptensor);
  ptensor->producer = this;
}

Tensor *Graph::getTensor(TensorId tenId) {
  auto found = tensors.find(tenId);
  if (found == tensors.end()) {
    throw error("no tensor with id " + tenId);
  }
  return found->second.get();
}

TensorTypes::TensorTypes() {
  tensor_types_m = {{TensorType::Activation, "Activation"},
                    {TensorType::Const, "Const"},
                    {TensorType::Gradient, "Gradient"},
                    {TensorType::Momentum, "Momentum"},
                    {TensorType::Other, "Other"},
                    {TensorType::Stream, "Stream"},
                    {TensorType::Unknown, "Unknown"},
                    {TensorType::Variable, "Variable"}};

  if (tensor_types_m.size() != static_cast<uint64_t>(TensorType::N)) {
    throw error("missing element in TensorTypes");
  }
}

std::string TensorTypes::asString(TensorType type) {
  return tensor_types_m[type];
}

// neuralnet streams and prints are "impolite" (will not add new line at end)

void Op::append(std::stringstream &ss) {
  appendIO(ss);
  ss << '\n';
  appendMore(ss);
}

void TensorIndexMap::append(std::stringstream &ss) {
  ss << '(';
  int index = 0;
  for (auto &index_tensor : tensor_map) {
    if (index != 0) {
      ss << ' ' << ' ';
    }
    ss << '@' << index_tensor.first << ':' << index_tensor.second->id << ':'
       << index_tensor.second->tensor_type;
    ++index;
  }
  ss << ')';
}

void Op::appendIO(std::stringstream &ss) {
  input.append(ss);
  ss << " --> [" << id << ':' << op_type << "] --> ";
  output.append(ss);
}

VectorAndSet::VectorAndSet(std::vector<std::string> &&vals) : v_vals(vals) {
  for (auto &v : v_vals) {
    m_vals.insert(v);
  }
}

Graph::Graph(onnx::ModelProto &&inMod,
             Recorder &&rec,
             // Schedule needed, if momentum the graph is different
             Schedule &&sched,
             // Weights tensors which are not to be updated
             std::vector<std::string> &&cTens)
    : recorder(rec), schedule(sched), constTensorIds(std::move(cTens)),
      onnxModel(inMod) {

  auto &onnxGraph = onnxModel.graph();
  auto &onnxNodes = onnxGraph.node();

  std::set<TensorId> onnxInitializers;
  for (const auto &initializer : onnxGraph.initializer()) {
    TensorId tenId = initializer.name();
    init[tenId]    = initializer;
    addInitTensor(tenId);
    onnxInitializers.emplace(tenId);
  }

  // onnx inputs which are not initializers are true inputs
  for (auto &valueInfo : onnxGraph.input()) {
    if (onnxInitializers.count(valueInfo.name()) == 0) {
      addStreamTensor(valueInfo.name());
    }
  }

  for (const auto &node : onnxNodes) {
    growFromNode(node);
  }

  validate();

  splitConvBias();

  removePadSizeZero();
}

// note : don't try too hard if tensors are logged,
// user is probably not concerned about performance

void Graph::removePadSizeZero() {
  for (auto &op : opsOfType(OpType::PAD)) {
    if (!isLogged(op->input.tensor(0)->id)) {
      std::cout << "removing pad of size 0" << std::endl;
      removeNullOp(op->input.tensor(0)->id, op->id);
    }
  }
}

std::vector<Op *> Graph::opsOfType(OpType opType) {
  std::vector<Op *> typedOps;
  for (auto &id_op : ops) {
    if (id_op.second->opType == opType) {
      typedOps.push_back(id_op.second.get());
    }
  }
  return typedOps;
}

void Graph::removeNullOp(TensorId name, OpId opId) {
  // [] (see header of ascii picture definitions)
  Tensor *tensorIn = tensors[name].get();
  // ()
  Op *op = ops[opId].get();
  // [.]
  Tensor *tensorOut = op->output.tensor(0);
  // (.)
  auto op0 = tensorIn->producer;
  // [.] gets all consumers of [] other than ()
  tensorOut->consumers.extend(tensorIn->consumers.getMap());
  tensorOut->consumers.decrement(op);
  // (.) produces [.] directly
  int index = op0->output.indices(tensorIn)[0];
  op0->output.reset(index, tensorOut);
  // delete []
  tensors.erase(name);
  // delete (.)
  ops.erase(opId);
}

bool Graph::isLogged(TensorId tenId) {
  (void)tenId;
  return false;
}

void Graph::splitConvBias() {}

const std::vector<std::string> &VectorAndSet::v() { return v_vals; }

void Graph::validate() {
  for (auto &tensorId : constTensorIds.v()) {
    if (tensors.find(tensorId) == tensors.end()) {
      throw error("no tensor " + tensorId +
                  " in graph, error in const tensor names");
    }
  }
}

bool VectorAndSet::contains(std::string name) {
  return m_vals.count(name) == 1;
}

void Graph::addInitTensor(TensorId name) {
  tensors[name] = std::unique_ptr<Tensor>(new Tensor(
      name,
      constTensorIds.contains(name) ? TensorType::Const : TensorType::Variable,
      this));
}

void Graph::addActivationTensor(TensorId tenId) {
  tensors[tenId] =
      std::unique_ptr<Tensor>(new Tensor(tenId, TensorType::Activation, this));
}

void Graph::addStreamTensor(TensorId tenId) {
  tensors[tenId] =
      std::unique_ptr<Tensor>(new Tensor(tenId, TensorType::Stream, this));
}

const std::vector<std::string> &Attributes::getNames() { return names; }

onnxAttPtr Attributes::at(std::string name) { return att_map[name]; }

Attributes::Attributes(decltype(onnx::NodeProto().attribute()) &attributes) {
  for (auto &attribute : attributes) {
    auto name = attribute.name();
    names.push_back(name);
    att_map[name] = &attribute;
  }
}

void Attributes::append(std::stringstream &ss) {
  using AttPro = onnx::AttributeProto;
  for (auto &name : names) {
    ss << '\n';
    ss << "  " << name << "  ";
    auto attptr = att_map[name];
    switch (attptr->type()) {
    case AttPro::UNDEFINED: {
      break;
    }
    case AttPro::FLOAT: {
      ss << attptr->f();
      break;
    }
    case AttPro::INT: {
      ss << attptr->i();
      break;
    }
    case AttPro::STRING: {
      ss << attptr->s();
      break;
    }
    case AttPro::TENSOR: {
      break;
    }
    case AttPro::GRAPH: {
      break;
    }
    case AttPro::FLOATS: {
      appendSequence(ss, attptr->floats());
      break;
    }
    case AttPro::INTS: {
      appendSequence(ss, attptr->ints());
      break;
    }
    case AttPro::STRINGS: {
      appendSequence(ss, attptr->strings());
      break;
    }
    case AttPro::TENSORS: {
      break;
    }
    case AttPro::GRAPHS: {
      break;
    }
    }
  }
}

void Graph::growFromNode(const onnx::NodeProto &node) {

  int nInputs = 0;
  for (auto &id : node.input()) {
    if (id != "") {
      ++nInputs;
    }
  }

  auto atts = Attributes(node.attribute());
  if (opTypes.get(node.op_type()) == OpType::CONSTANT) {
    TensorId name = node.output(0);
    init[name]    = node.attribute(0).t();
    addInitTensor(name);
  }

  // handling this CONV as a special case, as it
  // needs splitting into 2 (CONV and add bias)
  // will still add, changed in optimise step.
  // this first step builds exactly 1-1 with onnx graph
  else if (opTypes.get(node.op_type()) == OpType::CONV && nInputs == 3) {
    throw error("Conv with bias case not handled");
  }

  else {
    OpId opId = nOpsGenerated;
    ops[opId] = addOp(nOpsGenerated++, node);

    for (int inIndex = 0; inIndex < node.input_size(); ++inIndex) {
      auto &inName = node.input(inIndex);
      if (inName == "") {
        // no input at this position
      } else {
        auto found = tensors.find(inName);
        if (found == tensors.end()) {
          throw error("input should already be in tensor map");
        } else {
          ops[opId]->connectInTensor(inIndex, inName);
        }
      }
    }

    for (int outIndex = 0; outIndex < node.output_size(); ++outIndex) {
      auto &outName = node.output(outIndex);
      if (outName == "") {
        // no output at this position
      } else {
        // ONNX specifies that a tensor is the output of at most 1 node.
        ops[opId]->createAndConnectOutTensor(outIndex, outName);
      }
    }
  }
}

OpTypes::OpTypes() {

  opTypeMap = {{"AveragePool", OpType::AVERAGEPOOL},
               {"Conv", OpType::CONV},
               {"Relu", OpType::RELU},
               {"Constant", OpType::CONSTANT},
               {"Pad", OpType::PAD}};
}

OpType OpTypes::get(std::string op_type) {
  auto found = opTypeMap.find(op_type);
  if (found == opTypeMap.end()) {
    throw error("No OpType found for " + op_type);
  }
  return found->second;
}

bool Op::fromNode() { return ptrNode != nullptr; }

const Node *Op::getNode() {
  if (!fromNode()) {
    throw error("Op not from node");
  }
  return ptrNode;
}

void Graph::append(std::stringstream &ss) {
  ss << "-- Graph --\n";
  for (auto &id_op : ops) {
    id_op.second->append(ss);
  }
}

Op::Op(OpId id_, const onnx::NodeProto &node, Graph *pg)
    : id(id_), op_type(node.op_type()), opType(pg->opTypes.get(op_type)),
      domain(node.domain()), pgraph(pg) {}

std::unique_ptr<Op> Graph::addOp(OpId opId, const onnx::NodeProto &node) {
  using pOp = std::unique_ptr<Op>;
  switch (opTypes.get(node.op_type())) {
  case OpType::AVERAGEPOOL: {
    return pOp(new AveragePoolOp(opId, node, this));
  }
  case OpType::CONSTANT: {
    throw error("ILE. Constant Ops are not to be added");
  }
  case OpType::CONV: {
    return pOp(new ConvOp(opId, node, this));
  }
  case OpType::RELU: {
    return pOp(new ReluOp(opId, node, this));
  }
  case OpType::PAD: {
    return pOp(new ReluOp(opId, node, this));
  }
  default: { throw error("No class for " + node.op_type()); }
  }
}
class Graph;

} // namespace neuralnet
