// TIMELINE:
// 1) support basic conv nets.

#include <map>
#include <neuralnet/error.hpp>
#include <neuralnet/graph.hpp>
#include <neuralnet/tensor.hpp>
#include <neuralnet/tensorinfo.hpp>
#include <sstream>
#include <vector>

// The layers:
#include <neuralnet/averagepool.hpp>
#include <neuralnet/conv.hpp>
#include <neuralnet/logsoftmax.hpp>
#include <neuralnet/pad.hpp>
#include <neuralnet/relu.hpp>

namespace neuralnet {


void Op::setOutputInfos() const{
  throw error("No setOutputInfos for " + op_type);
}

std::vector<Op *> Graph::getTopologicallySorted(){

  std::vector<Op *> sorted;
  std::vector<Op *> opsToProcess;
  // map from each op to the number of input 
  // indices it is waiting on initialised as 
  // total number of input indices
  std::map<Op *, int> awaiting;
  for (auto & id_op : ops){
    awaiting[id_op.second.get()] = id_op.second->input.n();
  } 

  // processing a tensor involves 
  // reducing the counts in awaiting for
  // ops which use it, and detecting which
  // ops have nothing left to wait for as a 
  // result of such updating. 
  auto processTensor = [&opsToProcess, &awaiting](Tensor * tensor){
    for (auto & op_count  :  tensor->consumers.getMap()){
      awaiting[op_count.first] -= op_count.second;
      if (awaiting[op_count.first] == 0){
        opsToProcess.push_back(op_count.first);
      }
    }
  };

  // we will start by processing
  // the tensors which have no producers
  auto t0 = tensors.getNoProducerIds();
  for (auto & id : t0){
    processTensor(tensors.get(id));
  }

  while (opsToProcess.size() != 0){
    auto op = opsToProcess.back();
    opsToProcess.resize(opsToProcess.size() - 1);
    sorted.push_back(op);
    for (auto & tensor_indices : op->output.indicesMap()){
      processTensor(tensor_indices.first);
    }
  }

  if (sorted.size() != ops.size()){
    throw error("failure to sort topologically");
  }
  return sorted;
}

std::vector<TensorId> Tensors::getInitIds()  const{
  std::vector<TensorId> initIds;
  for (auto &id_pt : M) {
    if (id_pt.second->tensorType() == TensorType::Const ||
        id_pt.second->tensorType() == TensorType::Variable) {
      initIds.push_back(id_pt.first);
    }
  }
  return initIds;
}

std::vector<TensorId> Tensors::getIds(TensorType type)  const{
  std::vector<TensorId> ids;
  for (auto &id_pt : M) {
    if (id_pt.second->tensorType() == type){
      ids.push_back(id_pt.first);
    }
  }
  return ids;
}

Tensors::Tensors(std::vector<std::string> &&vals1, Graph *pg)
    : constIds(std::move(vals1)), pgraph(pg) {}

Op::~Op()       = default;
VectorAndSet::~VectorAndSet() = default;
Tensors::~Tensors() = default;
Graph::~Graph() = default;

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

const std::vector<int> &TensorIndexMap::indices(Tensor *ptensor)  const{
  return indices_map.at(ptensor);
}

void Op::connectInTensor(InIndex inIndex, TensorId tenId) {
  Tensor *ptensor = pgraph->tensors.get(tenId);
  input.insert(inIndex, ptensor);
  ptensor->consumers.increment(this);
}


void Op::createAndConnectOutTensor(OutIndex outIndex, TensorId tenId) {
  pgraph->tensors.addActivation(tenId);
  Tensor *ptensor = pgraph->tensors.get(tenId);
  output.insert(outIndex, ptensor);
  ptensor->producer = this;
}

Tensor *Tensors::get(TensorId tenId) {
  auto found = M.find(tenId);
  if (found == M.end()) {
    throw error("no tensor with id " + tenId);
  }
  return found->second.get();
}

// neuralnet streams and prints are "impolite" (will not add new line at end)

void Op::append(std::stringstream &ss)  const{
  appendIO(ss);
  ss << '\n';
  appendMore(ss);
}

void TensorIndexMap::append(std::stringstream &ss, std::string prefix) const {
  int index = 0;
  for (auto &index_tensor : tensor_map) {
    ss << prefix << '@' << index_tensor.first << ':' << index_tensor.second->id
       << " of type " << index_tensor.second->tensor_type();
    if (index_tensor.second->info.isSet()){
      ss << ' ';
      index_tensor.second->info.append(ss);
    }

    ++index;
    if (index != tensor_map.size()) {
      ss << '\n';
    }
  }
}

void Op::appendIO(std::stringstream &ss)  const{
  static std::string tab = "    ";
  ss << '\n' <<  "Op " << id << " of type " << op_type << '\n';
  ss << tab << "inputs" << '\n';
  input.append(ss, tab + tab);
  ss << '\n' << tab << "outputs" << '\n';
  output.append(ss, tab + tab);
}

VectorAndSet::VectorAndSet(std::vector<std::string> &&vals) : v_vals(vals) {
  for (auto &v : v_vals) {
    m_vals.insert(v);
  }
}

void PreRunKnowledge::addInfo(TensorId id, const TensorInfo &info) {
  infos[id] = info;
}

const TensorInfo &PreRunKnowledge::getInfo(TensorId id) { return infos[id]; }

bool PreRunKnowledge::hasInfo(TensorId id) {
  return infos.find(id) != infos.end();
}

Graph::Graph(onnx::ModelProto &&inMod,
             PreRunKnowledge &&perk,
             Recorder &&rec,
             // Schedule needed, if momentum the graph is different
             Schedule &&sched,
             // Weights tensors which are not to be updated
             std::vector<std::string> &&cTens)
    : preRunKnowledge(perk), recorder(rec), schedule(sched),
      // constIds(std::move(cTens)), 
      tensors(std::move(cTens), this), 
      onnxModel(inMod) {

  auto &onnxGraph = onnxModel.graph();
  auto &onnxNodes = onnxGraph.node();

  std::set<TensorId> onnxInitializers;
  for (const auto &initializer : onnxGraph.initializer()) {
    TensorId tenId = initializer.name();
    tensors.addInit(tenId, &initializer);
    onnxInitializers.emplace(tenId);
  }

  // onnx inputs which are not initializers are true inputs
  for (auto &valueInfo : onnxGraph.input()) {
    if (onnxInitializers.count(valueInfo.name()) == 0) {
      tensors.addStream(valueInfo.name());
    }
  }

  for (const auto &node : onnxNodes) {
    growFromNode(node);
  }

  // this checks that there are no contradictions in the user input, NOT
  // in the implementation of neuralnet.
  validate();

  splitConvBias();

  removePadSizeZero();

  inferTensorInfos();
}


std::vector<TensorId> Tensors::getNoProducerIds() const{
  // the tensors which are not generated by an Op
  std::vector<TensorId> t0 = getIds(TensorType::Stream);
  std::vector<TensorId> t1 = getInitIds();
  t0.insert(t0.end(), t1.begin(), t1.end());
  return t0;
}

void Graph::inferTensorInfos() { 
  for (const auto & tensorId: tensors.getInitIds()){
    auto pt = tensors.getOnnxInit(tensorId);
    tensors.get(tensorId)->info.set(*pt);
  }

  std::vector<TensorId> streamTensors = tensors.getIds(TensorType::Stream);
  for (const auto & id:  streamTensors){
    if (!(preRunKnowledge.hasInfo(id))){
      throw error("expected pre-run knowledge for stream tensor " + id);
    }
    tensors.get(id)->info = preRunKnowledge.getInfo(id);
  }


  // this is wrong: TODO topological sort
  for (Op* op : getTopologicallySorted()){
    op->setOutputInfos();
  }
}

const onnx::TensorProto *Tensors::getOnnxInit(TensorId id) const {
  return init.at(id);
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
  //Tensor *tensorIn = tensors[name].get();
  Tensor *tensorIn = tensors.get(name);
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
  tensors.remove(name);
  // tensors.erase(name);
  // delete (.)
  // removeOp(opId);
  ops.erase(opId);
}

int TensorIndexMap::n() const{
  return static_cast<int>(tensor_map.size());
}


// void Graph::removeOp(OpId id){
//   // ...
//   ops.erase(id);
// }



bool Graph::isLogged(TensorId tenId) {
  (void)tenId;
  return false;
}

void Graph::splitConvBias() {}

const std::vector<std::string> &VectorAndSet::v() const { return v_vals; }

void Graph::validate() {
  for (auto &tensorId : tensors.constIds.v()) {
    if (!tensors.contains(tensorId)) {
      throw error("no tensor " + tensorId +
                  " in graph, error in const tensor names");
    }
  }
}

bool VectorAndSet::contains(std::string name)  const{
  return m_vals.count(name) == 1;
}

void Tensors::addInit(TensorId name, const onnx::TensorProto *pt) {
  init[name]    = pt;
  M[name] = std::unique_ptr<Tensor>(new Tensor(
      name,
      constIds.contains(name) ? TensorType::Const : TensorType::Variable,
      pgraph));
}

void Tensors::addActivation(TensorId tenId) {
  M[tenId] =
      std::unique_ptr<Tensor>(new Tensor(tenId, TensorType::Activation, pgraph));
}

void Tensors::addStream(TensorId tenId) {
  M[tenId] =
      std::unique_ptr<Tensor>(new Tensor(tenId, TensorType::Stream, pgraph));
}

const std::vector<std::string> &Attributes::getNames() const { return names; }

onnxAttPtr Attributes::at(std::string name) const { return att_map.at(name); }

Attributes::Attributes(decltype(onnx::NodeProto().attribute()) &attributes) {
  for (auto &attribute : attributes) {
    auto name = attribute.name();
    names.push_back(name);
    att_map[name] = &attribute;
  }
}

void Attributes::append(std::stringstream &ss)  const {
  using AttPro = onnx::AttributeProto;
  for (auto &name : names) {
    ss << '\n';
    ss << "  " << name << "  ";
    auto attptr = att_map.at(name);
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

void Tensors::remove(TensorId id){
  M.erase(id);
}

bool Tensors::contains(TensorId id) const{
  return M.find(id) != M.end();
}

void Graph::growFromNode(const onnx::NodeProto &node) {

  auto atts = Attributes(node.attribute());
  if (opTypes.get(node.op_type()) == OpType::CONSTANT) {
    TensorId name = node.output(0);
    tensors.addInit(name, &node.attribute(0).t());
  }

  else {
    OpId opId = nOpsGenerated;
    ops[opId] = addOp(nOpsGenerated++, node);

    for (int inIndex = 0; inIndex < node.input_size(); ++inIndex) {
      auto &inName = node.input(inIndex);
      if (inName == "") {
        // no input at this position
      } else {
        //auto found = tensors.find(inName);
        if(!tensors.contains(inName)){  //if (found == tensors.end()) {
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
               {"Constant", OpType::CONSTANT},
               {"Conv", OpType::CONV},
               {"LogSoftmax", OpType::LOGSOFTMAX},
               {"Pad", OpType::PAD},
               {"Relu", OpType::RELU}};
}

OpType OpTypes::get(std::string op_type) {
  auto found = opTypeMap.find(op_type);
  if (found == opTypeMap.end()) {
    throw error("No OpType found for " + op_type);
  }
  return found->second;
}

bool Op::fromNode() const { return ptrNode != nullptr; }

const Node *Op::getNode() const{
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
  case OpType::LOGSOFTMAX: {
    return pOp(new LogSoftmaxOp(opId, node, this));
  }
  case OpType::PAD: {
    return pOp(new PadOp(opId, node, this));
  }
  case OpType::RELU: {
    return pOp(new ReluOp(opId, node, this));
  }
  default: { throw error("No class for " + node.op_type()); }
  }
}
class Graph;

} // namespace neuralnet
