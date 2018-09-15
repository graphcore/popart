// TIMELINE:
// 1) support basic conv nets.

#include <map>
#include <neuralnet/error.hpp>
#include <neuralnet/graph.hpp>
#include <neuralnet/tensor.hpp>
#include <neuralnet/tensorinfo.hpp>
#include <neuralnet/util.hpp>
#include <sstream>
#include <vector>

// The layers:
#include <neuralnet/averagepool.hpp>
#include <neuralnet/conv.hpp>
#include <neuralnet/logsoftmax.hpp>
#include <neuralnet/nll.hpp>
#include <neuralnet/pad.hpp>
#include <neuralnet/relu.hpp>

namespace neuralnet {


std::string getNeuralNetDomain() {
  return "gnilwen.semaj";
}

OpTypes initOpTypes() { return OpTypes(); }

const OpTypes &getOpTypes() {
  const static OpTypes X = initOpTypes();
  return X;
}

void Op::setup() { throw error("No setup() for " + op_type()); }

OpId Loss::getOpId() const {
  if (opId < 0) {
    throw error("opId of Loss not yet set");
  }
  return opId;
}

void Loss::set(OpId opId_, Graph * pgraph_){
  opId = opId_;
  pgraph = pgraph_;
  setInOut(input_, output_);
}

// const std::vector<TensorId> &Loss::getInput() const { return input_; }
// const std::vector<TensorId> &Loss::getOutput() const { return output_; }

int Loss::input_size() const { return static_cast<int>(input_.size()); }
const TensorId &Loss::input(int i) const { return input_.at(i); }
int Loss::output_size() const { return static_cast<int>(output_.size()); }
const TensorId &Loss::output(int i) const { return output_.at(i); }

Graph *Loss::getGraph() const {
  if (pgraph == nullptr) {
    throw error("pgraph of Loss not yet set");
  }
  return pgraph;
}

// Essentially Kahn's alogorithm (1962), see
// https://en.wikipedia.org/wiki/Topological_sorting
// Will people still be implementing this algorithm 
// in 1962 + 134 = 2096? :)
std::vector<Op *> Graph::getTopologicallySorted() {

  std::vector<Op *> sorted;
  std::vector<Op *> opsToProcess;
  // map from each op to the number of input
  // indices it is waiting on initialised as
  // total number of input indices
  std::map<Op *, int> awaiting;
  for (auto &id_op : ops) {
    awaiting[id_op.second.get()] = id_op.second->input.n();
  }

  // processing a tensor involves
  // reducing the counts in awaiting for
  // ops which use it, and detecting which
  // ops have nothing left to wait for as a
  // result of such updating.
  auto processTensor = [&opsToProcess, &awaiting](Tensor *tensor) {
    for (auto &op_count : tensor->consumers.getMap()) {
      awaiting[op_count.first] -= op_count.second;
      if (awaiting[op_count.first] == 0) {
        opsToProcess.push_back(op_count.first);
      }
    }
  };

  // we will start by processing
  // the tensors which have no producers
  auto t0 = tensors.getNoProducerIds();
  for (auto &id : t0) {
    processTensor(tensors.get(id));
  }

  while (opsToProcess.size() != 0) {
    auto op = opsToProcess.back();
    opsToProcess.resize(opsToProcess.size() - 1);
    sorted.push_back(op);
    for (auto &tensor_indices : op->output.indicesMap()) {
      processTensor(tensor_indices.first);
    }
  }

  if (sorted.size() != ops.size()) {
    throw error("failure to sort topologically");
  }
  return sorted;
}

std::vector<TensorId> Tensors::getInitIds() const {
  std::vector<TensorId> initIds;
  for (auto &id_pt : M) {
    if (id_pt.second->tensorType() == TensorType::Const ||
        id_pt.second->tensorType() == TensorType::Variable) {
      initIds.push_back(id_pt.first);
    }
  }
  return initIds;
}

std::vector<TensorId> Tensors::getIds(TensorType type) const {
  std::vector<TensorId> ids;
  for (auto &id_pt : M) {
    if (id_pt.second->tensorType() == type) {
      ids.push_back(id_pt.first);
    }
  }
  return ids;
}

Tensors::Tensors(std::vector<std::string> &&vals1, Graph *pg)
    : constIds(std::move(vals1)), pgraph(pg) {}

Op::~Op()                     = default;
VectorAndSet::~VectorAndSet() = default;
Tensors::~Tensors()           = default;
Graph::~Graph()               = default;

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

const Tensor *TensorIndexMap::tensor(int index) const {
  return tensor_map.at(index);
}

const std::vector<int> &TensorIndexMap::indices(Tensor *ptensor) const {
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
  ptensor->setProducer(this);
}

Tensor *Tensors::get(TensorId tenId) {
  auto found = M.find(tenId);
  if (found == M.end()) {
    throw error("no tensor with id " + tenId);
  }
  return found->second.get();
}

// neuralnet streams and prints are "impolite" (will not add new line at end)

void Op::append(std::stringstream &ss) const {
  appendIO(ss);
  ss << '\n';
  appendMore(ss);
}

void TensorIndexMap::append(std::stringstream &ss, std::string prefix) const {
  int index = 0;
  for (auto &index_tensor : tensor_map) {
    ss << prefix << '@' << index_tensor.first << ':'
       << padded(index_tensor.second->id, 4)

       << ' ' << padded(index_tensor.second->tensor_type(), 11);
    if (index_tensor.second->info.isSet()) {
      ss << ' ';
      index_tensor.second->info.append(ss);
    }

    ++index;
    if (index != tensor_map.size()) {
      ss << '\n';
    }
  }
}

void Op::appendIO(std::stringstream &ss) const {
  static std::string tab = "    ";
  ss << '\n' << "Op " << id << " of type " << op_type() << '\n';
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

const TensorInfo &PreRunKnowledge::getInfo(TensorId id) const {
  return infos.at(id);
}

bool PreRunKnowledge::hasInfo(TensorId id) const {
  return infos.find(id) != infos.end();
}

void Graph::confirmNoGradIds() const{

  auto &onnxGraph = onnxModel.graph();

  for (const auto &in_ : onnxGraph.input()) {
    confirmNonGradId(in_.name());
  }

  for (const auto &out_ : onnxGraph.output()) {
    confirmNonGradId(out_.name());
  }

  for (const auto & tenId : preRunKnowledge.getAllTensorIds()){
    confirmNonGradId(tenId);
  }

}

std::vector<TensorId> PreRunKnowledge::getAllTensorIds() const{
  // we first put the TensorIds into a set, so that duplication is removed
  std::set<TensorId> all_;
  for (const auto & id_info : infos){
    all_.insert(id_info.first);
  }

  // then any other TensorIds from other maps in PreRunKnowledge will be added
  // to the set here.
  
  std::vector<TensorId> all;
  for (auto & x : all_){
    all.push_back(x);
  }
  return all;
}

Graph::Graph(onnx::ModelProto &&inMod,
             PreRunKnowledge &&perk,
             Recorder &&rec,
             std::unique_ptr<Loss> &&ls,
             std::vector<std::unique_ptr<Regularizer>> &&regs,
             // Schedule needed, if momentum the graph is different
             Schedule &&sched,
             // Weights tensors which are not to be updated
             std::vector<std::string> &&cTens)
    : preRunKnowledge(perk), recorder(rec), loss(std::move(ls)),
      regularizers(std::move(regs)), schedule(sched),
      // constIds(std::move(cTens)),
      tensors(std::move(cTens), this), onnxModel(inMod) {


  confirmNoGradIds();

  auto &onnxGraph = onnxModel.graph();

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

  // other true inputs are for the loss calculation (class labels, etc)
  for (const auto & lossStreamTensorName : loss->getStreamTensorNames()){
    tensors.addStream(lossStreamTensorName);
  }

  constructForwards();

  // to developers: confirm fuctions like this
  // should be to check that there
  // are no contradictions in the user input, NOT
  // in the implementation of the library neuralnet.
  confirmConstIds();
  
  splitConvBias();
  
  removePadSizeZero();
  
  constructBackwards();
  
  inferTensorInfos();
}



std::vector<TensorId> Tensors::getNoProducerIds() const {
  // the tensors which are not generated by an Op
  std::vector<TensorId> t0 = getIds(TensorType::Stream);
  std::vector<TensorId> t1 = getInitIds();
  t0.insert(t0.end(), t1.begin(), t1.end());
  return t0;
}

void Graph::inferTensorInfos() {
  for (const auto &tensorId : tensors.getInitIds()) {
    auto pt = tensors.getOnnxInit(tensorId);
    tensors.get(tensorId)->info.set(*pt);
  }

  std::vector<TensorId> streamTensors = tensors.getIds(TensorType::Stream);
  for (const auto &id : streamTensors) {
    if (!(preRunKnowledge.hasInfo(id))) {
      throw error("expected pre-run knowledge for stream tensor " + id);
    }
    tensors.get(id)->info = preRunKnowledge.getInfo(id);
  }

  for (Op *op : getTopologicallySorted()) {
    op->setup();
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
  // Tensor *tensorIn = tensors[name].get();
  Tensor *tensorIn = tensors.get(name);
  // ()
  Op *op = ops[opId].get();
  // [.]
  Tensor *tensorOut = op->output.tensor(0);
  // (.)
  auto op0 = tensorIn->getProducer();
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

void Tensor::setProducer(Op * op){
  if (hasProducer()){
    throw error("Cannot set a producer for Tensor " + id  + " as already one");
  }
  producer = op;
}

int TensorIndexMap::n() const { return static_cast<int>(tensor_map.size()); }

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

void Graph::confirmConstIds()  const{
  for (auto &tensorId : tensors.constIds.v()) {
    if (!tensors.contains(tensorId)) {
      throw error("no tensor " + tensorId +
                  " in graph, error in const tensor names");
    }
  }
}

void Graph::constructForwards(){
  auto &onnxGraph = onnxModel.graph();
  auto &onnxNodes = onnxGraph.node();
  for (const auto &node : onnxNodes) {
    growFromNode(node);
  }
}


bool VectorAndSet::contains(std::string name) const {
  return m_vals.count(name) == 1;
}

void Tensors::addInit(TensorId name, const onnx::TensorProto *pt) {
  init[name] = pt;
  M[name]    = std::unique_ptr<Tensor>(new Tensor(
      name,
      constIds.contains(name) ? TensorType::Const : TensorType::Variable,
      pgraph));
}

std::string reservedPrefix(){
  return "d|=|_";
}

void Tensors::addActivation(TensorId tenId) {
  M[tenId] = std::unique_ptr<Tensor>(
      new Tensor(tenId, TensorType::Activation, pgraph));
}


void Graph::confirmNonGradId(TensorId tenId)  const{
  if (tenId.find(reservedPrefix()) != std::string::npos) {
    throw error("Provided tensor " + tenId +
                " has an invalid name: clash with reserved prefix " +
                reservedPrefix());
  }
}


void Tensors::addStream(TensorId tenId) {
  M[tenId] =
      std::unique_ptr<Tensor>(new Tensor(tenId, TensorType::Stream, pgraph));
}

const std::vector<std::string> &Attributes::getNames() const { return names; }

onnxAttPtr Attributes::at(std::string name) const { return att_map.at(name); }

template <> void Attributes::setIfPresent(int64_t &v, std::string s) const {
  auto found = att_map.find(s);
  if (found != att_map.end()) {
    v = found->second->i();
  }
}

template <> void Attributes::setIfPresent(std::string &v, std::string s) const {
  auto found = att_map.find(s);
  if (found != att_map.end()) {
    v = found->second->s();
  }
}

template <>
void Attributes::setIfPresent(std::vector<int64_t> &vs, std::string s) const {
  auto found = att_map.find(s);
  if (found != att_map.end()) {
    vs.resize(0);
    vs.reserve(found->second->ints_size());
    for (auto &v : found->second->ints()) {
      vs.push_back(v);
    }
  }
}

Attributes::Attributes(decltype(Node().attribute()) &attributes) {
  for (auto &attribute : attributes) {
    auto name = attribute.name();
    names.push_back(name);
    att_map[name] = &attribute;
  }
}

void Attributes::append(std::stringstream &ss) const {
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

TensorId getGradId(TensorId id){
  return  reservedPrefix() + id;
}

TensorId getGradId(TensorId tenId, OpId opId, int index){
  std::stringstream ss;
  ss << reservedPrefix() << opId << '_' << index << '_' << tenId;
  return ss.str();
}

void Tensors::remove(TensorId id) { M.erase(id); }

bool Tensors::contains(TensorId id) const { return M.find(id) != M.end(); }


TensorId getUniqueOutId(const onnx::ModelProto &m) {
  auto nOuts = m.graph().output_size();
  if (nOuts != 1) {
    throw error("cannot create NegLogLikeLoss from onnx Graph with " +
                std::to_string(nOuts) + " outputs");
  }
  return m.graph().output(0).name();
}

OpId Graph::getAndIncrOpsCounter() {
  OpId nOps0 = opsCounter;
  ++opsCounter;
  return nOps0;
}


Node Op::getGradientPartner() const {
  throw error("no getGradientPartner for " + op_type());
}

Op *Graph::growFromLoss() {
  OpId opId = getAndIncrOpsCounter();
  loss->set(opId, this);
  ops[opId] = loss->getOp();
  connectInputs(*loss, opId);
  connectOutputs(*loss, opId);
  return ops[opId].get();
}

void Graph::constructBackwards() {
  growFromLoss();
  //throw error("in construct backwards");
}

Op *Graph::growFromNode(const Node &node) {

  // special case of CONSTANT Node, no Op is created
  if (getOpTypes().get(node.op_type()) == OpType::CONSTANT) {
    TensorId name = node.output(0);
    tensors.addInit(name, &node.attribute(0).t());
    // no Op created for a Constant Node
    return nullptr;
  }

  OpId opId = getAndIncrOpsCounter();
  ops[opId] = addOp(opId, node);
  connectInputs(node, opId);
  connectOutputs(node, opId);
  return ops[opId].get();
}

template <typename T>
void Graph::connectInputs(const T & inContainer, OpId opId){
  for (int inIndex = 0; inIndex < inContainer.input_size(); ++inIndex) {
    auto &inName = inContainer.input(inIndex);
    if (inName == "") {
      // no input at this position
    } else {
      if (!tensors.contains(inName)) {
        throw error("input " + inName + " should already be in tensor map");
      } else {
        ops[opId]->connectInTensor(inIndex, inName);
      }
    }
  }
}

template <typename T>
void Graph::connectOutputs(const T & outContainer, OpId opId){
  for (int outIndex = 0; outIndex < outContainer.output_size(); ++outIndex) {
    auto &outName = outContainer.output(outIndex);
    if (outName == "") {
      // no output at this position
    } else {
      // ONNX specifies that a tensor is the output of at most 1 node.
      // here we create the Activation / Gradient Tensor and connect it
      // to the Op.
      ops[opId]->createAndConnectOutTensor(outIndex, outName);
    }
  }
}

OpTypes::OpTypes() {

  opTypes_ = {{"AveragePool", OpType::AVERAGEPOOL},
              {"Constant", OpType::CONSTANT},
              {"Conv", OpType::CONV},
              {"LogSoftmax", OpType::LOGSOFTMAX},
              {"NegLogLike", OpType::NEGLOGLIKE},
              {"Pad", OpType::PAD},
              {"Relu", OpType::RELU}};

  for (auto &x : opTypes_) {
    strings_[x.second] = x.first;
  }
}

const OpType &OpTypes::get(std::string op_type) const {
  auto found = opTypes_.find(op_type);
  if (found == opTypes_.end()) {
    throw error("No OpType found for " + op_type);
  }
  return found->second;
}

const std::string &OpTypes::get(OpType opType) const {
  return strings_.at(opType);
}


void Graph::append(std::stringstream &ss) {
  ss << "-- Graph --\n";
  for (auto &id_op : ops) {
    id_op.second->append(ss);
  }
}

const std::string & Op::domain(){
  return op_domain;
}

const std::string &Op::op_type() const { return *p_op_type; }

OpConstructorBundle::OpConstructorBundle(OpId opId_,
                                         std::string op_type_,
                                         Graph *pgraph_,
                                         Attributes atts_,
                                         std::string domain_)
    : id(opId_), op_type(op_type_), pgraph(pgraph_), atts(atts_),
      domain(domain_) {}

Op::Op(const OpConstructorBundle &b)
  : id(b.id),
      // opType (from string op_type)
      opType(getOpTypes().get(b.op_type)),
      // the Graph
      pgraph(b.pgraph),
      // the Attributes
      nAtts(b.atts),
      // opType
      p_op_type(&getOpTypes().get(opType)),
      // domain
      op_domain(b.domain) {}

Op::Op(OpId id_, const Node &node, Graph *pg)
    : id(id_),
      // We set opType, looked up in a map from the string node.op_type()
      opType(getOpTypes().get(node.op_type())),
      // pointer to the graph containing this node
      pgraph(pg),
      // neuralnet::Attributes constructed from contained of onnx::Attribute s
      nAtts(node.attribute()),
      // We set the pointer to the string version of opType, in another map
      p_op_type(&getOpTypes().get(opType)),
      // And finally we strip off the domain of the Node
      op_domain(node.domain()) {}


std::unique_ptr<Op> Graph::addOp(OpId opId, const Node &node) {
  using pOp = std::unique_ptr<Op>;
  switch (getOpTypes().get(node.op_type())) {
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
  case OpType::NEGLOGLIKE: {
    throw error("cannot construct NegLogLike from a Node");
    // return pOp(new NegLogLikeOp(opId, node, this));
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

} // namespace neuralnet
