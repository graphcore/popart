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
#include <neuralnet/squeeze.hpp>
#include <neuralnet/sum.hpp>
#include <neuralnet/varupdate.hpp>

namespace neuralnet {

// Passes calls to .size() and .at(int) to input_size() and input(int)
// so that standard containers can be used in Graph::connectInputs (as T)
template <typename T> class InputWrapper {
public:
  InputWrapper(const T &inputs_) : inputs(inputs_) {}
  int input_size() const { return static_cast<int>(inputs.size()); }
  const TensorId &input(int inIndex) const { return inputs.at(inIndex); }

private:
  const T &inputs;
};

// Passes calls to .size() and .at(int) to output_size() and output(int)
// so that standard containers can be used in Graph::connectOutputs.
template <typename T> class OutputWrapper {
public:
  OutputWrapper(const T &outputs_) : outputs(outputs_) {}
  int output_size() const { return static_cast<int>(outputs.size()); }
  const TensorId &output(int inIndex) const { return outputs.at(inIndex); }

private:
  const T &outputs;
};

std::vector<TensorId> TensorIndexMap::getSerialised() const {
  int maxIndex = 0;
  for (auto &ind_tensor : tensor_map) {
    if (ind_tensor.first > maxIndex) {
      maxIndex = ind_tensor.first;
    }
  }
  std::vector<TensorId> serialised(maxIndex, "");
  for (auto &ind_tensor : tensor_map) {
    serialised[ind_tensor.first] = ind_tensor.second->id;
  }
  return serialised;
}

// return a vector of 1 or several OpAndTensorIds for
// obtaining the gradient of the inputs of this Op.
// The Op in the OpAndTensorIds is the gradient op, and
// the TensorIds are the input indices of input of this
// Op for which the gradient is computed
std::vector<std::unique_ptr<Op>> Op::getGradOps() {
  throw error("Cannot get gradients for " + op_type());
}

bool TensorIndexMap::hasIndex(int index) const {
  return tensor_map.find(index) != tensor_map.end();
}

void TensorIndexMap::setInfoIfIndex(const TensorInfo &info_, int index) {
  if (hasIndex(index)) {
    tensor(index)->info = info_;
  }
}

std::string getNeuralNetDomain() { return "gnilwen.semaj"; }

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

std::unique_ptr<Op> Loss::finalSetAndGetOp(Graph *pgraph_) {
  pgraph = pgraph_;
  opId   = pgraph->getOpsCounter();
  setInOut(input_, output_);
  return getSpecificOp();
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

// Essentially Kahn's alogorithm (1962, 56 years ago!),
// see https://en.wikipedia.org/wiki/Topological_sorting
// but not quite Kahn's algorithm as it there are some
// additional constraints on the order of Ops imposed
// externally
std::vector<Op *> Graph::getTopologicallySorted() const {

  // the topological sorting (to construct in this function)
  std::vector<Op *> sorted;
  // ops which have all their input tensors
  // created (but might be waiting for other ops
  // to be inserted)
  std::vector<Op *> opsToProcess;
  // map from each op to the number of tensor input
  // indices it is waiting on
  std::map<Op *, int> nIndicesAwaiting;
  // initialise nIndicesAwatings as total
  // number of input indices
  for (auto &id_op : ops) {
    Op *op               = id_op.second.get();
    nIndicesAwaiting[op] = op->input.n();
  }

  // the next two variables are needed because of the
  // external constraints.
  // (1) map for each op to the number of ops which still
  // must be inserted before it can it can be inserted
  std::map<Op *, int> nOpsAwaiting;
  // (2) map from each op to a list of ops which are
  // waiting for it
  std::map<Op *, std::vector<Op *>> isWaitingFor;
  // initialise (1) and (2)
  for (auto &id_op : ops) {
    Op *op           = id_op.second.get();
    nOpsAwaiting[op] = 0;
    isWaitingFor[op] = {};
  }
  for (auto &id_op : ops) {
    Op *op = id_op.second.get();
    for (auto &tensor_indices : op->input.indicesMap()) {
      Tensor *inTen = tensor_indices.first;
      // which consumer(s) of inTens must appear before op?
      for (Op *otherCon : inTen->consumers.consumersWhichTopoBefore(op)) {
        if (std::find(isWaitingFor[otherCon].begin(),
                      isWaitingFor[otherCon].end(),
                      op) == isWaitingFor[otherCon].end()) {
          isWaitingFor[otherCon].push_back(op);
          ++nOpsAwaiting[op];
        }
      }
    }
  }

  auto readyToProcess = [&nIndicesAwaiting, &nOpsAwaiting](Op *op) {
    return (nIndicesAwaiting[op] == 0 && nOpsAwaiting[op] == 0);
  };

  // processing a tensor involves
  // reducing the counts in `awaiting' for
  // ops which use it, and detecting which
  // ops have nothing left to wait for as a
  // result of such updating.
  auto processTensor =
      [&opsToProcess, &nIndicesAwaiting, &readyToProcess](Tensor *tensor) {
        for (auto &op_count : tensor->consumers.getMap()) {
          Op *op = op_count.first;
          nIndicesAwaiting[op] -= op_count.second;
          if (readyToProcess(op)) {
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
    for (Op *waitingOp : isWaitingFor[op]) {
      --nOpsAwaiting[waitingOp];
      if (readyToProcess(waitingOp)) {
        opsToProcess.push_back(waitingOp);
      }
    }

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
  pgraph->tensors.addActGrad(tenId);
  Tensor *ptensor = pgraph->tensors.get(tenId);
  output.insert(outIndex, ptensor);
  ptensor->setProducer(this);
}

Tensor *Tensors::get(TensorId tenId) const {
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

void TensorIndexMap::append(std::stringstream &ss,
                            std::string prefix,
                            int max_id_length) const {
  int index = 0;

  for (auto &index_tensor : tensor_map) {
    ss << prefix << '@' << index_tensor.first << ':' << ' '
       << padded(index_tensor.second->id, max_id_length + 1)

       << ' ' << padded(index_tensor.second->tensor_type(), 9);
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

int TensorIndexMap::maxIdLength() const {
  int max_id_length = 0;
  for (const auto &tensor_indices : indicesMap()) {
    max_id_length = std::max(max_id_length,
                             static_cast<int>(tensor_indices.first->id.size()));
  }
  return max_id_length;
}

void Op::appendIO(std::stringstream &ss) const {
  static std::string tab = "    ";
  ss << '\n' << "Op " << id << " of type " << op_type() << '\n';
  ss << tab << "inputs" << '\n';

  int max_id_length = std::max(input.maxIdLength(), output.maxIdLength());
  input.append(ss, tab + tab, max_id_length);
  ss << '\n' << tab << "outputs" << '\n';
  output.append(ss, tab + tab, max_id_length);
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

void Graph::confirmNoReservedIds() const {

  auto &onnxGraph = onnxModel.graph();

  for (const auto &in_ : onnxGraph.input()) {
    confirmNonReservedId(in_.name());
  }

  for (const auto &out_ : onnxGraph.output()) {
    confirmNonReservedId(out_.name());
  }

  for (const auto &tenId : preRunKnowledge.getAllTensorIds()) {
    confirmNonReservedId(tenId);
  }
}

std::vector<TensorId> PreRunKnowledge::getAllTensorIds() const {
  // we first put the TensorIds into a set, so that duplication is removed
  std::set<TensorId> all_;
  for (const auto &id_info : infos) {
    all_.insert(id_info.first);
  }

  // then any other TensorIds from other maps in PreRunKnowledge will be added
  // to the set here.

  std::vector<TensorId> all;
  for (auto &x : all_) {
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

  confirmNoReservedIds();

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
  for (const auto &lossStreamTensorName : loss->getStreamTensorNames()) {
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
  // [] (see .hpp for ascii picture definitions)
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
  tensorOut->resetProducer(op0);
  // delete []
  tensors.remove(name);
  // tensors.erase(name);
  // delete (.)
  // removeOp(opId);
  ops.erase(opId);
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

void Graph::confirmConstIds() const {
  for (auto &tensorId : tensors.constIds.v()) {
    if (!tensors.contains(tensorId)) {
      throw error("no tensor " + tensorId +
                  " in graph, error in const tensor names");
    }
  }
}

void Graph::constructForwards() {
  auto &onnxGraph = onnxModel.graph();
  auto &onnxNodes = onnxGraph.node();
  for (const auto &node : onnxNodes) {
    growFromNode(node);
  }
}

bool VectorAndSet::contains(std::string name) const {
  return m_vals.count(name) == 1;
}

void Tensors::insert(TensorId name, std::unique_ptr<Tensor> t) {
  if (M.find(name) != M.end()) {
    throw error("ILE : tensor " + name + " already in M");
  }
  M[name] = std::move(t);
}

void Tensors::addInit(TensorId name, const onnx::TensorProto *pt) {
  init[name] = pt;
  insert(name,
         std::unique_ptr<Tensor>(new Tensor(
             name,
             constIds.contains(name) ? TensorType::Const : TensorType::Variable,
             pgraph)));
}

std::string reservedPrefix() { return "d__"; }

void Tensors::addActGrad(TensorId tenId) {
  insert(
      tenId,
      std::unique_ptr<Tensor>(new Tensor(tenId, TensorType::ActGrad, pgraph)));
}

void Graph::confirmNonReservedId(TensorId tenId) const {
  if (tenId.find(reservedPrefix()) != std::string::npos) {
    throw error("Provided tensor " + tenId +
                " has an invalid name: clash with reserved prefix " +
                reservedPrefix());
  }

  if (tenId == getLearningRateId()) {
    throw error("Provided tensor " + tenId + " has a reserved name");
  }
}

void Tensors::addStream(TensorId tenId) {
  insert(
      tenId,
      std::unique_ptr<Tensor>(new Tensor(tenId, TensorType::Stream, pgraph)));
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

TensorId getGradId(TensorId id) { return reservedPrefix() + id; }

TensorId getNonGradId(TensorId id) {
  return id.substr(reservedPrefix().size());
}

TensorId getEdgeGradId(TensorId tenId, OpId opId, int index) {
  // we don't need the name of the tensor which this is an edge-grad of,
  // the edge-gradient is uniquely defined by the the edge it flows on
  // in the forward pass (input at 'index' to 'opId')
  (void)tenId;
  std::stringstream ss;
  ss << reservedPrefix() << opId << '_' << index;
  return ss.str();
}

void Tensors::remove(TensorId id) {
  M.erase(id);
  auto found = non_gradients_.find(id);
  if (found != non_gradients_.end()) {
    non_gradients_.erase(id);
  }
}

Tensor *Tensors::getNonGradientOf(TensorId id) const {
  auto found = non_gradients_.find(id);
  if (found != non_gradients_.end()) {
    return found->second;
  }
  throw error("No non-gradient for " + id);
}

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

OpId Graph::getOpsCounter() const { return opsCounter; }

OpId Graph::moveIntoGraph(std::unique_ptr<Op> op) {
  OpId id = op->id;
  ops[id] = std::move(op);
  return id;
}

Op *Graph::growFromLoss() {
  OpId opId = moveIntoGraph(loss->finalSetAndGetOp(this));
  connectInputs(*loss, opId);
  connectOutputs(*loss, opId);
  Op *op = ops[opId].get();
  return op;
}

Op *Graph::growGradSumOp(Tensor *target, const std::vector<Tensor *> &toSum) {

  OpId opId = moveIntoGraph(
      std::unique_ptr<Op>(new SumOp({"Sum", this, {}, getNeuralNetDomain()})));

  std::vector<TensorId> inputs;
  inputs.reserve(toSum.size());
  for (auto &tensor : toSum) {
    inputs.push_back(tensor->id);
  }
  TensorId gradientId = getGradId(target->id);
  std::vector<TensorId> outputs{gradientId};

  connectInputs(InputWrapper<decltype(inputs)>(inputs), opId);
  connectOutputs(OutputWrapper<decltype(outputs)>(outputs), opId);
  tensors.addNonGradient(gradientId, target);
  return ops[opId].get();
}

std::vector<Op *> Graph::growGradOps(Op *nonGradOp) {

  OpId nonGradOpId = nonGradOp->id;
  auto backOps     = nonGradOp->getGradOps();
  std::vector<Op *> gradOps;
  for (auto &upop : backOps) {
    Op *gradOp    = upop.get();
    OpId gradOpId = moveIntoGraph(std::move(upop));

    // connect inputs of gradOp
    {
      // inputs to gradOp (to populate in this scope):
      std::map<int, std::string> m_inputs;
      int max_input_index = 0;
      for (auto &inOutMapper : gradOp->gradInputInfo()) {

        int indexGrad     = inOutMapper.iGrad;
        int indexFwd      = inOutMapper.iNonGrad;
        GradOpInType type = inOutMapper.type;

        max_input_index = std::max(indexGrad, max_input_index);

        // the input at index 'indexGrad' to gradOp is
        switch (type) {
        //  (1) the INPUT at index 'indexFwd' of nonGradOp
        case GradOpInType::IN: {
          m_inputs[indexGrad] = nonGradOp->input.tensor(indexFwd)->id;
          break;
        }

        //  (2) the OUTPUT at index 'indexFwd' of nonGradOp
        case GradOpInType::OUT: {
          m_inputs[indexGrad] = nonGradOp->output.tensor(indexFwd)->id;
          break;
        }

        //  (3) the GRADIENT of the OUTPUT
        //      at index 'indexFwd' of nonGradOp.
        case GradOpInType::GRADOUT: {
          if (!nonGradOp->output.hasIndex(indexFwd)) {
            std::stringstream ss;
            ss << "No gradient for non-grad-op " << nonGradOp->str()
               << " at index " << indexFwd << '.'
               << " Could it be that the path along that index "
               << "did not lead to loss, "
               << "in which case the gradient is zero?";
            throw error(ss.str());
          }
          m_inputs[indexGrad] =
              getGradId(nonGradOp->output.tensor(indexFwd)->id);
          break;
        }
        }
      }
      // convert m_imputs to a vector, a format supported by connectInputs
      std::vector<std::string> v_inputs(max_input_index + 1, "");
      for (auto &index_id : m_inputs) {
        v_inputs[index_id.first] = index_id.second;
      }

      connectInputs(InputWrapper<decltype(v_inputs)>(v_inputs), gradOpId);
      // modify topological constraints on consumers of inputs
      // gradOp->imposeTopoCons();
    }

    // connect outputs of gradOp
    {
      std::vector<TensorId> v_outputs;
      for (auto out_in : gradOp->gradOutToNonGradIn()) {
        int gradOut    = out_in.first;
        int nonGradIn  = out_in.second;
        TensorId inId  = nonGradOp->input.tensor(nonGradIn)->id;
        TensorId outId = getEdgeGradId(inId, nonGradOpId, nonGradIn);
        if (v_outputs.size() < gradOut + 1) {
          v_outputs.resize(gradOut + 1, "");
        }
        v_outputs[gradOut] = outId;
      }
      connectOutputs(OutputWrapper<decltype(v_outputs)>(v_outputs), gradOpId);
    }
    // note, as the outputs of gradOp are edge-grad-tensors and not
    // edge-grads, we do not need to match them to non-grad tensors.
    gradOps.push_back(gradOp);
  }

  return gradOps;
}

int Op::getNonGradInIndex(int) const {
  throw error("Op " + op_type() + " cannot `get non-grad in index'");
}

GradInOutMapper::GradInOutMapper(int iG, int iNG, GradOpInType t)
    : iGrad(iG), iNonGrad(iNG), type(t) {}

const std::vector<GradInOutMapper> &Op::gradInputInfo() const {
  throw error("Op " + op_type() + " cannot get `grad input info'");
}

const std::map<int, int> &Op::gradOutToNonGradIn() const {
  throw error("Op " + op_type() + " cannot get `grad out to non grad in'");
}

bool Op::readyToCreateGradients(std::set<int> &s) const {
  return s.size() == nPathsToLoss();
  // throw error("Op " + op_type() +
  //         " cannot determine if `ready to create gradients'");
}

void TensorGradRegistry::insert(Tensor *nonGrad, Tensor *grad) {
  auto found = partial.find(nonGrad);
  if (found == partial.end()) {
    partial[nonGrad] = {grad};
  } else {
    partial[nonGrad].push_back(grad);
  }
  if (partial[nonGrad].size() == nonGrad->nPathsToLoss()) {
    complete[nonGrad] = partial[nonGrad];
    partial.erase(nonGrad);
  }
}

void OpGradRegistry::insert(Op *nonGrad, int index) {
  auto found = partial.find(nonGrad);
  // so far NO gradients for nonGrad are in:
  if (found == partial.end()) {
    partial[nonGrad] = {};
  }
  // this should be removed when we're happy the IL (internal logic)
  // is correct:
  if (partial[nonGrad].count(index) != 0) {
    throw error("ILE : index already present in OpGradRegistry::insert");
  }
  partial[nonGrad].insert(index);
  // probably just checks that the size of partial is
  // nonGrad->output.n(), but maybe not.
  if (nonGrad->readyToCreateGradients(partial[nonGrad])) {
    complete.push_back(nonGrad);
    partial.erase(nonGrad);
  }
}

std::map<Tensor *, std::vector<Tensor *>> TensorGradRegistry::popComplete() {
  auto toRet = complete;
  complete   = {};
  return toRet;
}

std::vector<Op *> OpGradRegistry::popComplete() {
  auto toRet = complete;
  complete   = {};
  return toRet;
}

// communicate that op has computed gradients
void Graph::registerOpGrads(Op *op) {
  // For loss Op, nonGradOp == op, otherwise it is the unique
  // Op corresponding to op
  Op *nonGradOp = op->getNonGradOp();
  std::cout << "registering op " << op->id << " of type " << op->op_type()
            << std::endl;
  for (auto &index_tensor : op->gradOutMap()) {
    int opOutInd     = index_tensor.first;
    Tensor *partGrad = index_tensor.second;
    // what input index of nonGradOp does the
    // edge-gradient correspond to?
    int nonGradInInd      = op->getNonGradInIndex(opOutInd);
    Tensor *nonGradTensor = nonGradOp->input.tensor(nonGradInInd);
    std::cout << "     -> " << nonGradTensor->id << ", " << partGrad->id
              << std::endl;
    tensor_grad_registry.insert(nonGradTensor, partGrad);
  }
}

// communicate that a new gradient tensor
// (which is a sum along edges) is ready
void Graph::registerTensorGrad(Tensor *sum) {
  Tensor *nonGrad = tensors.getNonGradientOf(sum->id);
  if (nonGrad->hasProducer()) {
    Op *producer = nonGrad->getProducer();
    // the index at which nonGrad was produced
    int index = producer->output.indices(nonGrad).at(0);
    op_grad_registry.insert(producer, index);
  }
}

void Graph::setNPathsToLoss(Op *lossOp) {
  std::vector<Op *> opFront{lossOp};
  std::set<Op *> opsSeen{lossOp};
  while (opFront.size() != 0) {
    Op *op = opFront.back();
    opFront.resize(opFront.size() - 1);
    for (auto &ind_ten : op->input.tensorMap()) {
      auto tensor = ind_ten.second;
      tensor->incrNPathsToLoss();
      if (tensor->hasProducer()) {
        auto producer = tensor->getProducer();
        producer->incrNPathsToLoss();
        if (opsSeen.count(producer) == 0) {
          opFront.push_back(producer);
          opsSeen.insert(producer);
        }
      }
    }
  }
}

void Tensors::addNonGradient(TensorId id, Tensor *t) { non_gradients_[id] = t; }

void Graph::constructBackwards() {
  // definition: edge-gradient. What is output by a grad-op,
  // and which will be summed with other edge-gradients to create
  // a gradient. It is possible that an edge-gradient has the same
  // value as a gradient, if a tensor has only 1 consumer.

  // Add the Op which takes in activations and Streams, and
  // outputs edge-gradients (for all inputs) and the loss.
  Op *lossOp = growFromLoss();

  setNPathsToLoss(lossOp);

  // grad-ops which have created edge-gradients, but the
  // edge-gradients haven't signalled their existance
  std::vector<Op *> opsToRegister = {lossOp};

  while (!opsToRegister.empty()) {

    registerOpGrads(opsToRegister.back());
    opsToRegister.resize(opsToRegister.size() - 1);

    for (auto &nongrad_egrads : tensor_grad_registry.popComplete()) {
      Tensor *nongrad                     = nongrad_egrads.first;
      const std::vector<Tensor *> &egrads = nongrad_egrads.second;
      std::cout << "creating sum for tensor " << nongrad->id << std::endl;
      // nongrad required below, as the name of the output of the
      // created op (sumOp) will be based off of it. Also, we
      // register the link between sumOp's output and nongrad
      Op *sumOp = growGradSumOp(nongrad, egrads);

      switch (nongrad->tensorType()) {

      // if sumOp creates the gradient of an activation tensor,
      case TensorType::ActGrad: {
        registerTensorGrad(sumOp->output.tensor(0));
        break;
      }
      case TensorType::Variable: {
        // nothing to do, variable updates
        // follows at the end of this function
        break;
      }
      case TensorType::Const:
      case TensorType::Momentum:
      case TensorType::Stream: {
        // if the user wants the gradient of the
        // input data (unusual case) maybe we won't
        // break here. Example case : generating adversarials
        break;
      }
      case TensorType::Unknown:
      case TensorType::N:
        throw error("can't register gradient of " + nongrad->tensor_type() +
                    " tensor (yet?)");

      default: { throw error("only handling ActGrad and Variable for now"); }
      }
    }

    for (Op *op : op_grad_registry.popComplete()) {
      for (auto &gradOp : growGradOps(op)) {
        opsToRegister.push_back(gradOp);
      }
    }
  }

  std::cout << "while complete." << std::endl;

  //  // add weight ops (ignoring momentum's for now)
  //  for (auto & varId : tensors.getIds(TensorType::Variable)){
  //    Op * op = growVarUpdateOp(varId);
  //  }
}

TensorId Graph::getLearningRateId() const { return "learnRate"; }

Op *Graph::growVarUpdateOp(TensorId varId) {

  OpId opId = moveIntoGraph(std::unique_ptr<Op>(new VarUpdateOp(varId, this)));
  Op *op    = ops[opId].get();

  std::vector<TensorId> inputs{varId, getGradId(varId), getLearningRateId()};
  connectInputs(InputWrapper<decltype(inputs)>(inputs), opId);
  std::vector<TensorId> outputs{};

  connectOutputs(OutputWrapper<decltype(outputs)>(outputs), opId);
  trainTargetOps.push_back(op);

  throw error("impl impose cons for var update op");
  return op;
}

const std::map<int, Tensor *> &LossOp::gradOutMap() {
  if (!gradOutMapIsSet) {
    gradOutMap_ = output.tensorMap();
    // the last index is the loss, NOT a gradient thus remove it
    int largest_out_index{0};
    for (auto &x : gradOutMap_) {
      largest_out_index = std::max(largest_out_index, x.first);
    }
    gradOutMap_.erase(largest_out_index);
    gradOutMapIsSet = true;
  }
  return gradOutMap_;
}

LossOp::LossOp(const OpConstructorBundle &b) : Op(b) {}

const std::map<int, Tensor *> &TensorIndexMap::tensorMap() const {
  return tensor_map;
}

Op *Op::getNonGradOp() {
  throw error("No `get non grad op' for " + op_type() + " (yet?)");
}

const std::map<int, Tensor *> &Op::gradOutMap() {
  throw error("No `grad out map' for " + op_type());
}

Op *Graph::growFromNode(const Node &node) {

  // special case of CONSTANT Node, no Op is created
  if (getOpTypes().get(node.op_type()) == OpType::CONSTANT) {
    TensorId name = node.output(0);
    tensors.addInit(name, &node.attribute(0).t());
    // no Op created for a Constant Node
    return nullptr;
  }

  OpId opId = moveIntoGraph(addOp(node));

  connectInputs(node, opId);
  connectOutputs(node, opId);
  return ops[opId].get();
}

template <typename T>
void Graph::connectInputs(const T &inContainer, OpId opId) {
  for (int inIndex = 0; inIndex < inContainer.input_size(); ++inIndex) {
    auto &inName = inContainer.input(inIndex);
    if (inName == "") {
      // no input at this position
    } else {
      if (!tensors.contains(inName)) {
        throw error("input " + inName + " should already be in tensor map");
      } else {
        auto op = ops[opId].get();
        op->connectInTensor(inIndex, inName);
      }
    }
  }
}

template <typename T>
void Graph::connectOutputs(const T &outContainer, OpId opId) {
  for (int outIndex = 0; outIndex < outContainer.output_size(); ++outIndex) {
    auto &outName = outContainer.output(outIndex);
    if (outName == "") {
      // no output at this position
    } else {
      // ONNX specifies that a tensor is the output of at most 1 node.
      // here we create the Output (activation or gradient) Tensor and
      // connect it to the Op.
      ops[opId]->createAndConnectOutTensor(outIndex, outName);
    }
  }
}

OpTypes::OpTypes() {

  opTypes_ = {{"AveragePool", OpType::AVERAGEPOOL},
              {"AveragePoolGrad", OpType::AVERAGEPOOLGRAD},
              {"Constant", OpType::CONSTANT},
              {"Conv", OpType::CONV},
              {"ConvDataGrad", OpType::CONVDATAGRAD},
              {"ConvWeightsGrad", OpType::CONVWEIGHTSGRAD},
              {"LogSoftmax", OpType::LOGSOFTMAX},
              {"NegLogLike", OpType::NEGLOGLIKE},
              {"Pad", OpType::PAD},
              {"Relu", OpType::RELU},
              {"ReluGrad", OpType::RELUGRAD},
              {"Sum", OpType::SUM},
              {"Squeeze", OpType::SQUEEZE},
              {"SqueezeGrad", OpType::SQUEEZEGRAD},
              {"VarUpdate", OpType::VARUPDATE}};

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
  //  for (auto &id_op : ops) {
  //    id_op.second->append(ss);
  //  }

  for (auto &op : getTopologicallySorted()) {
    op->append(ss);
  }
}

const std::string &Op::domain() { return op_domain; }

const std::string &Op::op_type() const { return *p_op_type; }

OpConstructorBundle::OpConstructorBundle(std::string op_type_,
                                         Graph *pgraph_,
                                         Attributes atts_,
                                         std::string domain_)
    : op_type(op_type_), pgraph(pgraph_), atts(atts_), domain(domain_) {}

Op::Op(const OpConstructorBundle &b)
    : // opType (from string op_type)
      opType(getOpTypes().get(b.op_type)),
      // the Graph
      pgraph(b.pgraph),
      // the id
      id(pgraph->getAndIncrOpsCounter()),
      // the Attributes
      nAtts(b.atts),
      // opType
      p_op_type(&getOpTypes().get(opType)),
      // domain
      op_domain(b.domain) {}

Op::Op(const Node &node, Graph *pg)
    : // We set opType, looked up in a map from the string node.op_type()
      opType(getOpTypes().get(node.op_type())),
      // pointer to the graph containing this node
      pgraph(pg), id(pgraph->getAndIncrOpsCounter()),
      // neuralnet::Attributes constructed from contained of onnx::Attribute s
      nAtts(node.attribute()),
      // We set the pointer to the string version of opType, in another map
      p_op_type(&getOpTypes().get(opType)),
      // And finally we strip off the domain of the Node
      op_domain(node.domain()) {}

std::unique_ptr<Op> Graph::addOp(const Node &node) {
  using pOp = std::unique_ptr<Op>;
  switch (getOpTypes().get(node.op_type())) {
  case OpType::AVERAGEPOOL: {
    return pOp(new AveragePoolOp(node, this));
  }
  case OpType::CONSTANT: {
    throw error("ILE. Constant Ops are not to be added");
  }
  case OpType::CONV: {
    return pOp(new ConvOp(node, this));
  }
  case OpType::LOGSOFTMAX: {
    return pOp(new LogSoftmaxOp(node, this));
  }
  case OpType::NEGLOGLIKE: {
    throw error("cannot construct NegLogLike from a Node");
    // return pOp(new NegLogLikeOp( node, this));
  }
  case OpType::PAD: {
    return pOp(new PadOp(node, this));
  }
  case OpType::RELU: {
    return pOp(new ReluOp(node, this));
  }
  case OpType::SUM: {
    throw error("no constructor from node for Sum Op yet");
  }
  case OpType::SQUEEZE: {
    return pOp(new SqueezeOp(node, this));
  }
  case OpType::SQUEEZEGRAD:
  case OpType::RELUGRAD:
  case OpType::AVERAGEPOOLGRAD:
  case OpType::CONVDATAGRAD:
  case OpType::CONVWEIGHTSGRAD:
  case OpType::VARUPDATE:
    throw error("Gradient Ops not constructable from Node");

  default: { throw error("No class for " + node.op_type()); }
  }
}

std::string Op::str() const {
  return std::to_string(id) + " (" + op_type() + ')';
}

Op *LossOp::getNonGradOp() { return this; }

int LossOp::getNonGradInIndex(int index) const { return index; }

} // namespace neuralnet
