#include <array>
#include <fstream>
#include <map>
#include <queue>
#include <sstream>
#include <vector>
#include <willow/error.hpp>
#include <willow/filereader.hpp>
#include <willow/intervals.hpp>
#include <willow/ir.hpp>
#include <willow/loss.hpp>
#include <willow/optimizer.hpp>
#include <willow/patterns.hpp>
#include <willow/pbwrap.hpp>
#include <willow/tensor.hpp>
#include <willow/tensorinfo.hpp>
#include <willow/util.hpp>

// The layers:
#include <willow/add.hpp>
#include <willow/averagepool.hpp>
#include <willow/conv.hpp>
#include <willow/logsoftmax.hpp>
#include <willow/pad.hpp>
#include <willow/relu.hpp>
#include <willow/squeeze.hpp>
#include <willow/sum.hpp>
#include <willow/varupdate.hpp>

namespace willow {

void Ir::updateOptimizer(const Optimizer *) {
  throw error("update optimizer not implemented. Must throw if incompat");
}

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

void Ir::eraseOp(OpId id) {
  auto found = ops.find(id);
  if (found == ops.end()) {
    throw error("ILE: no op " + std::to_string(id) + " to erase");
  }
  ops.erase(id);
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

std::string getWillowDomain() { return "gnilwen.semaj"; }

OpTypes initOpTypes() { return OpTypes(); }

const OpTypes &getOpTypes() {
  const static OpTypes X = initOpTypes();
  return X;
}

// A note on non-determinism. For maps with
// pointers as keys, iterating through them
// is non-deterministic with the default comparitor.
// To prevent non-determinism in getTopologicallSorted,
// we could use the following non-default comparitor
// everywhere where there is a map with Op pointers,
// and a similar one with Tensor pointers. A fair amount
// of work...
struct POpCmp {
  bool operator()(const Op *const &a, const Op *const &b) const {
    return a->id < b->id;
  }
};

class OpPriorityComparer {
public:
  bool operator()(const Op *const &op1, const Op *const &op2) const {
    return op1->priority < op2->priority;
  }
};

void Op::setup() { throw error("No setup() for " + op_type()); }

// Essentially Kahn's alogorithm (1962, 56 years ago!),
// see https://en.wikipedia.org/wiki/Topological_sorting
// but not quite Kahn's algorithm as it there are some
// additional constraints on the order of Ops imposed
// externally. Also not quite Kahn, as the vertices which
// are ready to be inserted have an insertion "priority"
// set externally
std::vector<Op *> Ir::getTopologicallySorted() const {
  // the topological sorting (to construct in this function)
  std::vector<Op *> sorted;
  // ops which have all their input tensors
  // created, and are not waiting for any ops
  // to run before them
  // OpPriorityComparer opCompare;
  std::priority_queue<Op *, std::vector<Op *>, OpPriorityComparer> opsToProcess;
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
            opsToProcess.push(op_count.first);
          }
        }
      };

  // we will start by processing
  // the tensors which have no producers
  auto t0 = tensors.getNoProducerIds();
  for (auto &id : t0) {
    processTensor(tensors.get(id));
  }

  while (!opsToProcess.empty()) {
    auto op = opsToProcess.top();
    opsToProcess.pop();
    sorted.push_back(op);
    for (Op *waitingOp : isWaitingFor[op]) {
      --nOpsAwaiting[waitingOp];
      if (readyToProcess(waitingOp)) {
        opsToProcess.push(waitingOp);
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

void Ir::exportDot(const std::string dotfn) const {
  std::ofstream strm;
  strm.open(dotfn, std::ios::out);
  if (!strm.is_open()) {
    throw error("failed to open file `" + dotfn + '\'');
  }
  strm << "digraph net {\n";
  strm << "size=\"6,6\";\n";
  for (auto &n : getTopologicallySorted()) {
    strm << "n_" << n->id << " [shape= \"box\", label=\"" << n->op_type()
         << "\"];\n";
    for (auto &ind_ten : n->input.tensorMap()) {
      strm << ind_ten.second->id << " -> n_" << n->id << ';' << '\n';
    }

    for (auto &ind_ten : n->output.tensorMap()) {
      auto tenId = ind_ten.second->id;
      strm << "n_" << n->id << " -> " << tenId << ';' << '\n';
      TensorId possibleGradId = getGradId(tenId);
      if (tensors.contains(possibleGradId)) {
        // strm << "{rank=same; " << tenId << "; " << possibleGradId << ";}\n";
      }
    }
  }
  strm << '}' << '\n';
  strm.flush();
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

Tensors::Tensors(const std::vector<std::string> &vals1, Ir *pg)
    : constIds(vals1), pir(pg) {}

VectorAndSet::~VectorAndSet() = default;
Tensors::~Tensors()           = default;
Ir::~Ir()                     = default;

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
  Tensor *ptensor = pir->tensors.get(tenId);
  input.insert(inIndex, ptensor);
  ptensor->consumers.increment(this);
}

void Op::createAndConnectOutTensor(OutIndex outIndex, TensorId tenId) {
  pir->tensors.addActGrad(tenId);
  Tensor *ptensor = pir->tensors.get(tenId);
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

// willow streams and prints are "impolite" (will not add new line at end)

void Op::append(std::stringstream &ss) const {
  appendIO(ss);
  ss << '\n';
  appendMore(ss);
}

Speck Op::inputSpeckAt(int index) const {
  // perform a health check: is the index a valid input index?
  if (!input.hasIndex(index)) {
    throw error("no input index " + std::to_string(index));
  }
  // this is the default return type for an Op, this will be overwritten
  // where specific Specks are needed.
  return Speck::Any;
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

VectorAndSet::VectorAndSet(const std::vector<std::string> &vals)
    : v_vals(vals) {
  for (auto &v : v_vals) {
    m_vals.insert(v);
  }
}

void EarlyInfo::addInfo(TensorId id, const TensorInfo &info) {
  infos[id] = info;
}

const TensorInfo &EarlyInfo::getInfo(TensorId id) const { return infos.at(id); }

bool EarlyInfo::hasInfo(TensorId id) const {
  return infos.find(id) != infos.end();
}

void Ir::confirmNoReservedIds() const {

  auto &onnxGraph = onnxModel.graph();

  for (const auto &in_ : onnxGraph.input()) {
    confirmNonReservedId(in_.name());
  }

  for (const auto &out_ : onnxGraph.output()) {
    confirmNonReservedId(out_.name());
  }

  for (const auto &tenId : earlyInfo.getAllTensorIds()) {
    confirmNonReservedId(tenId);
  }
}

std::vector<TensorId> EarlyInfo::getAllTensorIds() const {
  // we first put the TensorIds into a set, so that duplication is removed
  std::set<TensorId> all_;
  for (const auto &id_info : infos) {
    all_.insert(id_info.first);
  }

  // then any other TensorIds from other maps in EarlyInfo will be added
  // to the set here.

  std::vector<TensorId> all;
  for (auto &x : all_) {
    all.push_back(x);
  }
  return all;
}

void Ir::setAllNodeInputsMap() {
  for (auto &node : onnxModel.graph().node()) {
    for (auto &name : node.input()) {
      allNodeInputsMap.insert(name);
    }
  }
}

IrBundle::IrBundle(std::string fnModel_,
                   const EarlyInfo &earlyInfo_,
                   const DataFlow &dataFlow_,
                   const std::vector<Loss *> &losses_,
                   const Optimizer *optimizer_,
                   const std::vector<std::string> &cTens_,
                   std::string logdir_,
                   const std::vector<std::string> &patternNames_)
    : fnModel(fnModel_), earlyInfo(earlyInfo_), dataFlow(dataFlow_),
      losses(losses_), optimizer(optimizer_), cTens(cTens_), logdir(logdir_),
      patternNames(patternNames_) {}

Ir::Ir(const IrBundle &gb)
    : tensors(gb.cTens, this), logdir(io::getCanonicalDirName(gb.logdir)),
      earlyInfo(gb.earlyInfo), dataFlow(gb.dataFlow) {

  optimizer = gb.optimizer->clone();

  io::confirmRegularFile(gb.fnModel);
  onnxModel = io::getModel(gb.fnModel);

  // TODO: this will change, obtained from optimizer:
  earlyInfo.addInfo(getLearningRateId(), {TP::FLOAT, {}});

  for (auto &l : gb.losses) {
    losses.emplace_back(l->clone());
  }

  confirmNoReservedIds();
  setAllNodeInputsMap();
  auto &onnxGraph = onnxModel.graph();
  std::set<TensorId> onnxInitializers;
  for (const auto &initializer : onnxGraph.initializer()) {
    TensorId tenId = initializer.name();
    addInitIfUsed(tenId, &initializer);
    onnxInitializers.emplace(tenId);
  }

  // onnx inputs which are not initializers are true inputs
  for (auto &valueInfo : onnxGraph.input()) {
    if (onnxInitializers.count(valueInfo.name()) == 0) {
      tensors.addStream(valueInfo.name());
    }
  }
  // other true inputs are for the loss calculation (class labels, etc)
  for (const auto &loss : losses) {
    for (const auto &tenId : loss->getStreamTensorNames()) {
      // another loss might have already registered this tensor
      if (!tensors.contains(tenId)) {
        tensors.addStream(tenId);
      } else {
        Tensor *tensorAlreadyPresent = tensors.get(tenId);
        if (tensorAlreadyPresent->tensorType() != TensorType::Stream) {
          throw error("type mismatch for tensor " + tenId);
        }
      }
    }
  }

  for (auto patternName : gb.patternNames) {
    switch (getPatternTypes().get(patternName)) {
    case PatternType::PREUNIREPL: {
      patterns.emplace_back(std::unique_ptr<Pattern>(new PreUniRepl));
      break;
    }

    case PatternType::POSTNREPL: {
      patterns.emplace_back(std::unique_ptr<Pattern>(new PostNRepl));
      break;
    }
    }
  }

  constructForwards();

  exportDot(io::appendDirFn(logdir, "jamForward0.dot"));
  // to developers: confirm fuctions like this
  // should be to check that there
  // are no contradictions in the user input, NOT
  // in the implementation of the library willow.
  confirmConstIds();

  splitConvBias();
  for (auto &pattern : patterns) {
    applyPattern(pattern.get());
  }
  growFinalLoss();
  setNPathsToLoss();

  constructBackwards();

  // confirm that all the anchor names provided
  // are indeed real tensor names. This is a check
  // that the user has not provided incorrect names.
  // We allow duplicates.
  validateAnchors();

  for (auto &pattern : patterns) {
    applyPattern(pattern.get());
  }

  prune();
  inferTensorInfos();
  addRecompute();
  prune();
  inferTensorInfos();

  exportDot(io::appendDirFn(logdir, "jam.dot"));

  std::stringstream ss2;
  append(ss2);
  std::cout << ss2.str();
}

std::vector<Op *> Ir::getTopologicallySortedTilLoss() const {
  std::vector<Op *> opsTowardsLoss;
  for (auto op : getTopologicallySorted()) {
    if (op->nPathsToLoss() > 0) {
      opsTowardsLoss.push_back(op);
    }
  }
  return opsTowardsLoss;
}

std::vector<std::set<Op *>>
Ir::getLiveSets(const std::vector<Op *> &topoOps) const {

  // the key op waits for the ops in val
  // so the key op is later in the sort.
  std::map<Op *, std::vector<Op *>> waiting;

  // the number of ops that are waiting for key
  // this is NOT the size of the values of is_waiting_for
  std::map<Op *, int> nWaiting;

  for (Op *op : topoOps) {
    nWaiting[op] = 0;
    waiting[op]  = {};
  }
  for (Op *op : topoOps) {
    for (auto t_inds : op->input.indicesMap()) {
      Tensor *tensor = t_inds.first;
      if (tensor->hasProducer()) {
        Op *prod = tensor->getProducer();
        // have we noted that op is waiting for prod yet? if not,
        if (std::find(waiting[op].begin(), waiting[op].end(), prod) ==
            waiting[op].end()) {
          // make note
          waiting[op].push_back(prod);
          // increase the number of ops waiting for prod
          ++nWaiting[prod];
        }
      }
    }
  }

  std::set<Op *> live = {};
  std::vector<std::set<Op *>> liveSets;
  for (Op *newOp : topoOps) {
    for (Op *isEarlier : waiting[newOp]) {
      if (live.count(isEarlier) == 0) {
        throw error(
            "ILE: op should still be live (newOp waits for its output)");
      }
      --nWaiting[isEarlier];
      if (nWaiting[isEarlier] == 0) {
        live.erase(isEarlier);
      }
    }
    live.insert(newOp);
    liveSets.push_back(live);
  }
  return liveSets;
}

int64_t Op::memOfOutputs() const {
  int64_t mem = 0;
  for (auto &t_inds : output.indicesMap()) {
    mem += t_inds.first->info.nbytes();
  }
  return mem;
}

void Ir::addRecompute() {
  std::vector<Op *> fwdOps = getTopologicallySortedTilLoss();

  // liveSets[i] : set of ops whose outputs have not all
  // been consumed by their (non-grad) consumers just after
  // linearised[i] has run. By this defn,
  // linearised[i] \in live[i]
  std::vector<std::set<Op *>> liveSets = getLiveSets(fwdOps);

  // The memory (bytes) which will be needed to
  // store all the output tensors in a liveness set.
  std::vector<int64_t> memoryOfLives;
  for (auto &liveSet : liveSets) {
    int64_t mem = 0;
    for (auto op : liveSet) {
      mem += op->memOfOutputs();
    }
    memoryOfLives.push_back(mem);
  }

  int nFwdOps = static_cast<int>(fwdOps.size());
  if (nFwdOps != liveSets.size() || memoryOfLives.size() != nFwdOps) {
    throw error("ILE : sizes of vectors do not match");
  }

  // TODO : this should change. resnet-50 has way more memory for early layers.
  // see
  // https://github.com/albanie/convnet-burden/blob/master/reports/resnet18.md
  // It should take in memoryOfLives, make intervals on cumulative memory.
  std::vector<std::array<int, 2>> intervals = getDecreasingIntervals(nFwdOps);

  //   defn, checkpoints: Ops whose
  //   outputs we guarantee will be available
  //   at any time
  std::set<Op *> checkpoints;

  // we choose the lowest memory set from each interval,
  // and add its members to checkpoints.
  for (auto interval : intervals) {
    int begin            = interval[0];
    int end              = interval[1];
    int64_t lowestMemory = std::numeric_limits<int64_t>::max();
    std::set<Op *> bestSet{};
    for (int i = begin; i < end; ++i) {
      if (memoryOfLives[i] < lowestMemory) {
        lowestMemory = memoryOfLives[i];
        bestSet      = liveSets[i];
      }
    }
    for (Op *op : bestSet) {
      if (checkpoints.count(op) == 0) {
        checkpoints.insert(op);
      }
    }
  }

  // all non-checkpoint pre-loss nodes.
  std::vector<Op *> nonCheckpoints;
  for (auto &op : fwdOps) {
    if (checkpoints.count(op) == 0) {
      nonCheckpoints.push_back(op);
    }
  }

  for (auto &op : nonCheckpoints) {
    growRecomputeOp(op, checkpoints);
  }
}

std::unique_ptr<Op> GradOp::clone() const {
  throw error("no clone for GradOp " + op_type() + " (not thought necessary)");
}

// see diagram 74 in notebook ;/)
Op *Ir::growRecomputeOp(Op *oriOp, const std::set<Op *> &checkpoints) {

  // the recompute op:
  OpId rcId = moveIntoIr(oriOp->clone());

  Op *rcOp = ops[rcId].get();

  // set inputs and outputs of  the new Op.
  std::map<int, TensorId> inputs;
  for (auto &index_tensor : oriOp->input.tensorMap()) {
    int index      = index_tensor.first;
    Tensor *tensor = index_tensor.second;
    // if the tensor was produced by a non-checkpointed op,
    // we need to use the recomputed version of it
    if (tensor->hasProducer() &&
        checkpoints.count(tensor->getProducer()) == 0) {
      inputs[index] = getRecompId(tensor->id);
    } else {
      inputs[index] = tensor->id;
    }
  }
  connectInputs(InputMapWrapper(inputs), rcId);

  std::map<int, TensorId> outputs;
  for (auto &index_tensor : oriOp->output.tensorMap()) {
    int index            = index_tensor.first;
    const Tensor *tensor = index_tensor.second;
    outputs[index]       = getRecompId(tensor->id);
  }
  connectOutputs(OutputMapWrapper(outputs), rcId);

  // yank down the priority of the new Op
  // (must be run as late as possible):
  rcOp->priority = std::numeric_limits<double>::lowest();

  // oriOp's outputs should not be consumed by grad op:
  for (auto &ind_ten : oriOp->output.tensorMap()) {
    Tensor *oriTen = ind_ten.second;
    Tensor *recTen = tensors.get(getRecompId(oriTen->id));
    for (auto &con : oriTen->consumers.getOps()) {
      if (con->isGradOp()) {
        for (auto &con_ind_ten : con->input.tensorMap()) {
          int gradIn = con_ind_ten.first;
          if (con_ind_ten.second == oriTen) {
            con->input.reset(gradIn, recTen);
            recTen->consumers.increment(con);
            oriTen->consumers.decrement(con);
          }
        }
      }
    }
  }

  // note: oriOp will still be pointed to
  // by grad op as it's creator. This design
  // choice might need revision.

  return rcOp;
}

void Tensors::append(std::stringstream &ss) const {
  bool frst = true;
  ss << '[';
  for (auto &id_ptr : M) {
    if (!frst) {
      ss << ' ';
    }
    frst = false;
    ss << id_ptr.first;
  }
  ss << ']';
}

void Ir::validateAnchors() const {
  for (TensorId id : dataFlow.anchors()) {
    if (!tensors.contains(id)) {
      std::stringstream ss;
      ss << "Anchor tensor `" << id << "' not in tensors. ";
      // add some trouble-shooting for a case I stumbled upon:
      if (id.find(reservedGradientPrefix()) != std::string::npos) {
        std::string degrad = id.substr(reservedGradientPrefix().size());
        if (tensors.contains(degrad)) {
          ss << "\nInterestingly, `" << degrad << '\'' << " IS in tensors.\n";
          ss << "Note that not all tensors can have their gradients "
             << "anchored:\nif an activation tensor does not lead "
             << "to the loss,\nits gradient is zero and never computed.";
        }
      } else {
        ss << "The tensors are:\n";
        tensors.append(ss);
      }
      throw error(ss.str());
    }
  }
}

const std::vector<TensorId> &DataFlow::anchors() const { return v_anchors; }

int DataFlow::nAnchors() const { return static_cast<int>(v_anchors.size()); }

void Ir::prune() {

  // initialise with all the var
  // update ops for training,
  // and work backwards. This
  // is the set which is returned
  std::set<Op *> required = trainTargetOps;

  // as we work backwards, we keep a
  // "front" of tensors,
  std::vector<Tensor *> tensorFront;

  // when a tensor enters the "front",
  // we record that it has been visited
  std::set<Tensor *> tensorsVisited;

  // the "front" is initialsed with (1) anchor tensors,
  for (auto &tensorId : dataFlow.anchors()) {
    Tensor *t = tensors.get(tensorId);
    // we have this check here as we allow
    // duplicated names from the (careless!) user
    if (tensorsVisited.count(t) == 0) {
      tensorFront.push_back(t);
      tensorsVisited.insert(t);
    }
  }

  // and (2), inputs to the training targets.
  for (auto &op : trainTargetOps) {
    for (auto t_inds : op->input.indicesMap()) {
      Tensor *t = t_inds.first;
      if (tensorsVisited.count(t) == 0) {
        tensorFront.push_back(t);
        tensorsVisited.insert(t);
      }
    }
  }

  while (tensorFront.size() != 0) {
    Tensor *t = tensorFront.back();
    tensorFront.resize(tensorFront.size() - 1);
    if (t->hasProducer()) {
      Op *op = t->getProducer();
      if (required.count(op) == 0) {
        required.insert(op);
        for (auto t_inds : op->input.indicesMap()) {
          Tensor *t_in = t_inds.first;
          if (tensorsVisited.count(t_in) == 0) {
            tensorFront.push_back(t_in);
            tensorsVisited.insert(t_in);
          }
        }
      }
    }
  }

  // at this point, "required" is the set
  // of all ops which are actually executed
  // to get targets

  // ops \ required
  std::vector<Op *> opsToDelete;
  // all outputs of opsToDelete
  std::vector<Tensor *> tensorsToDelete;

  for (auto &id_op : ops) {
    Op *op = id_op.second.get();
    if (required.count(op) == 0) {
      opsToDelete.push_back(op);
      for (auto &t_inds : op->output.indicesMap()) {
        tensorsToDelete.push_back(t_inds.first);
      }
    }
  }

  for (Op *op : opsToDelete) {
    // unwire the inputs
    for (auto index_tensor : op->input.tensorMap()) {
      Tensor *tensor = index_tensor.second;
      tensor->consumers.decrement(op);
    }
    // and delete the Op
    ops.erase(op->id);
  }

  for (Tensor *tensor : tensorsToDelete) {
    tensors.remove(tensor->id);
  }
}

void Ir::applyPattern(const Pattern *pattern) {
  std::vector<Op *> v_ops;
  for (auto &id_op : ops) {
    v_ops.push_back(id_op.second.get());
  }
  for (auto op : v_ops) {
    if (pattern->matches(op)) {
      if (pattern->removesNoAnchored(op)) {
        pattern->apply(op);
      }
    }
  }
}

std::vector<TensorId> Tensors::getNoProducerIds() const {
  // the tensors which are not generated by an Op
  std::vector<TensorId> t0 = getIds(TensorType::Stream);
  std::vector<TensorId> t1 = getInitIds();
  t0.insert(t0.end(), t1.begin(), t1.end());
  return t0;
}

void Ir::inferTensorInfos() {
  for (const auto &tensorId : tensors.getInitIds()) {
    auto pt = tensors.getOnnxInit(tensorId);
    tensors.get(tensorId)->info.set(*pt);
  }

  std::vector<TensorId> streamTensors = tensors.getIds(TensorType::Stream);
  for (const auto &id : streamTensors) {
    if (!(earlyInfo.hasInfo(id))) {
      throw error("expected pre-run knowledge for stream tensor " + id);
    }
    tensors.get(id)->info = earlyInfo.getInfo(id);
  }

  for (Op *op : getTopologicallySorted()) {
    op->setup();
  }
}

const onnx::TensorProto *Tensors::getOnnxInit(TensorId id) const {
  return init.at(id);
}

std::vector<Op *> Ir::opsOfType(OpType opType) {
  std::vector<Op *> typedOps;
  for (auto &id_op : ops) {
    if (id_op.second->opType == opType) {
      typedOps.push_back(id_op.second.get());
    }
  }
  return typedOps;
}

int TensorIndexMap::n() const { return static_cast<int>(tensor_map.size()); }

bool Ir::isAnchored(TensorId tenId) { return dataFlow.isAnchored(tenId); }

void Ir::splitConvBias() {}

const std::vector<std::string> &VectorAndSet::v() const { return v_vals; }

void Ir::confirmConstIds() const {
  for (auto &tensorId : tensors.constIds.v()) {
    if (!tensors.contains(tensorId)) {
      throw error("no tensor " + tensorId +
                  " in tensors, error in const tensor names");
    }
  }
}

void Ir::constructForwards() {
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

void Ir::addInitIfUsed(TensorId id, const onnx::TensorProto *t) {
  if (allNodeInputsMap.count(id) != 0) {
    tensors.addInit(id, t);
  } else {
    std::cout << "unused ONNX tensor " << id << std::endl;
  }
}

void Tensors::addInit(TensorId name, const onnx::TensorProto *pt) {
  init[name] = pt;
  insert(name,
         std::unique_ptr<Tensor>(new Tensor(
             name,
             constIds.contains(name) ? TensorType::Const : TensorType::Variable,
             pir)));
}

std::string reservedGradientPrefix() { return "d__"; }
std::string reservedRecomputePrefix() { return "r__"; }
std::vector<std::string> reservedPrefixes() {
  return {reservedGradientPrefix(), reservedRecomputePrefix()};
}

void Tensors::addActGrad(TensorId tenId) {
  insert(tenId,
         std::unique_ptr<Tensor>(new Tensor(tenId, TensorType::ActGrad, pir)));
}

void Ir::confirmNonReservedId(TensorId tenId) const {
  for (auto reservedPrefix : reservedPrefixes()) {
    if (tenId.find(reservedPrefix) != std::string::npos) {
      throw error("Provided tensor " + tenId +
                  " has an invalid name: clash with reserved prefix " +
                  reservedPrefix);
    }
  }
}

void Tensors::addStream(TensorId tenId) {
  insert(tenId,
         std::unique_ptr<Tensor>(new Tensor(tenId, TensorType::Stream, pir)));
}

TensorId getGradId(TensorId id) { return reservedGradientPrefix() + id; }

TensorId getRecompId(TensorId id) { return reservedRecomputePrefix() + id; }

TensorId getNonGradId(TensorId id) {
  return id.substr(reservedGradientPrefix().size());
}

TensorId getEdgeGradId(TensorId tenId, OpId opId, int index) {
  // we don't need the name of the tensor which this is an edge-grad of,
  // the edge-gradient is uniquely defined by the the edge it flows on
  // in the forward pass (input at 'index' to 'opId')
  (void)tenId;
  std::stringstream ss;
  ss << reservedGradientPrefix() << opId << '_' << index;
  TensorId edgeGradId = ss.str();
  return edgeGradId;
}

void Tensors::remove(TensorId id) { M.erase(id); }

bool Tensors::contains(TensorId id) const { return M.find(id) != M.end(); }

OpId Ir::getAndIncrOpsCounter() {
  OpId nOps0 = opsCounter;
  ++opsCounter;
  return nOps0;
}

OpId Ir::getOpsCounter() const { return opsCounter; }

OpId Ir::moveIntoIr(std::unique_ptr<Op> op) {
  OpId id = op->id;
  ops[id] = std::move(op);
  return id;
}

Op *Ir::growGradSumOp(Tensor *target, const std::vector<Tensor *> &toSum) {

  OpId opId = moveIntoIr(
      std::unique_ptr<Op>(new SumOp({"Sum", this, {}, getWillowDomain()})));

  std::vector<TensorId> inputs;
  inputs.reserve(toSum.size());
  for (auto &tensor : toSum) {
    inputs.push_back(tensor->id);
  }
  TensorId gradientId = getGradId(target->id);
  std::vector<TensorId> outputs{gradientId};

  connectInputs(InputVecWrapper(inputs), opId);
  connectOutputs(OutputVecWrapper(outputs), opId);
  return ops[opId].get();
}

std::vector<Op *> Ir::growGradOps(Op *nonGradOp) {

  OpId nonGradOpId = nonGradOp->id;
  auto backOps     = nonGradOp->getGradOps();
  std::vector<Op *> gradOps;
  for (auto &upop : backOps) {
    Op *gradOp    = upop.get();
    OpId gradOpId = moveIntoIr(std::move(upop));

    // connect inputs of gradOp
    {
      // inputs to gradOp (to populate in this scope):
      std::map<int, std::string> m_inputs;
      //  int max_input_index = 0;
      for (auto &inOutMapper : gradOp->gradInputInfo()) {

        int indexGrad     = inOutMapper.iGrad;
        int indexFwd      = inOutMapper.iNonGrad;
        GradOpInType type = inOutMapper.type;

        //  max_input_index = std::max(indexGrad, max_input_index);

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
               << "did not lead to final loss, "
               << "in which case the gradient is zero?";
            throw error(ss.str());
          }
          m_inputs[indexGrad] =
              getGradId(nonGradOp->output.tensor(indexFwd)->id);
          break;
        }
        }
      }

      connectInputs(InputMapWrapper(m_inputs), gradOpId);
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
      connectOutputs(OutputVecWrapper(v_outputs), gradOpId);
    }
    // note, as the outputs of gradOp are edge-grad-tensors and not
    // edge-grads, we do not need to match them to non-grad tensors.
    gradOps.push_back(gradOp);
  }

  return gradOps;
}

// the default is that there are no topo cons
void Op::imposeTopoCons() {}

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

// design choice: I could have a "irHasModeified"
// flag which is set to true whenever the Ir changes,
// and then if irHasModeified is false, calls
// to this (and other) functions can do nothing.
// The cost of maintaining irHasModeified is non-trivial
// and would require runtime overhead, for now I'm not
// going to implement it.
void Ir::setNPathsToLoss() {

  // initialize number of paths for
  // all Ops and Tensors to loss to be zero
  for (auto &id_op : ops) {
    Op *op = id_op.second.get();
    op->setNPathsToLossToZero();
    for (auto t_inds : op->input.indicesMap()) {
      t_inds.first->setNPathsToLossToZero();
    }
    for (auto t_inds : op->output.indicesMap()) {
      t_inds.first->setNPathsToLossToZero();
    }
  }

  std::vector<Op *> opFront{getFinalLossOp()};
  std::set<Op *> opsSeen{getFinalLossOp()};
  std::set<Tensor *> tensorsSeen{};
  while (opFront.size() != 0) {
    Op *op = opFront.back();
    opFront.resize(opFront.size() - 1);
    for (auto &ind_ten : op->input.tensorMap()) {
      auto tensor = ind_ten.second;
      tensor->incrNPathsToLoss();
      if (tensorsSeen.count(tensor) == 0) {
        tensorsSeen.insert(tensor);
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
}

void Ir::constructBackwards() {
  // definition: edge-gradient. What is output by a grad-op,
  // and which will be summed with other edge-gradients to create
  // a gradient. It is possible that an edge-gradient has the same
  // value as a gradient, if a tensor has only 1 consumer.

  // design decision w.r.t. lambda functions in this function:
  // see-sawing between lambda functions (see two following here)
  // and member functions. In general I don't like lambda functions,
  // their return types are not easily visible and capturing parameters
  // is tedious. However, I also don't like having class variables
  // which are only used in one bit of functionality, because it becomes
  // unclear whether they should be maintained in a valid state throughout
  // the objects life. In this case, I think the second is worse, so
  // going for the lambda solution.
  TensorGradRegistry tensor_grad_registry;
  OpGradRegistry op_grad_registry;

  // signal that a grad-op has created edge-gradients
  auto registerOpGrads = [&tensor_grad_registry](Op *gradOp, Op *nonGradOp) {
    for (auto &index_tensor : gradOp->output.tensorMap()) {
      int opOutInd     = index_tensor.first;
      Tensor *partGrad = index_tensor.second;
      // what input index of nonGradOp does the
      // edge-gradient correspond to?
      int nonGradInInd      = gradOp->getNonGradInIndex(opOutInd);
      Tensor *nonGradTensor = nonGradOp->input.tensor(nonGradInInd);
      tensor_grad_registry.insert(nonGradTensor, partGrad);
    }
  };

  // communicate that a new gradient tensor
  // (which is a sum along edges) is ready
  auto registerTensorGrad = [this, &op_grad_registry](Tensor *sum) {
    Tensor *nonGrad = tensors.get(getNonGradId(sum->id));
    if (nonGrad->hasProducer()) {
      Op *producer = nonGrad->getProducer();
      // the index at which nonGrad was produced
      int index = producer->output.indices(nonGrad).at(0);
      op_grad_registry.insert(producer, index);
    }
  };

  // grad-ops which have created edge-gradients, but the
  // edge-gradients haven't signalled their existance.
  // initialised as the gradients of the individual losses
  std::vector<Op *> opsToRegister = growLossGradients();

  while (!opsToRegister.empty()) {

    registerOpGrads(opsToRegister.back(),
                    opsToRegister.back()->getNonGradCreator());
    opsToRegister.resize(opsToRegister.size() - 1);

    for (auto &nongrad_egrads : tensor_grad_registry.popComplete()) {

      Tensor *nongrad                     = nongrad_egrads.first;
      const std::vector<Tensor *> &egrads = nongrad_egrads.second;
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
      case TensorType::Stream: {
        // if the user wants the gradient of the
        // input data (unusual case) maybe we won't
        // break here. Example case : generating adversarials
        break;
      }
      case TensorType::Const:
      case TensorType::Momentum:
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

  tensors.addStream(getLearningRateId());

  // add weight ops (ignoring momentum's for now)
  for (auto &varId : tensors.getIds(TensorType::Variable)) {
    growVarUpdateOp(varId);
  }
}

TensorId getLearningRateId() { return "learnRate"; }

Op *Ir::growVarUpdateOp(TensorId varId) {

  OpId opId = moveIntoIr(std::unique_ptr<Op>(new VarUpdateOp(varId, this)));
  Op *op    = ops[opId].get();

  std::vector<TensorId> inputs(3, "");
  inputs[VarUpdateOp::getVarIndex()]       = varId;
  inputs[VarUpdateOp::getVarGradIndex()]   = getGradId(varId);
  inputs[VarUpdateOp::getLearnRateIndex()] = getLearningRateId();
  connectInputs(InputVecWrapper(inputs), opId);

  // there are no outputs of var-op
  std::vector<TensorId> outputs{};
  connectOutputs(OutputVecWrapper(outputs), opId);

  trainTargetOps.insert(op);
  return op;
}

const std::map<int, Tensor *> &TensorIndexMap::tensorMap() const {
  return tensor_map;
}

Op *Op::getNonGradCreator() const {
  throw error("No `get non grad op' for " + op_type() + " (yet?)");
}

Op *Ir::growFromNode(const Node &node) {

  // special case of CONSTANT Node, no Op is created
  if (getOpTypes().get(node.op_type()) == OpType::CONSTANT) {
    TensorId name = node.output(0);

    // we confirm that this tensor is actually
    // the input of some Node in the onnx::Graph, because
    // we've seen (in pytorch) that some initializers
    // are not used (always '2', '3', '4' of shape (10,10,3,3)
    addInitIfUsed(name, &node.attribute(0).t());
    // no Op created for a Constant Node
    return nullptr;
  }

  OpId opId = moveIntoIr(addOp(node));

  connectInputs(node, opId);
  connectOutputs(node, opId);
  return ops[opId].get();
}

Op *Ir::getFinalLossOp() {
  if (finalLossOp == nullptr) {
    throw error("ILE : final loss not set");
  }
  return finalLossOp;
}

void Ir::growFinalLoss() {
  std::vector<Op *> lossOps;
  for (auto &loss : losses) {
    OpId opId = moveIntoIr(loss->getOp(this));
    connectInputs(*loss, opId);
    connectOutputs(*loss, opId);
    lossOps.push_back(ops[opId].get());
  }

  // now growing the FINAL loss:
  OpId opId = moveIntoIr(
      std::unique_ptr<Op>(new SumOp({"Sum", this, {}, getWillowDomain()})));

  std::vector<TensorId> inputs;
  inputs.reserve(lossOps.size());
  for (auto &op : lossOps) {
    inputs.push_back(op->output.tensor(0)->id);
  }
  std::vector<TensorId> outputs{getFinalLossId()};
  connectInputs(InputVecWrapper(inputs), opId);
  connectOutputs(OutputVecWrapper(outputs), opId);
  finalLossOp = ops[opId].get();
}

TensorId Ir::getFinalLossId() const { return "finalLoss"; }

template <typename T> void Ir::connectInputs(const T &inContainer, OpId opId) {
  Op *op = ops[opId].get();
  for (int inIndex = 0; inIndex < inContainer.input_size(); ++inIndex) {
    auto &inName = inContainer.input(inIndex);
    if (inName == "") {
      // no input at this position
    } else {
      if (!tensors.contains(inName)) {
        throw error("input " + inName + " should already be in tensor map");
      } else {
        op->connectInTensor(inIndex, inName);
      }
    }
  }
  op->imposeTopoCons();
}

template <typename T>
void Ir::connectOutputs(const T &outContainer, OpId opId) {
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

  opTypes_ = {{"Add", OpType::ADD},
              {"AddGrad", OpType::ADDGRAD},
              {"AveragePool", OpType::AVERAGEPOOL},
              {"AveragePoolGrad", OpType::AVERAGEPOOLGRAD},
              {"Constant", OpType::CONSTANT},
              {"Conv", OpType::CONV},
              {"ConvDataGrad", OpType::CONVDATAGRAD},
              {"ConvWeightsGrad", OpType::CONVWEIGHTSGRAD},
              {"L1", OpType::L1},
              {"L1Grad", OpType::L1GRAD},
              {"LogSoftmax", OpType::LOGSOFTMAX},
              {"LogSoftmaxGrad", OpType::LOGSOFTMAXGRAD},
              {"Nll", OpType::NLL},
              {"NllGrad", OpType::NLLGRAD},
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

void Ir::append(std::stringstream &ss) {
  ss << "-- Ir --\n";
  //  for (auto &id_op : ops) {
  //    id_op.second->append(ss);
  //  }

  for (auto &op : getTopologicallySorted()) {
    op->append(ss);
  }
}

Op::Op(const Op &op)
    : priority(op.priority), opType(op.opType), pir(op.pir),
      id(pir->getAndIncrOpsCounter()), nAtts(op.nAtts), p_op_type(op.p_op_type),
      op_domain(op.op_domain) {
  // input, output: empty.
}

const std::string &Op::domain() { return op_domain; }

const std::string &Op::op_type() const { return *p_op_type; }

OpConstructorBundle::OpConstructorBundle(std::string op_type_,
                                         Ir *pir_,
                                         Attributes atts_,
                                         std::string domain_)
    : op_type(op_type_), pir(pir_), atts(atts_), domain(domain_) {}

Op::Op(const OpConstructorBundle &b)
    : // opType (from string op_type)
      opType(getOpTypes().get(b.op_type)),
      // the Ir
      pir(b.pir),
      // the id
      id(pir->getAndIncrOpsCounter()),
      // the Attributes
      nAtts(b.atts),
      // opType
      p_op_type(&getOpTypes().get(opType)),
      // domain
      op_domain(b.domain) {}

Op::Op(const Node &node, Ir *pg)
    : // We set opType, looked up in a map from the string node.op_type()
      opType(getOpTypes().get(node.op_type())),
      // pointer to the Ir containing this node
      pir(pg), id(pir->getAndIncrOpsCounter()),
      // willow::Attributes constructed from contained of onnx::Attribute s
      nAtts(node.attribute()),
      // We set the pointer to the string version of opType, in another map
      p_op_type(&getOpTypes().get(opType)),
      // And finally we strip off the domain of the Node
      op_domain(node.domain()) {}

std::unique_ptr<Op> Ir::addOp(const Node &node) {
  using pOp = std::unique_ptr<Op>;
  switch (getOpTypes().get(node.op_type())) {
  case OpType::ADD: {
    return pOp(new AddOp(node, this));
  }
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
  case OpType::ADDGRAD:
  case OpType::SQUEEZEGRAD:
  case OpType::RELUGRAD:
  case OpType::AVERAGEPOOLGRAD:
  case OpType::CONVDATAGRAD:
  case OpType::CONVWEIGHTSGRAD:
  case OpType::NLLGRAD:
  case OpType::L1GRAD:
  case OpType::LOGSOFTMAXGRAD:
  case OpType::VARUPDATE:
    throw error("Gradient Ops not constructable from Node");

  case OpType::NLL:
  case OpType::L1:
    throw error("Loss Ops not constructable from Node");

  default: { throw error("No class for " + node.op_type()); }
  }
}

std::string Op::str() const {
  return std::to_string(id) + " (" + op_type() + ')';
}

std::vector<Op *> Ir::growLossGradients() {
  std::vector<Op *> gradops;
  for (auto &t_inds : getFinalLossOp()->input.indicesMap()) {
    Tensor *t  = t_inds.first;
    Op *lossOp = t->getProducer();
    for (Op *gradop : growGradOps(lossOp)) {
      gradops.push_back(gradop);
    }
  }
  return gradops;
}

bool DataFlow::isAnchored(TensorId id) const {
  return (s_anchors.count(id) != 0);
}

DataFlow::DataFlow(int BpR, int bs, const std::vector<TensorId> &v)
    : batchesPerRecord(BpR), batchSize(bs), v_anchors(v) {
  for (auto &id : v_anchors) {
    s_anchors.insert(id);
  }
}

Op *Ir::getOp(OpId opId) {
  auto found = ops.find(opId);
  if (found == ops.end()) {
    throw error("No Op `" + std::to_string(opId) + "'");
  }
  return found->second.get();
}

} // namespace willow
