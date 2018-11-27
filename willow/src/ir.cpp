#include <algorithm>
#include <array>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <poponnx/error.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/intervals.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/logging.hpp>
#include <poponnx/op/loss.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/optionflags.hpp>
#include <poponnx/pbwrap.hpp>
#include <poponnx/scheduler.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/util.hpp>

// The patterns
#include <poponnx/patterns/convbias.hpp>
#include <poponnx/patterns/inplace.hpp>
#include <poponnx/patterns/optoidentitypattern.hpp>
#include <poponnx/patterns/postnrepl.hpp>
#include <poponnx/patterns/preunirepl.hpp>
#include <poponnx/patterns/softmaxgraddirect.hpp>
#include <poponnx/patterns/subtractarg1gradoppattern.hpp>

// The layers:
#include <poponnx/op/add.hpp>
#include <poponnx/op/averagepool.hpp>
#include <poponnx/op/conv.hpp>
#include <poponnx/op/identity.hpp>
#include <poponnx/op/matmul.hpp>
#include <poponnx/op/maxpool.hpp>
#include <poponnx/op/negate.hpp>
#include <poponnx/op/pad.hpp>
#include <poponnx/op/reducesum.hpp>
#include <poponnx/op/relu.hpp>
#include <poponnx/op/softmax.hpp>
#include <poponnx/op/squeeze.hpp>
#include <poponnx/op/subtract.hpp>
#include <poponnx/op/sum.hpp>
#include <poponnx/op/varupdate.hpp>

namespace willow {

bool Op::isLossOp() const { return false; }

std::unique_ptr<Op> Op::clone() const {
  throw error("No clone implemented for " + op_type());
}

GradNonGradPair::GradNonGradPair(Op *g_, Op *ng_) : grad(g_), nongrad(ng_) {}

GradNonGradPair::GradNonGradPair() : GradNonGradPair(nullptr, nullptr) {}

onnx::ModelProto Ir::getModel() const { return onnxModel; }

TensorId TensorIndexMap::id(int index) const { return tensor(index)->id; }

std::vector<Tensor *> Ir::optimizerTensors() const {
  if (optimizer.get() == nullptr) {
    throw error("ILE : No optimizerTensors til Optimizer is set");
  }

  std::vector<Tensor *> optTensors;
  for (auto &id_info : optimizer->tensorInfos()) {
    optTensors.push_back(tensors.get(id_info.first));
  }
  return optTensors;
}

// the rule followed : a Stream tensor which is not an
// optimizer tensor is a streamed data tensor
std::vector<Tensor *> Ir::dataStreamTensors() const {
  std::vector<Tensor *> dsTensors;
  auto optTensorInfo = optimizer->tensorInfos();
  for (TensorId id : tensors.getIds(TensorType::Stream)) {

    if (optTensorInfo.find(id) == optTensorInfo.end()) {
      dsTensors.push_back(tensors.get(id));
    }
  }
  return dsTensors;
}

void Ir::updateOptimizer(const Optimizer *newOptimizer) {
  if (optimizer.get() == nullptr) {
    throw error("ILE: cannot update optimizer before it is set");
  }
  if (!optimizer->validReplacement(newOptimizer)) {
    throw error("This Optimizer of type " + newOptimizer->type_s() +
                " is not a valid replacement for optimizer of type " +
                optimizer->type_s());
  }
  optimizer = newOptimizer->clone();
  optimizer->resetTensorDatas(this);
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

void Op::setup() { throw error("No setup() for " + op_type()); }

void Ir::exportDot(const std::string dotfn) const {
  std::ofstream strm;
  strm.open(dotfn, std::ios::out);
  if (!strm.is_open()) {
    throw error("failed to open file `" + dotfn + '\'');
  }
  strm << "digraph net {\n";
  strm << "size=\"6,6\";\n";
  // the position in the schedule at which an op runs
  int scheduleIndex = 0;
  for (auto &n : getOpSchedule({})) {
    strm << "n_" << n->id << " [shape= \"box\", label=\"" << scheduleIndex
         << '.' << ' ' << n->op_type() << "\"];\n";
    ++scheduleIndex;
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

Tensors::Tensors(Ir &pg) : ir(pg) {}

Tensors::Tensors(const std::vector<TensorId> &vals1, Ir &pg)
    : constIds(vals1), ir(pg) {}

void Tensors::setConstIds(const std::vector<TensorId> &vals) {
  constIds.reset(vals);
}

VectorAndSet::~VectorAndSet() = default;
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

  if (indices_map.find(ptensor) == indices_map.end()) {
    indices_map[ptensor] = {};
  }
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

void TensorIndexMap::erase(int index) {
  const auto tm_itr = tensor_map.find(index);

  if (tm_itr != tensor_map.end()) {
    const auto im_itr = indices_map.find(tm_itr->second);

    auto &imv    = im_itr->second;
    auto imv_itr = std::find(imv.begin(), imv.end(), index);

    // Remove the index from indices_map.
    if (imv_itr != imv.end()) {
      imv.erase(imv_itr);
    }

    // If the Tensor has no indices, remove it from indices_map.
    if (imv.empty()) {
      indices_map.erase(im_itr);
    }

    // Remove the tensor from the tensor_map.
    tensor_map.erase(tm_itr);
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
  Tensor *ptensor = pir->getTensors().get(tenId);
  input.insert(inIndex, ptensor);
  ptensor->consumers.increment(this);
}

void Op::createAndConnectOutTensor(OutIndex outIndex, TensorId tenId) {
  pir->getTensors().addActGrad(tenId);
  Tensor *ptensor = pir->getTensors().get(tenId);
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

VectorAndSet::VectorAndSet() {}

VectorAndSet::VectorAndSet(const std::vector<std::string> &vals)
    : v_vals(vals) {
  for (auto &v : v_vals) {
    m_vals.insert(v);
  }
}

void VectorAndSet::reset(const std::vector<std::string> &vals) {

  // Replace the old with the new
  v_vals = vals;

  // Clear and initialise the m_vals set
  m_vals.clear();
  for (auto &v : v_vals) {
    m_vals.insert(v);
  }
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

// used for circumventing the pytorch bug,
// where some initializers are not used :
// https://github.com/pytorch/pytorch/issues/13552
void Ir::setAllNodeInputsMap() {
  for (auto &node : onnxModel.graph().node()) {
    for (auto &name : node.input()) {
      allNodeInputsMap.insert(name);
    }
  }
}

IrBundle::IrBundle(const onnx::ModelProto &modelProto_,
                   const EarlyInfo &earlyInfo_,
                   const DataFlow &dataFlow_,
                   const std::vector<Loss *> &losses_,
                   const Optimizer *optimizer_,
                   const std::vector<std::string> &cTens_,
                   const std::string &logdir_,
                   const SessionOptions &userOptions_,
                   const std::vector<std::string> &patternNames_)
    : modelProto(modelProto_), earlyInfo(earlyInfo_), dataFlow(dataFlow_),
      losses(losses_), optimizer(optimizer_), cTens(cTens_), logdir(logdir_),
      userOptions(userOptions_), patternNames(patternNames_) {}

Ir::Ir() : tensors(*this) {}

// FFS : Guard against multiple calls to prepare

void Ir::prepare(const IrBundle &gb) {

  scheduler.reset(new Scheduler(this));

  tensors.setConstIds(gb.cTens);
  dataFlow    = gb.dataFlow;
  logdir      = io::getCanonicalDirName(gb.logdir);
  userOptions = gb.userOptions;
  earlyInfo   = gb.earlyInfo;

  onnxModel = gb.modelProto;

  // Q : Is the optimizer optional?
  if (gb.optimizer) {
    optimizer = gb.optimizer->clone();
    for (auto &id_info : optimizer->tensorInfos()) {
      TensorId id     = id_info.first;
      TensorInfo info = id_info.second;
      tensors.addStream(id, info);
      optimizer->setTensorData(tensors.get(id));
    }
  }

  // A (jn) : No
  else {
    throw error("Optimizer required in IrBundle");
  }

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
    TensorId id = valueInfo.name();
    if (onnxInitializers.count(id) == 0) {
      tensors.addStream(id, earlyInfo.get(id));
    }
  }
  // other true inputs are for the loss calculation (class labels, etc)
  for (const auto &loss : losses) {
    for (const auto &tenId : loss->getStreamTensorNames()) {
      // another loss might have already registered this tensor
      if (!tensors.contains(tenId)) {
        tensors.addStream(tenId, earlyInfo.get(tenId));
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

    case PatternType::SOFTMAXGRADDIRECT: {
      patterns.emplace_back(std::unique_ptr<Pattern>(new SoftmaxGradDirect));
      break;
    }

    case PatternType::SPLITCONVBIAS: {
      patterns.emplace_back(new ConvBiasPattern);
      break;
    }

    case PatternType::OPTOIDENTITY: {
      patterns.emplace_back(new OpToIdentityPattern);
      break;
    }

    case PatternType::SUBTRACTARG1GRADOP: {
      patterns.emplace_back(make_unique<SubtractArg1GradOpPattern>());
      break;
    }

    case PatternType::INPLACE0: {
      patterns.emplace_back(new Inplace0);
      break;
    }

    default:
      throw error("unrecognised PatternType");
    }
  }

  // construct the forward pass from ONNX,
  constructForwards();
  if (userOptions.exportDot) {
    exportDot(io::appendDirFn(logdir, "fwd0.dot"));
  }

  // This function checks that there
  // are no contradictions in the names provided
  // as constant tensors
  confirmConstIds();
  for (auto &pattern : patterns) {
    if (pattern->phase() == PatternPhase::PRETOPOCONS) {
      applyPattern(pattern.get());
    }
  }
  growFinalLoss();
  updateVertices();
  setNPathsToLoss();
  constructBackwards();
  updateVertices();

  // confirm that all the anchor names provided
  // are indeed real tensor names. This is a check
  // that the user has not provided incorrect names.
  // We allow duplicates.
  validateAnchors();
  prune();

  for (auto &pattern : patterns) {
    if (pattern->phase() == PatternPhase::PRETOPOCONS) {
      applyPattern(pattern.get());
    }
  }

  updateVertices();
  addRecompute();
  updateVertices();

  // we now start applying topological constraints between
  // Ops directly. First, we ensure that the VarUpdate Ops
  // are the final consumers of the Variable tensors
  setVarUpdateCons();
  if (userOptions.exportDot) {
    exportDot(io::appendDirFn(logdir, "fwdBwd0.dot"));
  }

  prune();

  // Now, we apply the Patterns which can handle and create
  // topological constaints. Currently, this is only one
  // in-placing Pattern.
  for (auto &pattern : patterns) {
    if (pattern->phase() == PatternPhase::WITHTOPOCONS) {
      applyPattern(pattern.get());
    }
  }

  updateVertices();

  if (userOptions.exportDot) {
    exportDot(io::appendDirFn(logdir, "fwdBwd1.dot"));
  }
  std::stringstream ss2;
  append(ss2);
  logging::ir::info(ss2.str());
}

void Ir::resetWeights(const onnx::ModelProto &modelProto) {
  auto onnxGraph = modelProto.graph();

  for (const auto &initializer : onnxGraph.initializer()) {
    TensorId tenId = initializer.name();
    if (!tensors.contains(tenId)) {
      throw error("no tensor " + tenId + " in tensors");
    }
    auto tensor = tensors.get(tenId);
    if (tensor->info != TensorInfo(initializer)) {
      throw error(
          "trying to reset weights using tensor with non matching tensor info");
    }
    tensor->tensorData()->resetData(initializer);
  }
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
  std::vector<Op *> fwdOps;
  for (auto op : getOpSchedule({})) {
    if (op->isFwdToBwd()) {
      fwdOps.push_back(op);
    }
  }

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

  // TODO (see T5099)
  // this should change. resnet-50 has way more memory for early layers.
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

// We should make a diagram explaining Willow recompute
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
  rcOp->setup();

  // yank down the priority of the new Op
  // (must be run as late as possible):
  rcOp->priority = std::numeric_limits<double>::lowest();

  // oriOp's outputs should not be consumed by grad op:
  for (auto &ind_ten : oriOp->output.tensorMap()) {
    Tensor *oriTen = ind_ten.second;
    Tensor *recTen = tensors.get(getRecompId(oriTen->id));
    for (auto &con : oriTen->consumers.getOps()) {
      if (con->getPhase() == Phase::BWD) {
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
    // remove the topo cons which might exist
    for (auto tensor_indices : op->input.indicesMap()) {
      Tensor *tensor = tensor_indices.first;
      tensor->consumers.removeTopoCons(op);
    }
    // and delete the Op
    ops.erase(op->id);
  }

  for (Tensor *tensor : tensorsToDelete) {
    tensors.remove(tensor->id);
  }
}

// TODO T5616
// this iteration is potentially dangerous.
// An Op in v_ops might
// be deleted by an earlier Op.
void Ir::applyPattern(const Pattern *pattern) {
  std::vector<Op *> v_ops;
  for (auto &id_op : ops) {
    v_ops.push_back(id_op.second.get());
  }
  for (auto op : v_ops) {
    // T5616: This op might have deleted at this point!
    if (pattern->matches(op)) {
      if (!pattern->touchesAnchored(op)) {
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

const std::vector<std::string> &VectorAndSet::v() const { return v_vals; }

void Ir::confirmConstIds() const {
  for (auto &tensorId : tensors.getConstIds().v()) {
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
    Op *op = growFromNode(node);

    // Not necessary to set the phase here (it will be done in
    // updateVertices). To check our logic though, we do this here
    // and then check that we agree in updateVertices()
    if (op) {
      op->setPhase(Phase::FWD);
    }
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
    logging::ir::warn("Unused ONNX tensor  " + id);
  }
}

void Tensors::addInit(TensorId name, const onnx::TensorProto *pt) {
  insert(name,
         std::unique_ptr<Tensor>(new Tensor(
             name,
             constIds.contains(name) ? TensorType::Const : TensorType::Variable,
             ir)));

  Tensor *init = get(name);
  init->info   = TensorInfo(*pt);
  init->setTensorData(*pt);
}

void Tensors::addStream(TensorId tenId, const TensorInfo &info) {
  insert(tenId,
         std::unique_ptr<Tensor>(new Tensor(tenId, TensorType::Stream, ir)));
  get(tenId)->info = info;
}

std::string reservedGradientPrefix() { return "d__"; }
std::string reservedRecomputePrefix() { return "r__"; }
std::vector<std::string> reservedPrefixes() {
  return {reservedGradientPrefix(), reservedRecomputePrefix()};
}

void Tensors::addActGrad(TensorId tenId) {
  insert(tenId,
         std::unique_ptr<Tensor>(new Tensor(tenId, TensorType::ActGrad, ir)));
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
      std::unique_ptr<Op>(new SumOp({"Sum", this, {}, getOnnxDomain()})));

  std::vector<TensorId> inputs;
  inputs.reserve(toSum.size());
  for (auto &tensor : toSum) {
    inputs.push_back(tensor->id);
  }
  TensorId gradientId = getGradId(target->id);
  std::vector<TensorId> outputs{gradientId};

  connectInputs(InputVecWrapper(inputs), opId);
  connectOutputs(OutputVecWrapper(outputs), opId);
  Op *op = ops[opId].get();
  op->setup();
  return op;
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
    gradOp->setup();

    // note, as the outputs of gradOp are edge-grad-tensors and not
    // edge-grads, we do not need to match them to non-grad tensors.
    gradOps.push_back(gradOp);
  }

  return gradOps;
}

int Op::getNonGradInIndex(int gradOpOutIndex) const {
  return gradOutToNonGradIn().at(gradOpOutIndex);
}

GradInOutMapper::GradInOutMapper(int iG, int iNG, GradOpInType t)
    : iGrad(iG), iNonGrad(iNG), type(t) {}

bool GradInOutMapper::operator==(const GradInOutMapper &rhs) const {
  return (type == rhs.type) && (iGrad == rhs.iGrad) &&
         (iNonGrad == rhs.iNonGrad);
}

const std::vector<GradInOutMapper> &Op::gradInputInfo() const {
  throw error("Op " + op_type() + " cannot get `grad input info'");
}

const std::map<int, int> &Op::gradOutToNonGradIn() const {
  throw error("Op " + op_type() + " cannot get `grad out to non grad in'");
}

bool Op::hasInplaceVariant(InIndex) const { return false; }

std::unique_ptr<Op> Op::getInplaceVariant(InIndex) {
  throw error("Op " + op_type() + "cannot get an inplace Op");
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

// design choice: we could have an "irHasChanged"
// flag which is set to true whenever the Ir changes,
// and then if irHasModeified is false, calls
// to this (and other) functions can do nothing.
// The cost of maintaining irHasModeified is non-trivial
// and would require runtime overhead, for now I'm not
// going to implement it.

void Ir::updateVertices() {

  // for all vertices (Ops and Tensors),
  // what phase is it is (FWD, BWD, LOSS) ?

  // for all vertices (Ops and Tensors),
  // is there a path to a BWD vertex? (YES, NO)

  // determine the phase of all Ops
  for (auto &id_op : ops) {
    Op *op = id_op.second.get();

    // There are several potential sources of information
    // that can be used to determine the Phase of an Op.
    // We gather all such sources, and confirm that they
    // are in agreement.
    std::vector<Phase> suggestions;

    // source 1 : if the op already has a
    // phase set, it should be the same.
    Phase prevPhase = op->getPhase();
    if (prevPhase != Phase::UNDEFINED) {
      suggestions.push_back(prevPhase);
    }

    // source 2 : if a producer of the op's
    // inputs is BWD, then it must be BWD too.
    for (auto tensor_indices : op->input.indicesMap()) {
      Tensor *inTensor = tensor_indices.first;
      if (inTensor->hasProducer()) {
        if (inTensor->getProducer()->getPhase() == Phase::BWD) {
          suggestions.push_back(Phase::BWD);
        }
      }
    }

    // source 3 : if any of the consumers of the
    // op's outputs is FWD, then it must be FWD too.
    for (auto tensor_indices : op->output.indicesMap()) {
      Tensor *outTensor = tensor_indices.first;
      for (Op *consumer : outTensor->consumers.getOps()) {
        if (consumer->getPhase() == Phase::FWD) {
          suggestions.push_back(Phase::FWD);
        }
      }
    }

    // source 4 : if the op is inherits from the
    // LossOp base class, then it is LOSS.
    if (op->isLossOp()) {
      suggestions.push_back(Phase::LOSS);
    }

    // source 5: if the output is "finalLoss", then it is LOSS
    if (op->output.hasIndex(0) && op->output.id(0) == getFinalLossId()) {
      suggestions.push_back(Phase::LOSS);
    }

    // source 6 : if an input or an output has a gradient
    // or recompute prefix, it is BWD
    std::vector<TensorId> insNouts;
    for (auto tensor_indices : op->output.indicesMap()) {
      insNouts.push_back(tensor_indices.first->id);
    }
    for (auto tensor_indices : op->input.indicesMap()) {
      insNouts.push_back(tensor_indices.first->id);
    }
    for (auto id : insNouts) {
      if ((id.find(reservedGradientPrefix()) != std::string::npos) ||
          (id.find(reservedRecomputePrefix()) != std::string::npos)) {
        suggestions.push_back(Phase::BWD);
      }
    }

    if (suggestions.size() == 0) {
      // no suggestions, it must a FWD (assuming all
      // tensors in backwards hace a gradient or
      // recompute prefix in them)
      op->setPhase(Phase::FWD);
    } else {
      for (auto phase : suggestions) {
        if (phase != suggestions[0]) {
          std::stringstream ss;
          ss << "failed to determine phase of " + op->str() +
                    ", which has suggested phases: ";
          std::vector<std::string> suggestions_s;
          for (auto &x : suggestions) {
            suggestions_s.push_back(phase_names().at(x));
          }
          appendSequence(ss, suggestions_s);
          throw error(ss.str());
        }
      }
      op->setPhase(suggestions[0]);
    }
  }

  // now we set the tensor phases,
  // as the phase of the earliest
  // consumer or producer
  for (auto &id_op : ops) {
    Op *op = id_op.second.get();
    std::vector<Tensor *> associated_tensors;

    for (auto tensor_indices : op->output.indicesMap()) {
      associated_tensors.push_back(tensor_indices.first);
    }

    for (auto tensor_indices : op->input.indicesMap()) {
      associated_tensors.push_back(tensor_indices.first);
    }

    for (Tensor *tensor : associated_tensors) {
      auto ass_ops = tensor->associatedOps();
      if (ass_ops.size() == 0) {
        throw error("Tensor " + tensor->id + " has no associated ops");
      }
      // starting with the latest of the phases (BWD),
      // update whenever an associated op is in an earlier phase.
      tensor->setPhase(Phase::BWD);
      for (auto ass_op : ass_ops) {
        // FWD is the earliest Phase, if any associated Op is
        // in the FWD phase then so is this tensor
        if (ass_op->getPhase() == Phase::FWD) {
          tensor->setPhase(Phase::FWD);
        } else if (ass_op->getPhase() == Phase::LOSS &&
                   tensor->getPhase() == Phase::BWD) {
          tensor->setPhase(Phase::LOSS);
        }
      }
    }
  }

  // All phases now set.

  // Now, set if there is a path to a bwd op.
  // we do this starting from scratch.

  std::set<Op *> s_op_front;
  std::vector<Op *> v_op_front;

  // initialising all Ops and Vertices to NO
  for (auto &id_op : ops) {
    Op *op = id_op.second.get();
    op->setPathToBwd(PathToBwd::NO);
    for (auto &tensor_indices : op->input.indicesMap()) {
      tensor_indices.first->setPathToBwd(PathToBwd::NO);
    }
    for (auto &tensor_indices : op->output.indicesMap()) {
      tensor_indices.first->setPathToBwd(PathToBwd::NO);
    }
  }

  // initialising all backward and loss
  // Ops to YES, adding them to the front
  for (auto &id_op : ops) {
    Op *op = id_op.second.get();
    if (op->getPhase() == Phase::BWD || op->getPhase() == Phase::LOSS) {
      op->setPathToBwd(PathToBwd::YES);
      v_op_front.push_back(op);
      s_op_front.insert(op);
    }
  }

  while (v_op_front.size() != 0) {
    Op *onPath = v_op_front.back();
    v_op_front.resize(v_op_front.size() - 1);
    s_op_front.erase(onPath);
    for (auto &tensor_indices : onPath->input.indicesMap()) {
      Tensor *tOnPath = tensor_indices.first;
      tOnPath->setPathToBwd(PathToBwd::YES);
      if (tOnPath->hasProducer()) {
        Op *producer = tOnPath->getProducer();
        producer->setPathToBwd(PathToBwd::YES);
        if (s_op_front.count(producer) == 0) {
          s_op_front.insert(producer);
          v_op_front.push_back(producer);
        }
      }
    }
  }
}

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

  // Note: if the finalLossOp has been optimised out,
  // this function cannot be used.
  auto found = ops.find(finalLossId);
  if (found == ops.end()) {
    throw error(
        "The final loss op does not exist, it may have been optimized away");
  }
  Op *finalLossOp = found->second.get();

  std::vector<Op *> opFront{finalLossOp};
  std::set<Op *> opsSeen{finalLossOp};
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
  std::vector<GradNonGradPair> opsToRegister = growLossGradients();

  while (!opsToRegister.empty()) {

    registerOpGrads(opsToRegister.back().grad, opsToRegister.back().nongrad);

    opsToRegister.resize(opsToRegister.size() - 1);

    for (auto &nongrad_egrads : tensor_grad_registry.popComplete()) {

      Tensor *nongrad                     = nongrad_egrads.first;
      const std::vector<Tensor *> &egrads = nongrad_egrads.second;
      // nongrad required below, as the name of the output of the
      // created op (sumOp) will be based off of it. Also, we
      // register the link between sumOp's output and nongrad
      Op *sumOp = growGradSumOp(nongrad, egrads);

      // Not necessary to set the phase here (it will be done in
      // updateVertices). To check our logic though, we do this here
      // and then check that we agree in updateVertices()
      sumOp->setPhase(Phase::BWD);

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
        opsToRegister.push_back({gradOp, op});
      }
    }
  }

  // add weight update ops (we are ignoring momentums for now)
  for (auto &varId : tensors.getIds(TensorType::Variable)) {
    growVarUpdateOp(varId);
  }
}

Op *Ir::growVarUpdateOp(TensorId varId) {

  OpId opId   = moveIntoIr(optimizer->createOp(varId, this));
  auto inputs = optimizer->getInputIds(varId);

  Op *op = ops[opId].get();
  connectInputs(InputVecWrapper(inputs), opId);

  // there are no outputs of var-op
  std::vector<TensorId> outputs{};
  connectOutputs(OutputVecWrapper(outputs), opId);
  op->setup();

  // Not necessary to set the phase here (it will be done in
  // updateVertices). To check our logic though, we do this here
  // and then check that we agree in updateVertices()
  op->setPhase(Phase::BWD);

  trainTargetOps.insert(op);

  return op;
}

void Ir::setVarUpdateCons() {

  for (auto &varId : tensors.getIds(TensorType::Variable)) {
    // impose the constraint that the varupdates
    // are the last consumers of the vars
    Tensor *var = tensors.get(varId);

    // we first determine which consumer
    // is the updater. It is the void Op
    Op *varupdater = nullptr;
    for (Op *consumer : var->consumers.getOps()) {
      if (consumer->output.n() == 0) {
        varupdater = consumer;
        break;
      }
    }
    if (varupdater == nullptr) {
      throw error("Failed to determine updater of " + var->id);
    }

    // set the constraints
    for (Op *consumer : var->consumers.getOps()) {
      if (consumer != varupdater) {
        var->consumers.insertTopoCon(consumer, varupdater);
      }
    }
  }
}

const std::map<int, Tensor *> &TensorIndexMap::tensorMap() const {
  return tensor_map;
}

std::map<int, TensorId> TensorIndexMap::tensorIdMap() const {
  std::map<int, TensorId> M;
  for (auto &index_tensor : tensorMap()) {
    M[index_tensor.first] = index_tensor.second->id;
  }
  return M;
}

Op *Ir::growFromNode(const Node &node) {

  // special case of CONSTANT Node, no Op is created
  if (getOpTypes().get(node.op_type(), node.domain()) == OpType::CONSTANT) {
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
  Op *fromNodeOp = ops[opId].get();
  // finally, set the output tensor info for the output
  // tensors, and any other Op specific class variables
  fromNodeOp->setup();
  return fromNodeOp;
}

void Ir::growFinalLoss() {
  std::vector<Op *> lossOps;
  // first, grow each of the individual losses from the user
  for (auto &loss : losses) {
    OpId opId = moveIntoIr(loss->getOp(this));
    connectInputs(*loss, opId);
    connectOutputs(*loss, opId);
    Op *lossOp = ops[opId].get();
    lossOps.push_back(lossOp);
    lossOp->setup();

    // Not necessary to set the phase here (it will be done in
    // updateVertices). To check our logic though, we do this here
    // and then check that we agree in updateVertices()
    lossOp->setPhase(Phase::LOSS);
  }

  // now growing the FINAL loss (sum of individual losses)
  OpId opId = moveIntoIr(
      std::unique_ptr<Op>(new SumOp({"Sum", this, {}, getOnnxDomain()})));

  std::vector<TensorId> inputs;
  inputs.reserve(lossOps.size());
  for (auto &op : lossOps) {
    inputs.push_back(op->output.tensor(0)->id);
  }
  std::vector<TensorId> outputs{getFinalLossId()};
  connectInputs(InputVecWrapper(inputs), opId);
  connectOutputs(OutputVecWrapper(outputs), opId);
  ops[opId]->setup();

  // Not necessary to set the phase here (it will be done in
  // updateVertices). To check our logic though, we do this here
  // and then check that we agree in updateVertices()
  ops[opId]->setPhase(Phase::LOSS);
  finalLossId = opId;
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
}

void Ir::connectInputsFromInputMapWrapper(const InputMapWrapper &in, OpId id) {
  connectInputs(in, id);
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

void Ir::append(std::stringstream &ss) {
  for (auto &op : getOpSchedule({})) {
    op->append(ss);
  }
}

Op::Op(const Op &op)
    : Vertex(op), priority(op.priority), opType(op.opType), pir(op.pir),
      id(pir->getAndIncrOpsCounter()), nAtts(op.nAtts), p_op_type(op.p_op_type),
      p_op_domain(op.p_op_domain) {
  // input, output: empty.
}

const std::string &Op::domain() { return *p_op_domain; }

const std::string &Op::op_type() const { return *p_op_type; }

OpConstructorBundle::OpConstructorBundle(std::string op_type_,
                                         Ir *pir_,
                                         Attributes atts_,
                                         std::string domain_)
    : op_type(op_type_), pir(pir_), atts(atts_), domain(domain_) {}

Op::Op(const OpConstructorBundle &b)
    : // opType (from string op_type)
      opType(getOpTypes().get(b.op_type, b.domain)),
      // the Ir
      pir(b.pir),
      // the id
      id(pir->getAndIncrOpsCounter()),
      // the Attributes
      nAtts(b.atts),
      // opType
      p_op_type(&getOpTypes().getName(opType)),
      // domain
      p_op_domain(&getOpTypes().getDomain(opType)) {}

Op::Op(const Node &node, Ir *pg)
    : // We set opType, looked up in a map from the string node.op_type()
      opType(getOpTypes().get(node.op_type(), node.domain())),
      // pointer to the Ir containing this node
      pir(pg), id(pir->getAndIncrOpsCounter()),
      // willow::Attributes constructed from contained of onnx::Attribute s
      nAtts(node.attribute()),
      // We set the pointer to the string version of opType, in another map
      p_op_type(&getOpTypes().getName(opType)),
      // And finally we strip off the domain of the Node
      p_op_domain(&getOpTypes().getDomain(opType)) {}

std::unique_ptr<Op> Ir::addOp(const Node &node) {
  using pOp = std::unique_ptr<Op>;
  switch (getOpTypes().get(node.op_type(), node.domain())) {
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
  case OpType::IDENTITY: {
    return pOp(new IdentityOp(node, this));
  }
  case OpType::NEGATE: {
    return pOp(new NegateOp(node, this));
  }
  case OpType::SOFTMAX: {
    return pOp(new SoftmaxOp(node, this));
  }
  case OpType::MAXPOOL: {
    return pOp(new MaxPoolOp(node, this));
  }

  case OpType::PAD: {
    return pOp(new PadOp(node, this));
  }
  case OpType::REDUCESUM: {
    return pOp(new ReduceSumOp(node, this));
  }
  case OpType::RELU: {
    return pOp(new ReluOp(node, this));
  }
  case OpType::SUBTRACT: {
    return pOp(new SubtractOp(node, this));
  }
  case OpType::SUM: {
    throw error("no constructor from node for Sum Op yet");
  }
  case OpType::SQUEEZE: {
    return pOp(new SqueezeOp(node, this));
  }
  case OpType::MATMUL: {
    return pOp(new MatMulOp(node, this));
  }
  case OpType::ADDARG0GRAD:
  case OpType::ADDARG1GRAD:
  case OpType::ADDBIASBIASGRAD:
  case OpType::ADDBIASDATAGRAD:
  case OpType::SQUEEZEGRAD:
  case OpType::REDUCESUMGRAD:
  case OpType::RELUGRAD:
  case OpType::AVERAGEPOOLGRAD:
  case OpType::CONVDATAGRAD:
  case OpType::CONVWEIGHTSGRAD:
  case OpType::NEGATEGRAD:
  case OpType::IDENTITYGRAD:
  case OpType::NLLGRAD:
  case OpType::L1GRAD:
  case OpType::MAXPOOLGRAD:
  case OpType::SOFTMAXGRAD:
  case OpType::SGDVARUPDATE:
  case OpType::CONSTSGDVARUPDATE:
  case OpType::SUBTRACTARG0GRAD:
  case OpType::SUBTRACTARG1GRAD:
  case OpType::MATMULLHSGRAD:
  case OpType::MATMULRHSGRAD:
    throw error("Gradient Ops not constructable from Node");

  case OpType::NLL:
  case OpType::L1:
    throw error("Loss Ops not constructable from Node");

  case OpType::ADDBIAS:
  case OpType::RELUINPLACE:
  case OpType::SOFTMAXGRADDIRECT:
    throw error("Non-ONNX Ops not constructable from Node");

  default: { throw error("No class for " + node.op_type()); }
  }
}

std::string Op::str() const {
  return std::to_string(id) + " (" + op_type() + ')';
}

std::vector<GradNonGradPair> Ir::growLossGradients() {
  std::vector<GradNonGradPair> pairs;
  for (auto &t_inds : getOp(finalLossId)->input.indicesMap()) {
    Tensor *t  = t_inds.first;
    Op *lossOp = t->getProducer();
    for (Op *gradOp : growGradOps(lossOp)) {
      pairs.push_back({gradOp, lossOp});
    }
  }
  return pairs;
}

OpId Ir::getFinalLossOpId() const { return finalLossId; }

Op *Ir::getOp(OpId opId) {
  auto found = ops.find(opId);
  if (found == ops.end()) {
    throw error("No Op `" + std::to_string(opId) + "'");
  }
  return found->second.get();
}

std::vector<Op *> Ir::getOpSchedule(const OpsBeforeKey &gCons) const {
  auto sorted = scheduler->getPartialOpSchedule(gCons);
  if (sorted.size() != ops.size()) {
    throw error("failure to sort topologically in getOpSchedule");
  }
  return sorted;
}

// Are the Ops with all the dependencies a DAG?
bool Ir::isSchedulable(const OpsBeforeKey &gCons) const {
  auto sorted = scheduler->getPartialOpSchedule(gCons);
  if (sorted.size() != ops.size()) {
    return false;
  }
  return true;
}

} // namespace willow
