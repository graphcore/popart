#include <algorithm>
#include <array>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <poponnx/ces/constexpr.hpp>
#include <poponnx/error.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/intervals.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/logging.hpp>
#include <poponnx/op/loss.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/optionflags.hpp>
#include <poponnx/pbwrap.hpp>
#include <poponnx/scheduler.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/tensornames.hpp>
#include <poponnx/util.hpp>

// The transformations
#include <poponnx/transforms/interipucopy.hpp>
#include <poponnx/transforms/prune.hpp>
#include <poponnx/transforms/recompute.hpp>
#include <poponnx/transforms/virtual_graph_check.hpp>

// The layers required to construct the backwards pass
#include <poponnx/op/sum.hpp>
#include <poponnx/op/varupdate.hpp>

namespace poponnx {

std::vector<TensorId> Tensors::getAllTensorIds() const {
  std::vector<TensorId> allIds;
  allIds.reserve(M.size());
  for (auto &id_tensor : M) {
    allIds.push_back(id_tensor.first);
  }
  return allIds;
}

// remove all Tensors with no producer and no consumers
void Tensors::removeIsolated() {
  for (auto &id : getAllTensorIds()) {
    Tensor *tensor = M[id].get();
    if (tensor->hasProducer() == false && tensor->consumers.getTotal() == 0) {
      M.erase(id);
      logging::ir::info("Removing isolated Tensor {}", id);
    }
  }
}

GradNonGradPair::GradNonGradPair(Op *g_, Op *ng_) : grad(g_), nongrad(ng_) {}

GradNonGradPair::GradNonGradPair() : GradNonGradPair(nullptr, nullptr) {}

onnx::ModelProto Ir::getModel() const { return *onnxModel; }

std::vector<Tensor *> Ir::optimizerTensors() const {
  std::vector<Tensor *> optTensors;
  if (optimizer.get() != nullptr) {
    for (auto &id_info : optimizer->tensorInfos()) {
      optTensors.push_back(tensors.get(id_info.first));
    }
  }
  return optTensors;
}

// the rule followed : a Stream tensor which is not an
// optimizer tensor is a streamed data tensor
std::vector<Tensor *> Ir::dataStreamTensors() const {
  std::vector<Tensor *> dsTensors;
  std::map<TensorId, TensorInfo> optTensorInfo;
  if (optimizer != nullptr) {
    optTensorInfo = optimizer->tensorInfos();
  }
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

void Ir::eraseOp(OpId id) {
  auto found = ops.find(id);
  if (found == ops.end()) {
    throw error("ILE: no op " + std::to_string(id) + " to erase");
  }
  ops.erase(id);
}

void Ir::exportDot(const std::string dotfn) const {

  logging::ir::info("Writing dot file to {}", dotfn);
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
         << '.' << ' ' << n->opid;

    // Add the debug name if present
    if (!n->name().empty())
      strm << "(" << n->name() << ")";

    strm << "\"];\n";
    ++scheduleIndex;
    for (auto &ind_ten : n->input->tensorMap()) {
      strm << ind_ten.second->id << " -> n_" << n->id << ';' << '\n';
    }

    for (auto &ind_ten : n->output->tensorMap()) {
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

VectorAndSet::~VectorAndSet() = default;
Ir::~Ir()                     = default;

Tensor *Tensors::get(TensorId tenId) const {
  auto found = M.find(tenId);
  if (found == M.end()) {
    throw error("no tensor with id " + tenId);
  }
  return found->second.get();
}

// poponnx streams and prints are "impolite" (will not add new line at end)

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

void VectorAndSet::insert(const std::string &id) {
  if (m_vals.find(id) == m_vals.end()) {
    v_vals.push_back(id);
    m_vals.insert(id);
  }
}

void Ir::confirmNoReservedIds() const {

  auto &onnxGraph = onnxModel->graph();

  for (const auto &in_ : onnxGraph.input()) {
    confirmNonReservedId(in_.name());
  }

  for (const auto &out_ : onnxGraph.output()) {
    confirmNonReservedId(out_.name());
  }

  for (const auto &tenId : inputShapeInfo.getAllTensorIds()) {
    confirmNonReservedId(tenId);
  }
}

IrBundle::IrBundle(const onnx::ModelProto &modelProto_,
                   const InputShapeInfo &inputShapeInfo_,
                   const DataFlow &dataFlow_,
                   const std::vector<Loss *> &losses_,
                   const Optimizer *optimizer_,
                   const SessionOptions &userOptions_,
                   const Patterns &patterns_)
    : modelProto(modelProto_), inputShapeInfo(inputShapeInfo_),
      dataFlow(dataFlow_), losses(losses_), optimizer(optimizer_),
      userOptions(userOptions_), patterns(patterns_) {}

Ir::Ir() : tensors(*this), onnxModel(nullptr) {
  scheduler.reset(new Scheduler(this));
}

void Ir::setOnnxModel(const onnx::ModelProto &model) {
  onnxModel.reset(new onnx::ModelProto(model));
}

void Ir::setDataFlow(const DataFlow &df) { dataFlow = df; }

void Ir::setUserOptions(const SessionOptions &flags) { userOptions = flags; }
void Ir::setInputShapeInfo(const InputShapeInfo &info) {
  inputShapeInfo = info;
}
void Ir::setPatterns(const Patterns &p) { patterns = p; }

void Ir::removeIsolatedTensors() { tensors.removeIsolated(); }

void Ir::setExecutionMode(const ExecutionMode &mode) { executionMode = mode; }

void Ir::setLosses(const std::vector<Loss *> &_losses) {
  losses.clear();
  for (auto &l : _losses) {
    losses.emplace_back(l->clone());
  }
}

void Ir::setOptimizer(const Optimizer *o) {
  if (o) {
    optimizer = o->clone();

    for (auto &id_info : optimizer->tensorInfos()) {
      TensorId id     = id_info.first;
      TensorInfo info = id_info.second;
      tensors.addStream(id, info);
      optimizer->setTensorData(tensors.get(id));
    }
  }
}

void Ir::logIr() {
  std::stringstream ss2;
  append(ss2);
  logging::ir::info(ss2.str());
}

void Ir::prepare(const IrBundle &gb) {

  if (isPrepared) {
    throw error("Ir::prepare called more than once");
  }

  setDataFlow(gb.dataFlow);
  setUserOptions(gb.userOptions);
  setInputShapeInfo(gb.inputShapeInfo);
  setPatterns(gb.patterns);
  setOnnxModel(gb.modelProto);

  enableTransform(Recompute::id(), userOptions.enableRecomputation);

  // Require gb.losses.empty() => !gb.optimizer
  if (gb.losses.empty() && gb.optimizer) {
    throw error("An optimizer is set without any losses");
  }

  if (gb.optimizer) {
    setExecutionMode(ExecutionMode::TRAINING);
  } else if (gb.losses.empty()) {
    setExecutionMode(ExecutionMode::INFERENCE);
  } else {
    setExecutionMode(ExecutionMode::EVALUATION);
  }

  setLosses(gb.losses);

  confirmNoReservedIds();

  registerInputTensors();

  logging::ir::info("Patterns : {}", patterns);
  // todo : validate the selected patterns

  // construct the forward pass from ONNX,
  constructForwards();

  if (userOptions.exportDot) {
    exportDot(io::appendDirFn(userOptions.logDir, "fwd0.dot"));
  }

  applyPatterns(PatternPhase::PRETOPOCONS);

  if (canEvaluate()) {
    growFinalLoss();
    updateVertices();
    setNPathsToLoss();
  }

  // tensors with no producer and no consumers are removed
  // at this point. We may want something more subtle.
  removeIsolatedTensors();

  setOptimizer(gb.optimizer);

  if (canTrain()) {
    constructBackwards();
  }
  updateVertices();

  // confirm that all the anchor names provided
  // are indeed real tensor names. This is a check
  // that the user has not provided incorrect names.
  // We allow duplicates.
  validateAnchors();
  applyTransform(Prune::id());

  applyPatterns(PatternPhase::PRETOPOCONS);
  setNPathsToLoss();

  updateVertices();
  applyTransform(Recompute::id());
  updateVertices();

  // we now start applying topological constraints between
  // Ops directly. First, we ensure that the VarUpdate Ops
  // are the final consumers of the Variable tensors
  if (canTrain()) {
    setVarUpdateCons();
  }

  if (userOptions.exportDot) {
    exportDot(io::appendDirFn(userOptions.logDir, "fwdBwd0.dot"));
  }

  applyTransform(Prune::id());

  // Now, we apply the Patterns which can handle and create
  // topological constraints. Currently, this is only one
  // in-placing Pattern.
  applyPatterns(PatternPhase::WITHTOPOCONS);

  updateVertices();

  // Check to make sure that all or none have assigned to an ipu
  applyTransform(VirtualGraphCheck::id());

  // Add internal ops to copy tensors between ipu's as needed
  applyTransform(InterIpuCopy::id());

  updateVertices();

  isPrepared = true;

  if (userOptions.exportDot) {
    exportDot(io::appendDirFn(userOptions.logDir, "fwdBwd1.dot"));
  }

  logIr();
  // some checks, now that prepare is complete
  for (auto &id_op : ops) {
    if (id_op.second->opid == Onnx::CustomGradOperators::NllGrad) {
      logging::ir::warn("Computing gradient of the probabilities to Nll "
                        "might be less efficient than computing "
                        "pre-probability gradients directly with Pattern "
                        "SoftMaxGradDirect");
    }
  }
  // TODO : test described in T5690 will go here

  // end of checks
}

void Ir::resetWeights(const onnx::ModelProto &modelProto) {
  auto &onnxGraph = modelProto.graph();

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

void Ir::registerInputTensors() {
  auto &onnxGraph = onnxModel->graph();

  std::set<TensorId> onnxInitializers;
  for (const auto &initializer : onnxGraph.initializer()) {
    TensorId tenId = initializer.name();
    logging::info("Init tensor is {}", tenId);
    tensors.addVarInit(tenId, &initializer);
    onnxInitializers.emplace(tenId);
  }

  // onnx inputs which are not initializers are true inputs
  for (auto &valueInfo : onnxGraph.input()) {
    TensorId id = valueInfo.name();
    if (onnxInitializers.count(id) == 0) {
      if (valueInfo.has_type() && valueInfo.type().tensor_type().has_shape()) {
        tensors.addStream(id, TensorInfo(valueInfo.type()));
      } else {
        tensors.addStream(id, inputShapeInfo.get(id));
      }
    }
  }

  // other true inputs are for the loss calculation (class labels, etc)
  for (const auto &loss : losses) {
    for (const auto &tenId : loss->getStreamTensorNames()) {
      // another loss might have already registered this tensor
      if (!tensors.contains(tenId)) {
        tensors.addStream(tenId, inputShapeInfo.get(tenId));
      } else {
        Tensor *tensorAlreadyPresent = tensors.get(tenId);
        if (tensorAlreadyPresent->tensorType() != TensorType::Stream) {
          throw error("type mismatch for tensor " + tenId);
        }
      }
    }
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
    for (auto t_inds : op->input->indicesMap()) {
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

bool Ir::applyPattern(const Pattern *pattern) {
  bool result = false;

  std::vector<OpId> v_ops;
  v_ops.reserve(ops.size());

  for (auto &id_op : ops) {
    v_ops.push_back(id_op.first);
  }

  for (auto opId : v_ops) {
    auto itr = ops.find(opId);

    // If the op still exists
    if (itr != ops.end()) {
      Op *op = itr->second.get();

      if (pattern->matches(op)) {
        if (!pattern->touchesAnchored(op)) {
          logging::pattern::debug("Applying pattern {} to op {}",
                                  pattern->getPatternName(),
                                  op->str());
          result |= pattern->apply(op);
        }
      }
    }
  }

  return result;
}

void Ir::applyPatterns(PatternPhase phase) {
  bool keepRunning = true;

  std::vector<std::unique_ptr<Pattern>> patternList = patterns.getPatternList();

  while (keepRunning) {
    keepRunning = false;

    for (auto &pattern : patternList) {
      if (pattern->phase() == phase) {
        keepRunning |= applyPattern(pattern.get());
      }
    }
  }
}

void Ir::applyTransform(std::size_t transformId) {
  // Unless explictly set, a transform is enabled
  if (transformEnableMap.count(transformId) == 0 ||
      transformEnableMap.at(transformId)) {
    Transform::applyTransform(transformId, *this);
  }
}

void Ir::enableTransform(std::size_t transformId, bool enable) {
  transformEnableMap[transformId] = enable;
}

std::vector<TensorId> Tensors::getNoProducerIds() const {
  // the tensors which are not generated by an Op
  std::vector<TensorId> t0 = getIds(TensorType::Stream);
  std::vector<TensorId> t1 = getIds(TensorType::Const);
  std::vector<TensorId> t2 = getIds(TensorType::Variable);
  t0.insert(t0.end(), t1.begin(), t1.end());
  t0.insert(t0.end(), t2.begin(), t2.end());
  return t0;
}

std::vector<Op *> Ir::opsOfType(const OperatorIdentifier &opid) {
  std::vector<Op *> typedOps;
  for (auto &id_op : ops) {
    if (id_op.second->opid == opid) {
      typedOps.push_back(id_op.second.get());
    }
  }
  return typedOps;
}

bool Ir::isAnchored(TensorId tenId) { return dataFlow.isAnchored(tenId); }

const std::vector<std::string> &VectorAndSet::v() const { return v_vals; }

void Ir::constructForwards() {

  ConstExprUtil ce_util;
  // Select the relevant input tensors
  // see constexpr.hpp for details
  std::vector<TensorId> nonConstExprIn = tensors.getIds(TensorType::Stream);
  if (executionMode == ExecutionMode::TRAINING) {
    auto varIds = tensors.getIds(TensorType::Variable);
    nonConstExprIn.insert(nonConstExprIn.end(), varIds.begin(), varIds.end());
  }
  auto constExprClassifier =
      ce_util.getClassifier(onnxModel->graph(), nonConstExprIn);

  auto &onnxGraph = onnxModel->graph();
  auto &onnxNodes = onnxGraph.node();
  for (const auto &node : onnxNodes) {
    // if a node has multiple outputs, we assume it is not const-expr.
    // we may want to relax this assumption at a later point.
    if (node.output_size() == 1 &&
        constExprClassifier.isConstExprTensor(node.output(0))) {
      // the Node must be processed now, it is a ConstExprNode
      ce_util.processNode(node, this);
    } else {
      Op *op = growFromNode(node);
      // Not necessary to set the phase here (it will be done in
      // updateVertices). To check our logic though, we do this here
      // and then check that we agree in updateVertices()
      if (op) {
        op->setPhase(Phase::FWD);
      }
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

void Tensors::addConstInit(const TensorId &name, const onnx::TensorProto *pt) {
  addInit(name, pt, TensorType::Const);
  insertConstId(name);
}

void Tensors::addVarInit(const TensorId &name, const onnx::TensorProto *pt) {
  addInit(name, pt, TensorType::Variable);

  // A sanity check: if the tensor is fixed point, it is Const
  if (get(name)->info.getDataTypeInfo()->isFixedPoint()) {
    if (!constIds.contains(name)) {
      std::stringstream ss;
      ss << "A fixed-point Variable tensor `" << name
         << "'. Currently only floating-point tensors can be Variable. "
         << " Consider setting fixed-point tensors to be outputs of Constant "
         << "Ops, using (for example) "
         << "convertAllFixedPointInitializersToConstants().";
      throw error(ss.str());
    }
  }
}
void Tensors::addConstInit(const TensorId &name,
                           const TensorInfo &info,
                           const void *src) {
  insert(name,
         std::unique_ptr<Tensor>(new Tensor(name, TensorType::Const, ir)));

  insertConstId(name);

  Tensor *init = get(name);
  init->info   = info;
  init->setTensorData(info, src);
}

void Tensors::addInit(const TensorId &name,
                      const onnx::TensorProto *pt,
                      TensorType tt) {

  insert(name, std::unique_ptr<Tensor>(new Tensor(name, tt, ir)));
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

  OpId opId = moveIntoIr(OpManager::createOp(Onnx::Operators::Sum, this));

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
          m_inputs[indexGrad] = nonGradOp->input->tensor(indexFwd)->id;
          break;
        }

        //  (2) the OUTPUT at index 'indexFwd' of nonGradOp
        case GradOpInType::OUT: {
          m_inputs[indexGrad] = nonGradOp->output->tensor(indexFwd)->id;
          break;
        }

        //  (3) the GRADIENT of the OUTPUT
        //      at index 'indexFwd' of nonGradOp.
        case GradOpInType::GRADOUT: {
          if (!nonGradOp->output->hasIndex(indexFwd)) {
            std::stringstream ss;
            ss << "No gradient for non-grad-op " << nonGradOp->str()
               << " at index " << indexFwd << '.'
               << " Could it be that the path along that index "
               << "did not lead to final loss, "
               << "in which case the gradient is zero?";
            throw error(ss.str());
          }
          m_inputs[indexGrad] =
              getGradId(nonGradOp->output->tensor(indexFwd)->id);
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
        TensorId inId  = nonGradOp->input->tensor(nonGradIn)->id;
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
  // nonGrad->output->n(), but maybe not.
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
    for (auto tensor_indices : op->input->indicesMap()) {
      Tensor *inTensor = tensor_indices.first;
      if (inTensor->hasProducer()) {
        if (inTensor->getProducer()->getPhase() == Phase::BWD) {
          suggestions.push_back(Phase::BWD);
        }
      }
    }

    // source 3 : if any of the consumers of the
    // op's outputs is FWD, then it must be FWD too.
    for (auto tensor_indices : op->output->indicesMap()) {
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
    if (op->output->hasIndex(0) && op->output->id(0) == getFinalLossId()) {
      suggestions.push_back(Phase::LOSS);
    }

    // source 6 : if an input or an output has a gradient
    // or recompute prefix, it is BWD
    std::vector<TensorId> insNouts;
    for (auto tensor_indices : op->output->indicesMap()) {
      insNouts.push_back(tensor_indices.first->id);
    }
    for (auto tensor_indices : op->input->indicesMap()) {
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

    for (auto tensor_indices : op->output->indicesMap()) {
      associated_tensors.push_back(tensor_indices.first);
    }

    for (auto tensor_indices : op->input->indicesMap()) {
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
    for (auto &tensor_indices : op->input->indicesMap()) {
      tensor_indices.first->setPathToBwd(PathToBwd::NO);
    }
    for (auto &tensor_indices : op->output->indicesMap()) {
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
    for (auto &tensor_indices : onPath->input->indicesMap()) {
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
  auto found = ops.find(finalLossId);
  if (found == ops.end()) {
    // There will be no losses at all for an inference
    return;
  }
  Op *finalLossOp = found->second.get();

  // initialize number of paths for
  // all Ops and Tensors to loss to be zero
  for (auto &id_op : ops) {
    Op *op = id_op.second.get();
    op->setNPathsToLossToZero();
    for (auto t_inds : op->input->indicesMap()) {
      t_inds.first->setNPathsToLossToZero();
    }
    for (auto t_inds : op->output->indicesMap()) {
      t_inds.first->setNPathsToLossToZero();
    }
  }

  std::vector<Op *> opFront{finalLossOp};
  std::set<Op *> opsSeen{finalLossOp};
  std::set<Tensor *> tensorsSeen{};
  while (opFront.size() != 0) {
    Op *op = opFront.back();
    opFront.resize(opFront.size() - 1);
    for (auto &ind_ten : op->input->tensorMap()) {
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
    for (auto &index_tensor : gradOp->output->tensorMap()) {
      int opOutInd     = index_tensor.first;
      Tensor *partGrad = index_tensor.second;
      // what input index of nonGradOp does the
      // edge-gradient correspond to?
      int nonGradInInd      = gradOp->getNonGradInIndex(opOutInd);
      Tensor *nonGradTensor = nonGradOp->input->tensor(nonGradInInd);
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
      int index = producer->output->indices(nonGrad).at(0);
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
        registerTensorGrad(sumOp->output->tensor(0));
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
      case TensorType::Const: {
        break;
      }
      case TensorType::Momentum:
      case TensorType::Unknown:
      case TensorType::N:
        throw error("can't currently register gradient of " +
                    nongrad->tensor_type() + " tensor, " + nongrad->str());

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

  // A sanity check that the Tensor is not fixed point type
  if (tensors.get(varId)->info.getDataTypeInfo()->isFixedPoint()) {
    throw error("Currently only floating point variable tensors are updatable");
  }

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
      if (consumer->output->n() == 0) {
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

void Tensors::insertConstId(const std::string &id) { constIds.insert(id); }

Op *Ir::growFromNode(const Node &node) {

  OpId opId = moveIntoIr(addOp(node));

  connectInputs(node, opId);
  connectOutputs(node, opId);

  // finally, set the output tensor info for the output
  // tensors, and any other Op specific class variables
  Op *fromNodeOp = ops[opId].get();
  fromNodeOp->setup();
  return fromNodeOp;
}

void Ir::growFinalLoss() {
  // There may be no losses (in inference especially)
  if (losses.size() == 0) {
    return;
  }

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
  OpId opId = moveIntoIr(OpManager::createOp(Onnx::Operators::Sum, this));

  std::vector<TensorId> inputs;
  inputs.reserve(lossOps.size());
  for (auto &op : lossOps) {
    inputs.push_back(op->output->tensor(0)->id);
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
        // default: connects tensor <-> op, in both directions.
        // Note that this is a virtual function, and so specific Ops
        // may to do something different to the default here.
        op->connectInTensor(inIndex, inName);
      }
    }
  }
}

void Ir::connectInputsFromInputMapWrapper(const InputMapWrapper &in, OpId id) {
  connectInputs(in, id);
}

void Ir::connectOutputsFromOutputMapWrapper(const OutputMapWrapper &out,
                                            OpId id) {
  connectOutputs(out, id);
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

Tensor *Op::inTensor(InIndex index) { return input->tensor(index); }
const Tensor *Op::inTensor(InIndex index) const { return input->tensor(index); }
Tensor *Op::outTensor(OutIndex index) { return output->tensor(index); }
const Tensor *Op::outTensor(OutIndex index) const {
  return output->tensor(index);
}

const Shape &Op::inShape(InIndex index) const {
  return inTensor(index)->info.shape();
}

const Shape &Op::outShape(OutIndex index) const {
  return outTensor(index)->info.shape();
}

int Op::inRank(InIndex index) { return inTensor(index)->info.rank(); }

int Op::outRank(InIndex index) { return outTensor(index)->info.rank(); }

std::vector<GradNonGradPair> Ir::growLossGradients() {
  std::vector<GradNonGradPair> pairs;
  if (ops.find(finalLossId) != ops.end()) {
    for (auto &t_inds : getOp(finalLossId)->input->indicesMap()) {
      Tensor *t  = t_inds.first;
      Op *lossOp = t->getProducer();
      for (Op *gradOp : growGradOps(lossOp)) {
        pairs.push_back({gradOp, lossOp});
      }
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

Ir::ExecutionMode Ir::getExecutionMode() const { return executionMode; }

bool Ir::canInfer() const {
  return getExecutionMode() == ExecutionMode::INFERENCE || canEvaluate();
}

bool Ir::canEvaluate() const {
  return getExecutionMode() == ExecutionMode::EVALUATION || canTrain();
}

bool Ir::canTrain() const {
  return getExecutionMode() == ExecutionMode::TRAINING;
}

bool Ir::containsInitialisers() {
  return !(onnxModel->graph().initializer().empty());
}

} // namespace poponnx
