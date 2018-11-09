#include <iostream>
#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/popx/addx.hpp>
#include <poponnx/popx/averagepoolx.hpp>
#include <poponnx/popx/convx.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/l1x.hpp>
#include <poponnx/popx/nllx.hpp>
#include <poponnx/popx/opx.hpp>
#include <poponnx/popx/padx.hpp>
#include <poponnx/popx/relux.hpp>
#include <poponnx/popx/softmaxx.hpp>
#include <poponnx/popx/squeezex.hpp>
#include <poponnx/popx/sumx.hpp>
#include <poponnx/popx/varupdatex.hpp>
#include <poponnx/pritask.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensordata.hpp>
#include <poponnx/util.hpp>

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#pragma clang diagnostic pop // stop ignoring warnings

namespace willow {
namespace popx {

void Devicex::weightsToHost(
    const std::map<TensorId, MutableVoidData> &onnxModelData) {
  std::cout << "Writing weights to host" << std::endl;
  // write weights from IPU to host stream memory points
  pEngine->run(PopPrograms::ProgramIndex::WEIGHTSTOHOST);
  std::cout << "Writing weights to ONNX ModelProto" << std::endl;
  // copy from the host stream memory points to the
  // addresses on onnxModelData
  for (auto initId : pir->tensors.getInitIds()) {
    auto found = onnxModelData.find(initId);
    if (found == onnxModelData.end()) {
      throw error("No TensorId " + initId + " in final host destination map");
    }
    MutableVoidData mv_data = found->second;
    hostStreamToHost(mv_data, initId);
  }
}

std::string Devicex::getConstTensorKey(const poplar::Type &type,
                                       const std::vector<size_t> &shape,
                                       double val) const {
  std::stringstream ss;
  ss << type << "___";
  appendSequence(ss, shape);
  ss << "___";
  ss << val;
  std::string key = ss.str();
  return key;
}

const poplar::Tensor &Devicex::getConst(const poplar::Type &type,
                                        const std::vector<size_t> &shape,
                                        double val) {
  std::string key = getConstTensorKey(type, shape, val);
  if (constTensors.find(key) == constTensors.end()) {
    std::cout << "Creating const tensor " << key << std::endl;
    constTensors[key] = graph().addConstant(type, shape, val);
  }
  return constTensors[key];
}

PopTensors::PopTensors(const Ir *ir_) : pir(ir_) {}

void PopTensors::insert(TensorId id, const poplar::Tensor &pt) {
  auto found = tensors_.find(id);
  if (found != tensors_.end()) {
    throw error("ILE: poplar::Tensor " + id + " already in map");
  }

  if (!pir->tensors.contains(id)) {
    throw error("ILE: no tensor named " + id +
                " in pir, is this a valid poplar::Tensor?");
  }

  // confirm shapes agree (up to squeezing out the extra 1s)
  auto expectedShape = pir->tensors.get(id)->info.shape_szt();

  if (squeeze(pt.shape()) != squeeze(expectedShape)) {
    std::stringstream ss;
    ss << "poplar::Tensor " << id << " of unexpected shape. "
       << "Poplar tensor shape: ";
    appendSequence(ss, pt.shape());
    ss << ". Expected (Ir) tensor shape: ";
    appendSequence(ss, expectedShape);
    throw error(ss.str());
  }

  tensors_[id] = pt;
}

const poplar::Tensor &PopTensors::get(TensorId id) const {
  auto found = tensors_.find(id);
  if (found == tensors_.end()) {
    throw error("no poplar::Tensor " + id);
  }
  return found->second;
}

poplar::program::Sequence &PopPrograms::weightsFromHost() {
  return seqs[ProgramIndex::WEIGHTSFROMHOST];
}

poplar::program::Sequence &PopPrograms::optimizerFromHost() {
  return seqs[ProgramIndex::OPTIMIZERFROMHOST];
}

poplar::program::Sequence &PopPrograms::step() {
  return seqs[ProgramIndex::STEP];
}

poplar::program::Sequence &PopPrograms::weightsToHost() {
  return seqs[ProgramIndex::WEIGHTSTOHOST];
}

std::vector<poplar::program::Program> PopPrograms::progs() {
  std::vector<poplar::program::Program> ps;
  for (auto &x : seqs) {
    ps.push_back(x);
  }
  return ps;
}

poplar::Graph &Devicex::graph() { return *pGraph; }

Devicex::Devicex(const Ir *pir) : willow::Device(pir), tensors(pir) {
  // TODO, better device handling (see T5105)
  popDevice = poplar::Device::createCPUDevice();
  if (!popDevice.attach()) {
    throw error("failed to attach to popDevice");
  }

  // TODO (see T5100) : if inference, forward should be INFERENCE_FWD
  fwdConvOptions.pass = enigma::Pass::TRAINING_FWD;
  bwdConvOptions.pass = enigma::Pass::TRAINING_BWD;
  wuConvOptions.pass  = enigma::Pass::TRAINING_WU;
  engineOptions.set({{"target.workerStackSizeInBytes", "0x200"}});
}

void Devicex::weightsFromHost() {
  std::cout << "Writing weights from host, " << std::flush;
  pEngine->run(PopPrograms::ProgramIndex::WEIGHTSFROMHOST);
  std::cout << "done." << std::endl;
}

void Devicex::optimizerFromHost() {
  std::cout << "Writing optimizer from host, " << std::flush;
  pEngine->run(PopPrograms::ProgramIndex::OPTIMIZERFROMHOST);
  std::cout << "done." << std::endl;
}

void Devicex::hostToHostStream(
    void *dst,                 // destination of copy (a step tensor)
    const void *src,           // source of copy
    const TensorInfo &dstInfo, // the info for dst
    const TensorInfo &srcInfo, // user provided info for src
    TensorId id // for clear error message, we need the id of the tensor
) {

  // confirm that the shapes of dst and src agree
  if (dstInfo.shape() != srcInfo.shape()) {
    std::stringstream ss;
    ss << "Shape discrepency for tensor " << id
       << ",\nStep tensor info (user) : ";
    srcInfo.append(ss);
    ss << "\nStep tensor info (expected) : ";
    dstInfo.append(ss);
    ss << ",\nBatches per step : " << pir->dataFlow.batchesPerStep() << '.';
    throw error(ss.str());
  }

  auto srcType = srcInfo.dataType();
  auto dstType = dstInfo.dataType();

  // check type compatibility
  if (srcType == dstType) {
    // copy the full step data from src to dst
    std::memcpy(dst, src, srcInfo.nbytes());
  }

  else if (srcType == TP::INT64 && dstType == TP::INT32) {
    auto dst_int32 = static_cast<int *>(dst);
    auto src_int64 = static_cast<const int64_t *>(src);
    for (auto i = 0; i < dstInfo.nelms(); ++i) {
      dst_int32[i] = static_cast<int>(src_int64[i]);
    }
  }
  // add more custom copies here. Design decision: don't
  // just blindly cast, if the user provides an int
  // tensor when a float tensor is expected they might
  // have made a mistake.

  else {
    std::stringstream ss;
    ss << "Type disrcepency for tensor " << id
       << ". User provided : " << srcInfo.data_type()
       << " and expected : " << dstInfo.data_type()
       << ". Consider a custom copy here (as memcpy cannot be used)";
    throw error(ss.str());
  }
}

// Copy from the host end of a d2h stream, to some final host memory.
// This is the step which follows a copy from device to host.
// poplar::Streams cannot write to an arbitrary dynamic address,
// they are connected to a fixed host address. This function copies
// from that fixed address to a dynamic address (mv_data).
void Devicex::hostStreamToHost(const MutableVoidData &mv_data, TensorId id) {

  // The host end of the poplar::Stream,
  // we will try to copy from here
  auto src = static_cast<const void *>(d2hBuffers.at(id).data());

  auto dst = mv_data.data;

  // size of the host end of the poplar stream.
  // It is a char vector, so this is in bytes.
  int64_t nbytes_src = d2hBuffers.at(id).size();

  // number of bytes of the destination.
  int64_t nbytes_dst = mv_data.info.nbytes();

  // We confirm that the sizes of src and dst are the same
  if (nbytes_src != nbytes_dst) {
    std::stringstream errms;
    errms << "sizes (in bytes) of src (" << nbytes_src << ") and dst ("
          << nbytes_dst << ") differ in hostStreamToHost";
    throw error(errms.str());
  }

  std::memcpy(dst, src, nbytes_src);
}

void Devicex::step(const StepIO &stepio) {
  std::cout << "Performing one step: " << std::endl;
  std::string prefix = "     ";
  std::cout << prefix << "Copying to h2d stream address(es) " << std::endl;
  for (Tensor *tensor : pir->dataStreamTensors()) {
    ConstVoidData stepin = stepio.in(tensor->id);

    // where to write to on host,
    auto dst = static_cast<void *>(h2dBuffers.at(tensor->id).data());
    // where to read from on host,
    auto src = stepin.data;

    // we calculate the TensorInfo for dst. It is almost tensor->info,
    // except it's for a full step tensor, so the first shape dimension
    // is larger by a factor batchesPerStep().
    auto stepDstShape = tensor->info.shape();
    stepDstShape[0] *= pir->dataFlow.batchesPerStep();
    TensorInfo dstInfo{tensor->info.dataType(), stepDstShape};

    // the info of the user provided src step tensor
    TensorInfo srcInfo = stepin.info;

    hostToHostStream(dst, src, dstInfo, srcInfo, tensor->id);
  }

  std::cout << prefix << "Running the step program " << std::endl;
  // TODO : this should be in a poplar for loop (see T5093)
  for (int i = 0; i < pir->dataFlow.batchesPerStep(); ++i) {
    pEngine->run(PopPrograms::ProgramIndex::STEP);
  }

  std::cout << prefix << "Copying from d2h stream address(es) " << std::endl;
  for (TensorId anchorId : pir->dataFlow.anchors()) {
    MutableVoidData stepout = stepio.out(anchorId);
    hostStreamToHost(stepout, anchorId);
  }
}

std::unique_ptr<Opx> Devicex::createOpx(Op *op) {
  switch (op->opType) {

  case OpType::ADD: {
    return std::unique_ptr<Opx>(new AddOpx(op, this));
  }

  case OpType::ADDGRAD: {
    return std::unique_ptr<Opx>(new AddGradOpx(op, this));
  }

  case OpType::AVERAGEPOOL: {
    return std::unique_ptr<Opx>(new AveragePoolOpx(op, this));
  }

  case OpType::AVERAGEPOOLGRAD: {
    return std::unique_ptr<Opx>(new AveragePoolGradOpx(op, this));
  }

  case OpType::CONSTANT: {
    throw error("ILE: No Opx for CONSTANT");
  }

  case OpType::CONV: {
    return std::unique_ptr<Opx>(new ConvOpx(op, this));
  }

  case OpType::CONVDATAGRAD: {
    return std::unique_ptr<Opx>(new ConvDataGradOpx(op, this));
  }

  case OpType::CONVWEIGHTSGRAD: {
    return std::unique_ptr<Opx>(new ConvWeightsGradOpx(op, this));
  }

  case OpType::CONSTSGDVARUPDATE: {
    return std::unique_ptr<Opx>(new ConstSGDVarUpdateOpx(op, this));
  }

  case OpType::L1: {
    return std::unique_ptr<Opx>(new L1Opx(op, this));
  }

  case OpType::L1GRAD: {
    return std::unique_ptr<Opx>(new L1GradOpx(op, this));
  }

  case OpType::SOFTMAX: {
    return std::unique_ptr<Opx>(new SoftmaxOpx(op, this));
  }

  case OpType::SOFTMAXGRAD: {
    return std::unique_ptr<Opx>(new SoftmaxGradOpx(op, this));
  }

  case OpType::SOFTMAXGRADDIRECT: {
    return std::unique_ptr<Opx>(new SoftmaxGradDirectOpx(op, this));
  }

  case OpType::NLL: {
    return std::unique_ptr<Opx>(new NllOpx(op, this));
  }

  case OpType::NLLGRAD: {
    return std::unique_ptr<Opx>(new NllGradOpx(op, this));
  }

  case OpType::PAD: {
    return std::unique_ptr<Opx>(new PadOpx(op, this));
  }

  case OpType::RELU: {
    return std::unique_ptr<Opx>(new ReluOpx(op, this));
  }

  case OpType::RELUGRAD: {
    return std::unique_ptr<Opx>(new ReluGradOpx(op, this));
  }

  case OpType::SGDVARUPDATE: {
    return std::unique_ptr<Opx>(new SGDVarUpdateOpx(op, this));
  }

  case OpType::SQUEEZE: {
    return std::unique_ptr<Opx>(new SqueezeOpx(op, this));
  }

  case OpType::SQUEEZEGRAD: {
    return std::unique_ptr<Opx>(new SqueezeGradOpx(op, this));
  }

  case OpType::SUM: {
    return std::unique_ptr<Opx>(new SumOpx(op, this));
  }

  default: { throw error("No get pop op for " + op->op_type()); }
  }
}

Opx *Devicex::getOpx(OpId id) { return opxs.at(id).get(); }

TaskId Devicex::taskWhichCreates(TensorId id) const {
  Tensor *tensor = pir->tensors.get(id);
  // streamed and init tensors are created with
  // tasks with names from initTensorTaskId
  // These tensors are recognisable as having no producing Op.
  if (tensor->hasProducer() == false) {
    return initTensorTaskId(id);
  }

  else {
    return opTaskId(tensor->getProducer());
  }
}

// Design decision : leave the option for a Tensor to be
// created based on complex global criteria open.
PriTask Devicex::initTensorTask(Tensor *tensor) {

  auto errorbase = [&tensor]() {
    std::stringstream ss;
    ss << "Failed to add tensor " << tensor->id << '.';
    tensor->consumers.append(ss);
    return ss.str();
  };

  // Do any of the consumers know how to create a poplar::Tensor?
  // If so, collect those that do, and the index at which consumed.
  // Note that an Opx may appear several times, with different
  // consumption indices.
  std::vector<OpxAndInIndex> candidates;
  for (Op *op : tensor->consumers.getOps()) {
    for (int index : op->input.indices(tensor)) {
      auto conOpId = op->id;
      Opx *opx     = getOpx(conOpId);
      if (opx->canCreateInput(index)) {
        candidates.push_back({index, opx});
      }
    }
  }

  if (candidates.size() > 1) {
    // check that all creators are in agreement on how
    // to create the poplar::Tensor. If they are, just keep
    // the first one.
    bool allEquivalent = true;
    auto cand0         = candidates[0];
    for (int i = 1; i < candidates.size(); ++i) {
      auto cand1 = candidates[i];
      if (!cand0.opx->createsEquiv(cand0.index, cand1.opx, cand1.index)) {
        allEquivalent = false;
        break;
      }
    }

    // they're all equivalent, select the first candidate as the creator
    if (allEquivalent) {
      candidates.resize(1);
    }
  }

  // a unique candidate creator will create the tensor
  if (candidates.size() == 1) {
    Opx *creator = candidates[0].opx;
    int inIndex  = candidates[0].index;
    auto f       = [this, creator, inIndex, tensor]() {
      std::cout << "Creating poplar::Tensor " << tensor->id << std::endl;
      tensors.insert(tensor->id, creator->createInput(inIndex));
    };
    // the inputs of creator which must have poplar::Tensors
    // before creator creates input tensor at index inIndex.
    std::vector<TaskId> deps;
    for (TensorId tenId : creator->mustExistBeforeCreate(inIndex)) {
      TaskId dep = taskWhichCreates(tenId);
      deps.push_back(dep);
    }

    // Discussion with David Norman suggests creating tensors as
    // late as possible gives better IPU memory use, so
    // giving this low priority.
    return {-1e6,
            initTensorTaskId(tensor->id), // the task name
            deps,
            f};
  }

  else if (candidates.size() > 1) {
    throw error(errorbase() + "\nConflicting creator candidates.");
  }

  else {

    auto f = [this, tensor]() {
      std::cout << "Creating " << tensor->id << " linearly. "
                << "WARNING :  "
                << "No creator candidates. We should perform a "
                << "depth search to find a candidate. " << std::endl;

      auto newTensor = graph().addVariable(
          popType(tensor->info), tensor->info.shape_szt(), tensor->id);
      poputil::mapTensorLinearly(graph(), newTensor);
      tensors.insert(tensor->id, newTensor);
    };

    return {1e6, initTensorTaskId(tensor->id), {}, f};
  }
}

PriTask Devicex::streamFromHostTask(Tensor *tensor) {
  auto f = [this, tensor]() {
    std::cout << "Creating host-to-device FIFO " << tensor->id << std::endl;
    fromHostStreams.emplace(tensor->id,
                            graph().addHostToDeviceFIFO(h2dId(tensor->id),
                                                        popType(tensor->info),
                                                        tensor->info.nelms()));
  };

  return {
      0,                                // priority unimportant
      streamFromHostTaskId(tensor->id), // name of this task
      {initTensorTaskId(tensor->id)},   // poplar::Tensor must exist
      f                                 // what to run when the task is executed
  };
}

PriTask Devicex::streamToHostTask(Tensor *tensor) {
  auto f = [this, tensor]() {
    std::cout << "Creating device-to-host FIFO " << tensor->id << std::endl;
    toHostStreams.emplace(tensor->id,
                          graph().addDeviceToHostFIFO(d2hId(tensor->id),
                                                      popType(tensor->info),
                                                      tensor->info.nelms()));
  };

  return {
      0,                              // priority unimportant
      streamToHostTaskId(tensor->id), // name of this task
      {taskWhichCreates(tensor->id)}, // poplar::Tensor must exist
      f                               // what to run when the task is executed
  };
}

PriTask Devicex::opTask(Op *op, double priority) {

  OpId id  = op->id;
  Opx *opx = getOpx(id);

  // although priority should guarantee that this
  // task is only run after inputs are all created,
  // we add a dependency to the input tensors, just
  // in case someone plays with the priorities.
  // Moreover, we must state the copy-from-host deps
  std::vector<TaskId> deps;
  for (auto t_inds : op->input.indicesMap()) {
    Tensor *tensor = t_inds.first;
    deps.push_back(taskWhichCreates(tensor->id));
    // if the tensor is streamed on, we must wait
    // 'til the Copy has happened
    if (tensor->tensorType() == TensorType::Stream) {
      deps.push_back(fromHostTaskId(tensor->id));
    }
  }

  auto f = [opx]() {
    std::cout << "Creating output tensors for " << opx->op_p->str()
              << std::endl;
    opx->grow();
  };

  return {priority, opTaskId(op), deps, f};
}

OpxAndInIndex::OpxAndInIndex(int conIndex_, Opx *opx_)
    : index(conIndex_), opx(opx_) {}

// go all the way to creating the engine and connecting streams
void Devicex::prepare() {

  pGraph.reset(new poplar::Graph(popDevice));
  popops::addCodelets(graph());
  poplin::addCodelets(graph());
  popnn::addCodelets(graph());

  // create a Opx for every Op
  for (Op *op : pir->getTopologicallySorted()) {
    opxs[op->id] = createOpx(op);
  }

  PriTasks tasks;

  // initializers :
  // 1) make tensor,
  // 2) make stream from host,
  // 3) create write prog,
  // 4) make stream to host,
  // 5) create read prog.
  for (auto id : pir->tensors.getInitIds()) {
    Tensor *tensor = pir->tensors.get(id);
    // 1
    tasks.add(initTensorTask(tensor));
    // 2
    tasks.add(streamFromHostTask(tensor));
    // 3
    tasks.add(fromHostTask(tensor, progs.weightsFromHost()));
    // 4
    tasks.add(streamToHostTask(tensor));
    // 5
    tasks.add(toHostTask(tensor, progs.weightsToHost()));
  }

  // stream-to-device tensors : 1)  make tensor 2) make stream
  for (auto id : pir->tensors.getIds(TensorType::Stream)) {
    Tensor *tensor = pir->tensors.get(id);
    // 1
    tasks.add(initTensorTask(tensor));
    // 2
    tasks.add(streamFromHostTask(tensor));
  }

  // stream-to-host tensors : 1) make streams 2) make copy programs
  // note that the order in which tasks are added does not matter,
  // they will be topologically sorted before running
  for (auto id : pir->dataFlow.anchors()) {
    // 1
    tasks.add(streamToHostTask(pir->tensors.get(id)));
    // 2
    tasks.add(toHostTask(pir->tensors.get(id), progs.step()));
  }

  // create Program to write optimizer tensors to device
  for (Tensor *tensor : pir->optimizerTensors()) {
    tasks.add(fromHostTask(tensor, progs.optimizerFromHost()));
  }

  // making the network!
  for (Tensor *tensor : pir->dataStreamTensors()) {
    tasks.add(fromHostTask(tensor, progs.step()));
  }
  std::vector<Op *> ops = pir->getTopologicallySorted();
  double priority       = 0.;
  for (int i = 0; i < ops.size(); ++i) {
    Op *op = ops[i];
    tasks.add(opTask(op, priority));
    priority -= 1.;
  }

  for (auto &task : tasks.getLinearised()) {
    task.f();
  }
  std::cout << "All tasks complete" << std::endl;
  std::cout << "Creating poplar::Engine" << std::endl;

  pEngine.reset(new poplar::Engine(graph(), progs.progs(), engineOptions));
  std::cout << "Engine has been created" << std::endl;

  pEngine->load(popDevice);
  std::cout << "Engine has loaded device" << std::endl;

  std::cout << "Connecting initializer streams" << std::endl;
  for (auto id : pir->tensors.getInitIds()) {
    Tensor *tensor = pir->tensors.get(id);
    pEngine->connectStream(h2dId(id), tensor->tensorData()->data());
  }

  std::cout << "Connecting optimizer streams" << std::endl;
  for (Tensor *tensor : pir->optimizerTensors()) {
    pEngine->connectStream(h2dId(tensor->id), tensor->tensorData()->data());
  }

  auto engineToStream =
      [this](char *data0, int64_t n_bytes, PopStreamId streamId) {
        // Poplar has no const void * version, disappointing
        auto addr0 = static_cast<void *>(data0);
        auto addr1 = static_cast<void *>(data0 + n_bytes);
        // connect the stream (circular buffer)
        pEngine->connectStream(streamId, addr0, addr1);
      };

  std::cout << "Creating host buffers for h2d streams, and connecting"
            << std::endl;
  for (Tensor *tensor : pir->dataStreamTensors()) {
    PopStreamId streamId = h2dId(tensor->id);
    // allocate host memory, where the poplar::Stream will read data from
    int64_t n_bytes = pir->dataFlow.batchesPerStep() * tensor->info.nbytes();
    h2dBuffers[tensor->id] = std::vector<char>(n_bytes);
    char *data0            = h2dBuffers[tensor->id].data();
    engineToStream(data0, n_bytes, streamId);
  }

  std::cout << "Creating host buffers for anchor d2h streams, connecting"
            << std::endl;
  for (TensorId anchorId : pir->dataFlow.anchors()) {
    PopStreamId streamId = d2hId(anchorId);
    Tensor *tensor       = pir->tensors.get(anchorId);
    int64_t batch_bytes  = tensor->info.nbytes();
    int64_t n_bytes;
    switch (pir->dataFlow.art()) {
    case (AnchorReturnType::FINAL): {
      n_bytes = batch_bytes;
      break;
    }
    case (AnchorReturnType::SUM): {
      n_bytes = batch_bytes / pir->dataFlow.samplesPerBatch();
      break;
    }
    case (AnchorReturnType::ALL): {
      n_bytes = batch_bytes * pir->dataFlow.batchesPerStep();
      break;
    }
    }
    d2hBuffers[anchorId] = std::vector<char>(n_bytes);
    char *data0          = d2hBuffers[tensor->id].data();
    engineToStream(data0, n_bytes, streamId);
  }

  std::cout << "Creating host buffers for weight d2h streams, connecting"
            << std::endl;

  for (auto initId : pir->tensors.getInitIds()) {
    PopStreamId streamId = d2hId(initId);
    Tensor *tensor       = pir->tensors.get(initId);
    int64_t n_bytes      = tensor->info.nbytes();
    d2hBuffers[initId]   = std::vector<char>(n_bytes);
    char *data0          = d2hBuffers[initId].data();
    engineToStream(data0, n_bytes, streamId);
  }
}

TaskId Devicex::streamFromHostTaskId(TensorId id) const {
  return "streamFromHostTask_" + id;
}

TaskId Devicex::streamToHostTaskId(TensorId id) const {
  return "streamToHostTask_" + id;
}

TaskId Devicex::fromHostTaskId(TensorId id) const {
  return "fromHostTask_" + id;
}

TaskId Devicex::toHostTaskId(TensorId id) const { return "toHostTask_" + id; }

TaskId Devicex::initTensorTaskId(TensorId id) const {
  return "initTensorTaskId_" + id;
}

TaskId Devicex::opTaskId(Op *op) const {
  return "fromOpTask_" + std::to_string(op->id) + '_' + op->op_type();
}

PopStreamId Devicex::h2dId(TensorId id) const { return "h2d_" + id; }

PopStreamId Devicex::d2hId(TensorId id) const { return "d2h_" + id; }

PriTask Devicex::fromHostTask(Tensor *tensor,
                              poplar::program::Sequence &sq) const {

  auto f = [&sq, tensor, this]() {
    std::cout << "Adding poplar::program::Copy from host " << tensor->id
              << std::endl;
    sq.add(poplar::program::Copy(fromHostStreams.at(tensor->id),
                                 tensors.get(tensor->id)));
  };

  return {-1e6, // writes to device: always as late as possible
          fromHostTaskId(tensor->id),
          {
              streamFromHostTaskId(tensor->id), // poplar::Stream created
              initTensorTaskId(tensor->id)      // poplar::Tensor created
          },
          f};
}

PriTask Devicex::toHostTask(Tensor *tensor,
                            poplar::program::Sequence &sq) const {

  auto f = [&sq, tensor, this]() {
    std::cout << "Adding poplar::program::Copy to host " << tensor->id
              << std::endl;
    sq.add(poplar::program::Copy(tensors.get(tensor->id),
                                 toHostStreams.at(tensor->id)));
  };

  return {+1e6, // writes to host: always as early as possible
          toHostTaskId(tensor->id),
          {
              // the dependencies:
              streamToHostTaskId(tensor->id), // poplar::Stream creation task,
              taskWhichCreates(tensor->id)    // poplar::Tensor creation task.
          },
          f};
}

poplar::Type popType(const TensorInfo &info) {
  switch (info.dataType()) {
  case TP::FLOAT: {
    return poplar::FLOAT;
  }
  case TP::INT32: {
    return poplar::INT;
  }

  case TP::UNDEFINED:
  case TP::UINT8:
  case TP::INT8:
  case TP::UINT16:
  case TP::INT16:
  case TP::INT64:
  case TP::STRING:
  case TP::BOOL:
  case TP::FLOAT16:
  case TP::BFLOAT16:
  case TP::DOUBLE:
  case TP::UINT32:
  case TP::UINT64:
  case TP::COMPLEX64:
  case TP::COMPLEX128:
  default:
    throw error("Is there a poplar type for " + info.data_type() + "?");
  }
}

// piggy-backing on TensorInfo's data_type()
// function to get a string of the DataType
poplar::Type popType(DataType type) { return popType(TensorInfo(type, {1})); }

} // namespace popx
} // namespace willow
