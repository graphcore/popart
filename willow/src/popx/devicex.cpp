#include <willow/error.hpp>
#include <willow/ir.hpp>
#include <willow/popx/addx.hpp>
#include <willow/popx/averagepoolx.hpp>
#include <willow/popx/convx.hpp>
#include <willow/popx/devicex.hpp>
#include <willow/popx/l1x.hpp>
#include <willow/popx/logsoftmaxx.hpp>
#include <willow/popx/nllx.hpp>
#include <willow/popx/opx.hpp>
#include <willow/popx/padx.hpp>
#include <willow/popx/relux.hpp>
#include <willow/popx/squeezex.hpp>
#include <willow/popx/sumx.hpp>
#include <willow/popx/varupdatex.hpp>
#include <willow/pritask.hpp>
#include <willow/stepio.hpp>
#include <willow/tensor.hpp>

namespace willow {
namespace popx {

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

Devicex::Devicex(const Ir *pir) : willow::Device(pir) {
  poplar::IPUModel ipumodel;
  popDevice = ipumodel.createDevice();
  if (!popDevice.attach()) {
    throw error("failed to attach to popDevice");
  }

  // TODO : if inference, forward should be INFERENCE_FWD
  fwdConvOptions.pass = enigma::Pass::TRAINING_FWD;
  bwdConvOptions.pass = enigma::Pass::TRAINING_BWD;
  wuConvOptions.pass  = enigma::Pass::TRAINING_WU;
  engineOptions.set({{"target.workerStackSizeInBytes", "0x200"}});
}

void Devicex::weightsFromHost() {
  std::cout << "writing weights from host, " << std::flush;
  pEngine->run(PopPrograms::ProgramIndex::WEIGHTSFROMHOST);
  std::cout << "done." << std::endl;
}

void Devicex::optimizerFromHost() {
  std::cout << "writing optimizer from host, " << std::flush;
  pEngine->run(PopPrograms::ProgramIndex::OPTIMIZERFROMHOST);
  std::cout << "done." << std::endl;
}

void Devicex::copyToStreamHostAddr(
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

void Devicex::step(const StepIO &stepio) {
  std::cout << "performing one step, " << std::flush;

  std::cout << "first copying from StepIO.in(...) to streams, " << std::flush;
  for (Tensor *tensor : pir->dataStreamTensors()) {
    StepInData stepin = stepio.in(tensor->id);

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

    copyToStreamHostAddr(dst, src, dstInfo, srcInfo, tensor->id);
  }

  std::cout << "now running the step program, " << std::flush;
  pEngine->run(PopPrograms::ProgramIndex::STEP);

  std::cout << "finally copying from streams to StepIO.out(). " << std::endl;
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

  case OpType::LOGSOFTMAX: {
    return std::unique_ptr<Opx>(new LogSoftmaxOpx(op, this));
  }

  case OpType::LOGSOFTMAXGRAD: {
    return std::unique_ptr<Opx>(new LogSoftmaxGradOpx(op, this));
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

TaskId Devicex::taskWhichCreates(TensorId) {
  throw error("must impl taskWhichCreates");
}

// Design decision : leave the option for a Tensor to be
// created based on complex global criteria open.
PriTask Devicex::popTensorTask(Tensor *tensor) {

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
      popTensors[tensor->id] = creator->createInput(inIndex);
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
            popTensorTaskId(tensor->id), // the task name
            deps,
            f};
  }

  else if (candidates.size() > 1) {
    throw error(errorbase() + "\nConflicting creator candidates.");
  }

  else {
    std::cout << "\nWARNING\n"
              << errorbase() << "\nNo creator candidates. We should perform a "
              << "depth search to find a candidate. Creating linearly. ";

    auto f = [this, tensor]() {
      std::cout << "Creating poplar::Tensor " << tensor->id
                << " by mapping linearly" << std::endl;
      auto newTensor = graph().addVariable(
          popType(tensor->info), tensor->info.shape_szt(), tensor->id);
      poputil::mapTensorLinearly(graph(), newTensor);
      popTensors[tensor->id] = newTensor;
    };

    return {1e6, popTensorTaskId(tensor->id), {}, f};
  }
}

PriTask Devicex::streamFromHostTask(Tensor *tensor) {
  auto f = [this, tensor]() {
    std::cout << "Creating Host to Device FIFO " << tensor->id << std::endl;

    fromHostStreams.emplace(tensor->id,
                            graph().addHostToDeviceFIFO(h2dId(tensor->id),
                                                        popType(tensor->info),
                                                        tensor->info.nelms()));
  };

  return {
      0,                                // priority unimportant
      streamFromHostTaskId(tensor->id), // name of this task
      {popTensorTaskId(tensor->id)}, // depends on poplar::Tensor being created
      f                              // what to run when the task is executed
  };
}

OpxAndInIndex::OpxAndInIndex(int conIndex_, Opx *opx_)
    : index(conIndex_), opx(opx_) {}

// go all the way to creating the engine
void Devicex::prepare() {

  PriTasks tasks;
  pGraph.reset(new poplar::Graph(popDevice));

  // create a Opx for every Op
  for (Op *op : pir->getTopologicallySorted()) {
    opxs[op->id] = createOpx(op);
  }

  // initializers : 1) make tensor, 2) make stream, 3) create write prog.
  for (auto id : pir->tensors.getInitIds()) {
    Tensor *tensor = pir->tensors.get(id);
    // 1
    tasks.add(popTensorTask(tensor));
    // 2
    tasks.add(streamFromHostTask(tensor));
    // 3
    tasks.add(fromHostTask(tensor, progs.weightsFromHost()));
  }

  // stream-to-device tensors : 1)  make tensor 2) make stream
  for (auto id : pir->tensors.getIds(TensorType::Stream)) {
    Tensor *tensor = pir->tensors.get(id);
    // 1
    tasks.add(popTensorTask(tensor));
    // 2
    tasks.add(streamFromHostTask(tensor));
  }

  // create prog. to write optimizer tensors to dev.
  for (Tensor *tensor : pir->optimizerTensors()) {
    tasks.add(fromHostTask(tensor, progs.optimizerFromHost()));
  }

  for (auto &task : tasks.getLinearised()) {
    task.f();
  }
  std::cout << "All tasks complete" << std::endl;

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

  std::cout << "Creating host buffers for data-from-host-streams and connecting"
            << std::endl;
  for (Tensor *tensor : pir->dataStreamTensors()) {
    PopStreamId streamId = h2dId(tensor->id);
    // allocate host memory, where the poplar::Stream will read data from
    int64_t n_bytes = pir->dataFlow.batchesPerStep() * tensor->info.nbytes();
    h2dBuffers[tensor->id] = std::vector<char>(n_bytes);
    char *data0            = h2dBuffers[tensor->id].data();
    // casting to void *, although in theory const void * would be enough
    // as the these addresses are only read from. However, Poplar
    // only has a void * version.
    auto addr0 = static_cast<void *>(data0);
    auto addr1 = static_cast<void *>(data0 + n_bytes);

    // connect the stream (circular buffer)
    pEngine->connectStream(streamId, addr0, addr1);
  }
}

TaskId Devicex::streamFromHostTaskId(TensorId id) {
  return "streamFromHostTask_" + id;
}

TaskId Devicex::fromHostTaskId(TensorId id) { return "fromHostTask_" + id; }

TaskId Devicex::popTensorTaskId(TensorId id) { return "popTensorTaskId_" + id; }

PopStreamId Devicex::h2dId(TensorId id) { return "h2d_" + id; }

PriTask Devicex::fromHostTask(Tensor *tensor, poplar::program::Sequence &sq) {

  auto f = [&sq, tensor, this]() {
    std::cout << "Adding poplar::program::Copy from host " << tensor->id
              << std::endl;
    sq.add(poplar::program::Copy(fromHostStreams.at(tensor->id),
                                 popTensors.at(tensor->id)));
  };

  return {-1e6, // writes to device: always as late as possible
          fromHostTaskId(tensor->id),
          {
              streamFromHostTaskId(tensor->id), // poplar::Stream created
              popTensorTaskId(tensor->id)       // poplar::Tensor created
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
  default: {
    throw error("Is there a poplar type for " + info.data_type() + "?");
  }
  }
}

} // namespace popx
} // namespace willow
