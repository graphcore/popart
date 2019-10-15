#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <set>

#include <boost/range/algorithm/find.hpp>

#include <poplar/CSRFunctions.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>
#include <poputil/exceptions.hpp>
#include <popart/devicemanager.hpp>
#include <popart/error.hpp>
#include <popart/filereader.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/op.hpp>
#include <popart/op/call.hpp>
#include <popart/op/if.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/restore.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/devicexmanager.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/poplaroptionsx.hpp>
#include <popart/pritask.hpp>
#include <popart/recompute.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensors.hpp>
#include <popart/tojson.hpp>
#include <popart/topocons.hpp>

#include <popart/op/sgd1acclupdate.hpp>
#include <popart/op/sgd1varupdate.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/popx/op/ipucopyx.hpp>
#include <popart/tensornames.hpp>

namespace popart {
namespace popx {

class devicex_memory_allocation_err : public popart::memory_allocation_err {

  const poplar::graph_memory_allocation_error exception;
  const poplar::OptionFlags reportOptions;

public:
  devicex_memory_allocation_err(const devicex_memory_allocation_err &rhs)
      : popart::memory_allocation_err(rhs.what()),
        exception(std::move(rhs.exception)), reportOptions(rhs.reportOptions) {}

  devicex_memory_allocation_err(const poplar::graph_memory_allocation_error &e,
                                const poplar::OptionFlags &_reportOptions)
      : popart::memory_allocation_err(e.what()), exception(std::move(e)),
        reportOptions(_reportOptions) {}

  std::unique_ptr<memory_allocation_err> clone() const {
    return std::make_unique<devicex_memory_allocation_err>(*this);
  }

  std::string getSummaryReport() const {

    if (exception.graphProfile.type() == poplar::ProfileValue::Type::MAP &&
        exception.graphProfile.size() != 0) {

      std::stringstream ss;
      poplar::printProfileSummary(
          ss, exception.graphProfile, {}, reportOptions);
      return ss.str();
    } else {
      throw error("Need to set the 'debug.allowOutOfMemory' engine option to "
                  "true to get the graph report");
    }
  }

  std::string getGraphReport(bool use_cbor) const {

    if (exception.graphProfile.type() == poplar::ProfileValue::Type::MAP &&
        exception.graphProfile.size() != 0) {

      std::stringstream ss;
      if (use_cbor) {
        serializeToCBOR(ss, exception.graphProfile);
      } else {
        serializeToJSON(ss, exception.graphProfile);
      }
      return ss.str();
    } else {
      throw error("Need to set the 'debug.allowOutOfMemory' engine option to "
                  "true to get the graph report");
    }
  }
};

Devicex::Datastream::Datastream(Tensor *t, PopStreamId s)
    : tensor(t), streamId(s), io(nullptr) {}

TensorId Devicex::Datastream::getTensorId() { return tensor->id; }

Devicex::InputDatastream::InputDatastream(Tensor *t, PopStreamId s)
    : Datastream(t, s) {}

Devicex::PrefetchCallback::PrefetchCallback(
    std::shared_ptr<InputDatastream> ds_)
    : ds(ds_) {}

poplar::StreamCallback::Result
Devicex::PrefetchCallback::prefetch(void *dest) noexcept {
  if (ds->readPrefetch(dest)) {
    return poplar::StreamCallback::Result::Success;
  } else {
    return poplar::StreamCallback::Result::NotAvailable;
  }
}

void Devicex::PrefetchCallback::fetch(void *dest) noexcept { ds->read(dest); }

void Devicex::PrefetchCallback::complete() noexcept { ds->readComplete(); }

void Devicex::InputDatastream::read(void *ptr) {

  if (io) {

    ConstVoidData data = io->in(getTensorId(), tensor->info.nelms(), false);

    const void *srcAddr = data.data;
    void *dstAddr       = ptr;

    auto srcInfo = data.info;
    auto dstInfo = tensor->info;

    // check the shape

    // Not sure how best to match the shape as the shape of the input
    // does not match the shape of the data.info. Infact that is a bit
    // wrong now.

    // check the type
    if (srcInfo.dataType() == dstInfo.dataType()) {
      memcpy(dstAddr, srcAddr, tensor->info.nbytes());
    } else if (srcInfo.dataType() == DataType::INT64 &&
               dstInfo.dataType() == DataType::INT32) {

      static bool loggingWarning = false;
      if (loggingWarning == false) {
        logging::devicex::warn(
            "Copying (host) tensor {} from INT64 to INT32. Will only warn once",
            getTensorId());
        loggingWarning = true;
      }
      int32_t *dest      = static_cast<int32_t *>(dstAddr);
      const int64_t *src = static_cast<const int64_t *>(srcAddr);
      for (int i = 0; i < tensor->info.nelms(); ++i) {
        dest[i] = static_cast<int32_t>(src[i]);
      }
    } else {
      std::stringstream ss;
      ss << "Type discrepency for tensor " << getTensorId()
         << ". User provided : " << srcInfo.data_type()
         << " and expected : " << dstInfo.data_type()
         << ". Consider a custom copy here (as memcpy cannot be used)";
      throw error(ss.str());
    }

  } else {
    logging::devicex::warn(
        "No stepio set for tensor {} stream {}", getTensorId(), streamId);
  }
}

bool Devicex::InputDatastream::readPrefetch(void *ptr) {

  if (io) {

    ConstVoidData data = io->in(getTensorId(), tensor->info.nelms(), true);

    if (data.data == nullptr) {
      logging::devicex::info("readPrefetch returning false");
      return false;
    } else {

      const void *srcAddr = data.data;
      void *dstAddr       = ptr;

      auto srcInfo = data.info;
      auto dstInfo = tensor->info;

      // check the shape

      // Not sure how best to match the shape as the shape of the input
      // does not match the shape of the data.info. Infact that is a bit
      // wrong now.

      // check the type
      if (srcInfo.dataType() == dstInfo.dataType()) {
        memcpy(dstAddr, srcAddr, tensor->info.nbytes());
      } else if (srcInfo.dataType() == DataType::INT64 &&
                 dstInfo.dataType() == DataType::INT32) {

        static bool loggingWarning = false;
        if (loggingWarning == false) {
          logging::devicex::warn("Copying (host) tensor {} from INT64 to "
                                 "INT32. Will only warn once",
                                 getTensorId());
          loggingWarning = true;
        }
        int32_t *dest      = static_cast<int32_t *>(dstAddr);
        const int64_t *src = static_cast<const int64_t *>(srcAddr);
        for (int i = 0; i < tensor->info.nelms(); ++i) {
          dest[i] = static_cast<int32_t>(src[i]);
        }
      } else {
        std::stringstream ss;
        ss << "Type discrepency for tensor " << getTensorId()
           << ". User provided : " << srcInfo.data_type()
           << " and expected : " << dstInfo.data_type()
           << ". Consider a custom copy here (as memcpy cannot be used)";
        throw error(ss.str());
      }

      return true;
    }

  } else {
    logging::devicex::warn(
        "No stepio set for tensor {} stream {}", getTensorId(), streamId);
    return false;
  }
}

void Devicex::InputDatastream::readComplete() {
  if (io) {
    io->inComplete(getTensorId(), tensor->info.nelms());
  }
}

Devicex::OutputDatastream::OutputDatastream(Tensor *t, PopStreamId s)
    : Datastream(t, s) {}

void Devicex::OutputDatastream::write(void *ptr) {

  if (io) {
    MutableVoidData data = io->out(getTensorId(), tensor->info.nelms());
    memcpy(data.data, ptr, tensor->info.nbytes());
    io->outComplete(getTensorId());
  } else {
    logging::devicex::warn(
        "No stepio set for tensor {} stream {}", getTensorId(), streamId);
  }
}

std::map<Op *, int> Devicex::getMainGraphOpSeriesNums() const {
  std::map<Op *, int> nums;
  int num = 0;
  for (auto op : mainGraphOpRegistery) {
    auto found = nums.find(op);
    if (found == nums.end()) {
      nums.insert({op, num});
      ++num;
    }
  }
  return nums;
}

std::string Devicex::getMainGraphOpString() const {

  std::stringstream ss;
  auto seriesNums = getMainGraphOpSeriesNums();
  std::set<Op *> seen;
  for (auto op : mainGraphOpRegistery) {
    auto found = seen.count(op);
    seen.insert(op);
    std::string type;
    if (op->scheduledPreLoss == ScheduledPreLoss::Yes) {
      type = "preL";
    } else if (found != 0) {
      type = "re.1";
    } else if (op->settings.recomputeType == RecomputeType::RECOMPUTE) {
      type = "re.0";
    } else {
      std::ostringstream ss2;
      ss2 << ((op->toLoss == PathToLoss::Yes) ? "tY" : "tN");
      ss2 << ((op->fromLoss == PathFromLoss::Yes) ? "fY" : "fN");
      type = ss2.str();
    }
    ss << type << "  " << seriesNums[op] << "  " << op->str() << '\n';
  }
  return ss.str();
}

std::map<Op *, int> Devicex::getMainGraphOpCounts() const {
  std::map<Op *, int> counts;
  for (auto op : mainGraphOpRegistery) {
    auto found = counts.find(op);
    if (found == counts.end()) {
      counts.insert({op, 1});
    } else {
      ++found->second;
    }
  }
  return counts;
}

void Devicex::run(PopPrograms::ProgramIndex ind) {
  if (isEngineLoaded() == false) {
    logging::devicex::debug("Reloading engine & connecting streams");
    loadEngineAndConnectStreams();
  }
  pEngine->run(ind);
}

void Devicex::weightsToHost() {

  if (useSyntheticData() == false) {
    logging::devicex::debug("Writing weights to host");
    pEngine->disableExecutionProfiling();
    run(PopPrograms::ProgramIndex::WEIGHTSTOHOST);
    logging::devicex::debug("Writing weights to host complete.");
  }
}

void Devicex::readWeights(const IWeightsIO &weights) {

  // Better to do this the otherway round
  for (auto id : ir().getTensorIds(TensorType::Variable)) {
    if (weights.contains(id)) {
      logging::devicex::debug("Reading weights (host stream -> host) for {}",
                              id);
      MutableVoidData stepout = weights.weight(id);
      hostStreamToHost(stepout, id);
    } else {
      logging::devicex::debug(
          "Not reading weights (host stream -> host) for {}", id);
    }
  }
}

void Devicex::writeWeights(const IWeightsIO &weights) {
  // Better to do this the otherway round
  // Also : should check that all weights have valid names
  for (auto id : ir().getTensorIds(TensorType::Variable)) {
    if (weights.contains(id)) {
      auto tensor             = ir().getTensor(id);
      MutableVoidData stepout = weights.weight(id);
      tensor->tensorData()->resetData(stepout.info, stepout.data);
    }
  }
}

void Devicex::weightsToHost(
    const std::map<TensorId, MutableVoidData> &onnxModelData) {

  if (useSyntheticData() == false) {
    logging::devicex::debug("Writing weights to host");
    // write weights from IPU to host stream memory points

    pEngine->disableExecutionProfiling();
    run(PopPrograms::ProgramIndex::WEIGHTSTOHOST);

    logging::devicex::debug("Writing weights to ONNX ModelProto");
    // copy from the host stream memory points to the
    // addresses on onnxModelData
    for (auto id : ir().getTensorIds(TensorType::Variable)) {
      auto found = onnxModelData.find(id);
      if (found == onnxModelData.end()) {
        std::ostringstream oss;
        oss << "No TensorId " << id
            << " in final host destination map. The TensorIds are [ ";
        for (auto x : onnxModelData) {
          oss << x.first << ' ';
        }
        oss << ']';
        throw error(oss.str());
      }
      MutableVoidData mv_data = found->second;
      hostStreamToHost(mv_data, id);
    }
  }
}

poplar::Tensor Devicex::getConst(poplar::Graph &graph,
                                 const poplar::Type &type,
                                 const std::vector<size_t> &shape,
                                 double val,
                                 const std::string &name) {
  static unsigned tileCounter = 0;

  auto tensor = graph.addConstant(type, shape, val, name);
  auto tile   = tileCounter % graph.getTarget().getTilesPerIPU();
  tileCounter++;
  graph.setTileMapping(tensor, tile);
  return tensor;
}

PopTensors::PopTensors(const Ir &ir_) : ir(ir_) {}

void PopTensors::insert(TensorId id, const poplar::Tensor &pt) {
  auto found = tensors_.find(id);
  if (found != tensors_.end()) {
    throw error("ILE: poplar::Tensor " + id + " already in map");
  }

  if (!ir.containsTensor(id)) {
    throw error("ILE: no tensor named " + id +
                " in ir, is this a valid poplar::Tensor?");
  }

  // confirm shapes agree (up to squeezing out the extra 1s)
  auto irTensor = ir.getTensor(id);
  auto shape    = pt.shape();

  if (pt.shape() != irTensor->info.shape_szt()) {

    // squeeze out extra 1s
    while (!shape.empty() && shape[0] == 1) {
      shape.erase(shape.begin());
    }

    if (shape != irTensor->info.shape_szt()) {
      std::stringstream ss;
      ss << "poplar::Tensor " << id << " of unexpected shape. "
         << "Poplar tensor shape: " << pt.shape() << "."
         << "Expected (Ir) tensor shape: " << irTensor->info.shape_szt() << "."
         << "This for tensor " << irTensor->str();
      throw error(ss.str());
    }
  }

  // confirm types agree
  auto expectedType = popType(ir.getTensor(id)->info);
  if (pt.elementType() != expectedType) {
    std::stringstream ss;
    ss << "poplar::Tensor " << id << " of unexpected Type. "
       << "Poplar tensor type : " << pt.elementType();
    ss << ". Expected (Ir) tensor type : " << expectedType;
    ss << ". This for tensor " << irTensor->str();
    throw error(ss.str());
  }

  tensors_[id] = pt;
}

bool PopTensors::contains(TensorId id) const {
  return tensors_.find(id) != tensors_.end();
}

const poplar::Tensor &PopTensors::get(TensorId id) const {
  auto found = tensors_.find(id);
  if (found == tensors_.end()) {
    throw error("no poplar::Tensor " + id);
  }
  return found->second;
}

const std::map<TensorId, poplar::Tensor> &PopTensors::getTensors() const {
  return tensors_;
}

PipelineInfo::PipelineInfo(int _batchesPerStep,
                           int _gradAcclFactor,
                           int _numPipelineStages,
                           bool _doTraining,
                           bool _doGradAccl)
    : doTraining(_doTraining), doGradAccl(_doGradAccl) {

  auto bps                  = static_cast<int64_t>(_batchesPerStep);
  auto gradAcclFactor       = static_cast<int64_t>(_gradAcclFactor);
  auto numPipelineStages    = static_cast<int64_t>(_numPipelineStages);
  auto fillFlushPhaseCycles = numPipelineStages;
  fillPhase.start           = 0;
  fillPhase.end             = fillFlushPhaseCycles - 1;

  int64_t mainCycles;
  if (doGradAccl) {
    mainCycles = gradAcclFactor - fillFlushPhaseCycles;
  } else {
    mainCycles = bps - fillFlushPhaseCycles;
  }
  if (mainCycles < 1) {
    throw error(
        "ILE: mainCycles should not be less than 1. Current value is {}.",
        mainCycles);
  }

  mainPhase.start = fillPhase.end + 1;
  mainPhase.end   = mainPhase.start + mainCycles - 1;

  flushPhase.start = mainPhase.end + 1;
  flushPhase.end   = flushPhase.start + fillFlushPhaseCycles - 1;
}

bool PipelineInfo::doStage(PipelineCycle pCycle, PipelineStage pStage) const {
  bool doStageLower = (pCycle >= pStage);
  bool doStageUpper = (pCycle < pStage + flushPhase.start);

  return (doStageLower && doStageUpper);
}

poplar::Graph &Devicex::graph() { return *pGraph; }
const poplar::Graph &Devicex::graph() const { return *pGraph; }

poplar::Graph &Devicex::getVirtualGraph(VGraphId virtualGraphIndex) {
  if (virtualGraphIndex < 0 || virtualGraphIndex >= virtualGraphs.size()) {
    throw error("Invalid virtual graph index {} ({} available)",
                virtualGraphIndex,
                virtualGraphs.size());
  }
  return virtualGraphs.at(virtualGraphIndex);
}
Devicex::~Devicex() = default;

Devicex::Devicex(const Ir &ir, std::shared_ptr<DeviceInfo> deviceInfo_)
    : _ir(ir), progs(PopPrograms(this)), tensors(ir), deviceInfo(deviceInfo_),
      prepareHasBeenCalled(false) {

  logging::devicex::info("Setting selected device: {}", *deviceInfo);

  if (!deviceInfo->attach()) {
    throw error("failed to attach to device");
  }

  // Set the opxTrace flag based on the environment variable
  auto POPART_OPX_TRACE = getPopartEnvVar("OPX_TRACE");
  opxTrace = POPART_OPX_TRACE ? strncmp(POPART_OPX_TRACE, "1", 1) == 0 : false;

  // TODO (see T5100) : if inference, forward should be INFERENCE_FWD
  for (auto it : ir.getSessionOptions().convolutionOptions) {
    logging::devicex::info(
        "Setting user convolution option {} = {}", it.first, it.second);
    fwdConvOptions.options[it.first] = it.second;
    bwdConvOptions.options[it.first] = it.second;
    wuConvOptions.options[it.first]  = it.second;
  }

  if (ir.getExecutionMode() == Ir::ExecutionMode::TRAINING) {
    fwdConvOptions.options["pass"] = "TRAINING_FWD";
    lstmOptions.set("inferenceOnly", "false");
  } else {
    fwdConvOptions.options["pass"] = "INFERENCE_FWD";
    lstmOptions.set("inferenceOnly", "true");
  }

  bwdConvOptions.options["pass"] = "TRAINING_BWD";
  wuConvOptions.options["pass"]  = "TRAINING_WU";

  if (ir.getSessionOptions().enableFullyConnectedPass) {
    if (ir.getExecutionMode() == Ir::ExecutionMode::TRAINING) {
      fwdMmOptions.options.insert({"fullyConnectedPass", "TRAINING_FWD"});
      bwdMmLhsOptions.options.insert({"fullyConnectedPass", "TRAINING_BWD"});
      bwdMmRhsOptions.options.insert({"fullyConnectedPass", "TRAINING_WU"});
    } else {
      fwdMmOptions.options.insert({"fullyConnectedPass", "INFERENCE_FWD"});
    }
  }

  if (ir.getSessionOptions().enablePipelining) {
    pInfo = PipelineInfo(
        ir.getDataFlow().batchesPerStep(),
        static_cast<int>(ir.getSessionOptions().accumulationFactor),
        static_cast<int>(getMaxPipelineStage()),
        ir.canTrain(),
        ir.getSessionOptions().enableGradientAccumulation);
  }

  engineOptions.set("target.workerStackSizeInBytes", "0x200");

  if (ir.getSessionOptions().enablePrefetchDatastreams) {
    logging::devicex::info("Setting engine options for prefetch data streams "
                           "(exchange.streamBufferOverlap = hostRearrangeOnly, "
                           "exchange.enablePrefetch = true");
    engineOptions.set("exchange.streamBufferOverlap", "hostRearrangeOnly");
    engineOptions.set("exchange.enablePrefetch", "true");
  }

  for (auto it : ir.getSessionOptions().engineOptions) {
    logging::devicex::info(
        "Setting engine option {} = {}", it.first, it.second);
    engineOptions.set(it.first, it.second);
  }

  for (auto it : ir.getSessionOptions().reportOptions) {
    logging::devicex::info(
        "Setting report option {} = {}", it.first, it.second);
    reportOptions.set(it.first, it.second);
  }
}

void Devicex::weightsFromHost() {
  if (useSyntheticData() == false) {
    logging::devicex::debug("Writing weights from host, ");
    pEngine->disableExecutionProfiling();
    run(PopPrograms::ProgramIndex::WEIGHTSFROMHOST);
    logging::devicex::debug("done.");
  }
}

void Devicex::optimizerFromHost() {
  if (useSyntheticData() == false) {
    logging::devicex::debug("Writing optimizer from host, ");
    pEngine->disableExecutionProfiling();
    run(PopPrograms::ProgramIndex::OPTIMIZERFROMHOST);
    logging::devicex::debug("done.");
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
  const void *src;

  // size of the host end of the poplar stream.
  // It is a char vector, so this is in bytes.
  int64_t nbytes_src;

  src        = static_cast<const void *>(d2hWeightBuffers.at(id).data());
  nbytes_src = d2hWeightBuffers.at(id).size();

  auto dst = mv_data.data;

  // number of bytes of the destination.
  int64_t nbytes_dst = mv_data.info.nbytes();

  // display which tensors are being copied
  logging::devicex::debug("       {} {}", id, ir().getTensor(id)->info.shape());

  // We confirm that the sizes of src and dst are the same
  if (nbytes_src != nbytes_dst) {
    std::stringstream errms;
    errms << "sizes (in bytes) of src (" << nbytes_src << ") and dst ("
          << nbytes_dst << ") differ in hostStreamToHost for " << id;
    throw error(errms.str());
  }

  std::memcpy(dst, src, nbytes_src);
}

void Devicex::anchorsHostToHostStreams(IStepIO &stepio) {

  if (useSyntheticData() == false) {
    std::string prefix = "     ";
    logging::devicex::debug(prefix + "Copying to h2d stream address(es) ");
    for (Tensor *tensor : ir().dataStreamTensors()) {
      inputStreams.at(tensor->id)->setStepIO(&stepio);
    }
  }
}

void Devicex::anchorsHostFromHostStreams(IStepIO &stepio) {

  if (useSyntheticData() == false) {
    std::string prefix = "     ";
    logging::devicex::debug(prefix + "Copying from d2h stream address(es) ");
    for (TensorId anchorId : ir().getDataFlow().anchors()) {
      outputStreams.at(anchorId)->setStepIO(&stepio);
    }
  }
}

void Devicex::run(IStepIO &stepio) {
  if (!prepareHasBeenCalled) {
    throw error("Devicex::prepare() must be called before"
                " Devicex::run(const IStepIO &) is called.");
  }
  logging::devicex::debug("Performing one step: ");

  // Configure the inputstreams
  anchorsHostToHostStreams(stepio);

  // Configure the outputstreams
  anchorsHostFromHostStreams(stepio);

  pEngine->enableExecutionProfiling();
  run(PopPrograms::ProgramIndex::PROGRAM);
}

std::unique_ptr<Opx> Devicex::createOpx(Op *op) {

  auto opx = OpxManager::createOpx(op, this);

  if (!opx) {
    if (op->opid == Onnx::Operators::Constant_1 ||
        op->opid == Onnx::Operators::Constant_9) {
      throw error("ILE: No Opx for {}", op->opid);
    } else {
      throw error("Could not create opx for '{}'", op->opid);
    }
  }

  return opx;
}

Opx *Devicex::getOpx(OpId id) { return opxs.at(id).get(); }

const Opx *Devicex::getOpx(OpId id) const { return opxs.at(id).get(); }

// The Id of the task which adds a Tensor to a poplar::Graph
TaskId Devicex::taskWhichCreates(TensorId id) const {
  Tensor *tensor = ir().getTensor(id);
  // Tensors without producers are created by special tasks,
  if (!tensor->hasProducer()) {
    return initTensorTaskId(id);
  }

  // Tensors with producer Ops are created (added to a Graph) by their
  // producer's OpTask
  else {
    return opTaskId(tensor->getProducer());
  }
}

TaskId Devicex::taskWhichPopulates(TensorId id) const {
  Tensor *tensor = ir().getTensor(id);

  // OpTasks both initialize a Tensor, and generate the code to set its value
  if (tensor->hasProducer()) {
    return opTaskId(tensor->getProducer());
  }

  // if a Tensor is of type Stream, the Copy from host to device populates it
  else if (!useSyntheticData() && tensor->tensorType() == TensorType::Stream) {
    return fromHostTaskId(tensor->id);
  }

  // default:
  else {
    return initTensorTaskId(id);
  }
}

std::vector<ICreatorCandidatePtr> Devicex::getCreatorEndpoints(
    Tensor *tensor,
    std::vector<ICreatorCandidate::OpxInAndOutIndex> pathFromInput,
    bool excludeEndpointsFromPath,
    bool includeDeadends) const {

  std::vector<ICreatorCandidatePtr> endpoints;
  for (Op *op : tensor->consumers.getOps()) {
    auto conOpId   = op->id;
    const Opx *opx = getOpx(conOpId);

    for (int inIndex : op->input->indices(tensor)) {
      auto updatedPath = pathFromInput;

      switch (opx->getInputCreatorType(inIndex)) {
      // Opx has poplar call to layout tensor at this
      // inIndex
      case InputCreatorType::CANCREATE: {
        if (!excludeEndpointsFromPath) {
          updatedPath.push_back({opx, inIndex, -1}); // note: no valid outIndex
        }
        endpoints.push_back(
            std::make_shared<InputCreatorCandidate>(inIndex, opx, updatedPath));
        break;
      }
      // Recursively search the DAG downstream of the op until we
      // have set of endpoints that can create the tensor
      case InputCreatorType::CANUNWIND: {
        for (auto &ind_ten : op->output->tensorMap()) {
          auto nextOutputTensor = ind_ten.second;
          auto outIndex         = ind_ten.first;
          updatedPath.push_back({opx, inIndex, outIndex});
          for (auto &candidate :
               getCreatorEndpoints(nextOutputTensor, updatedPath)) {
            endpoints.push_back(candidate);
          }
        }
        break;
      }
      // Recursively search the DAG downstream of the op until we
      // have set of creator endpoints that can create the tensor
      case InputCreatorType::CANUNWIND_MULTIPLE_CREATORS: {
        // This does not handle multiple output op's
        // TODO : Need to handle CANUNWIND_MULTIPLE_CREATORS with multiple
        // outputs
        auto outputName = opx->op_p->output->tensorMap().at(0)->id;

        // Get the list of creator candicates
        // ?? Better name ??
        auto creators = opx->getCreatorCandicates(inIndex);

        // Create the InputMultiCreatorCandidate
        std::vector<ICreatorCandidate::OpxInAndOutIndex> path = {};
        std::shared_ptr<InputMultiCreatorCandidate> creatorCandidate =
            std::make_shared<InputMultiCreatorCandidate>(inIndex, opx, path);

        // For each creator input recursively work out their creator candidates
        for (auto creator : creators) {
          auto ind_ten = creator.first->input->tensorMap().at(creator.second);

          // Create a new path for each creator
          for (auto candidate : getCreatorEndpoints(ind_ten, {})) {
            creatorCandidate->addCreateorCandidate(candidate);
          }
        }

        // Save the path from the 'input' to this creator cadidator
        creatorCandidate->setPathFromInput(updatedPath);

        // Add creator to the options
        endpoints.push_back(creatorCandidate);

        break;
      }
      // Consuming op can't create tensor
      case InputCreatorType::DEADEND: {
        if (includeDeadends) {
          if (!excludeEndpointsFromPath) {
            updatedPath.push_back(
                {opx, inIndex, -1}); // note: no valid outIndex
          }
          endpoints.push_back(std::make_shared<InputCreatorCandidate>(
              inIndex, opx, updatedPath));
        }
        break;
      }
      default: {
        throw error("InputCreatorType not implemented for Opx of OpId {}",
                    op->id);
      }
      }
    }
  }
  return endpoints;
}

ICreatorCandidatePtr Devicex::getTensorCreator(Tensor *tensor) const {
  // Search of the graph to get the candidate Opxs that
  // know how to create this tensor.
  // The pathFromInput argument is an empty vector, as
  // we are starting the search from the root (input)

  std::vector<ICreatorCandidatePtr> candidates =
      getCreatorEndpoints(tensor, {});

  // Filter out all but highest priority candidates

  if (candidates.size() > 0) {

    auto hasSmallerPriority = [&](ICreatorCandidatePtr icc1,
                                  ICreatorCandidatePtr icc2) -> bool {
      return icc1->getMaxCreatorPriority() < icc2->getMaxCreatorPriority();
    };
    auto maxPriorityCandidate = *std::max_element(
        candidates.begin(), candidates.end(), hasSmallerPriority);

    auto hasNonMaxPriority =
        [maxPriorityCandidate](ICreatorCandidatePtr &icc) -> bool {
      return icc->getMaxCreatorPriority() <
             maxPriorityCandidate->getMaxCreatorPriority();
    };

    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(), hasNonMaxPriority),
        candidates.end());
  }

  if (candidates.size() > 1) {
    // check that all creators are in agreement on how
    // to create the poplar::Tensor. If they are, just keep
    // the first one.
    bool allEquivalent         = true;
    ICreatorCandidatePtr cand0 = candidates[0];
    for (int i = 1; i < candidates.size(); ++i) {
      auto cand1 = candidates[i];
      if (!cand0->createsEquivalent(cand1)) {
        allEquivalent = false;
        break;
      }
    }
    if (!allEquivalent) {
      logging::devicex::warn(
          "Input tensor '{}' has multiple creator candidates with the same "
          "priority, but they are not in agreement. Picking first creator "
          "candidate.",
          tensor->id);
    }
  }

  if (candidates.size() > 0) {
    return candidates.front();
  } else {
    return nullptr;
  }
}

// Design decision : leave the option for a Tensor to be
// created based on complex global criteria open.
PriTask Devicex::initTensorTask(Tensor *tensor) {
  auto candidate = getTensorCreator(tensor);

  // InputCreatorCandidate* candidate  =
  // (dynamic_cast<InputCreatorCandidate*>(candidate_.get()));

  // 1. A unique candidate creator will create the tensor
  // 2. The tensor will be unwound (have its layout modified)
  //    by view-changing opxs on the path from the input to
  //    the candidate candidate
  if (candidate) {

    auto f = [this, candidate, tensor]() {
      logging::devicex::debug(
          "Creating poplar::Tensor {}, with layout allocated by {}",
          tensor->id,
          candidate->str());

      poplar::Tensor input = candidate->createInput(tensor->str());

      tensors.insert(tensor->id, input);
      efficientlyCreatedInputTensors.insert(tensor->id);
    };
    // the inputs of creator which must have poplar::Tensors
    // before creator creates input tensor at index inIndex.
    std::vector<TaskId> deps;
    for (TensorId tenId : candidate->mustExistBeforeCreate()) {
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
  } else {

    auto f = [this, tensor]() {
      logging::devicex::debug("Creating poplar::Tensor '{}' linearly. No "
                              "operator specific allocator found",
                              tensor->id);

      // Get paths to both creator candidates and deadends, and print for debug
      std::vector<ICreatorCandidatePtr> endpoints =
          getCreatorEndpoints(tensor, {}, false, true);
      int endpointId = 1;
      logging::devicex::debug("Printing paths to {} endpoint(s) found when "
                              "searching for a creator candidate for {}",
                              endpoints.size(),
                              tensor->id);
      for (auto endpoint : endpoints) {
        auto path = endpoint->getPathFromInput();
        logging::devicex::debug("  Path to endpoint {}, starting from input",
                                endpointId);
        for (auto opxOnPath : path) {
          Op *opOnPath = opxOnPath.opx->op_p;
          logging::devicex::debug(
              "    Op {} : {}", opOnPath->str(), opOnPath->name());
        }
        endpointId += 1;
      }

      // Find the ipu the op that consumes with tensor is on and create the
      // tensor on that graph
      auto consumerOps = tensor->consumers.getOps();

      if (logging::shouldLog(logging::Module::devicex, logging::Level::Trace)) {
        std::stringstream ss;
        for (auto *op : consumerOps) {
          auto index = op->input->indicesMap().at(tensor)[0];
          ss << std::endl
             << "    {" << op->debugName() << ", VGID: "
             << (op->hasVirtualGraphId()
                     ? op->getIntrospectionInVirtualGraphId(index)
                     : -1)
             << "}";
        }
        logging::trace(
            "[initTensorTask] Tensor: {}, type: {}, consumed by ops: [{}]",
            tensor->id,
            tensor->getTensorTypeInfo()->type_s(),
            ss.str());
      }

      std::vector<VGraphId> ipus;
      for (auto *op : consumerOps) {
        VGraphId vgid = -1;
        if (op->hasVirtualGraphId()) {
          // VirtualGraphId with subgraph call introspection
          // for the current tensor
          auto index = op->input->indicesMap().at(tensor)[0];
          vgid       = op->getIntrospectionInVirtualGraphId(index);
        }

        // The copyToIpu op assume that the tensor will already
        // have been copied to the ipu from another op
        if (op->opid != Onnx::CustomOperators::IpuCopy) {

          if (ipus.end() == std::find(ipus.begin(), ipus.end(), vgid)) {

            auto &graph =
                vgid > -1 ? getVirtualGraph(vgid) : getOpx(op->id)->graph();

            auto newTensor = graph.addVariable(
                popType(tensor->info), tensor->info.shape_szt(), tensor->str());
            linearMapper.mapTensor(graph, newTensor);

            tensors.insert(tensor->id, newTensor);
            linearlyCreatedInputTensors.insert(tensor->id);
            ipus.push_back(vgid);
          }
        }
      }
    };

    return {1e6, initTensorTaskId(tensor->id), {}, f};
  }
}

PriTask Devicex::initRandomSeed() {
  auto initRandomSeedTask = [this]() {
    logging::devicex::debug("Initializing random seed.");

    randomSeedTensor =
        graph().addVariable(poplar::UNSIGNED_INT, {2}, randomSeedId());
    graph().setTileMapping(randomSeedTensor, 0);

    auto &sq = progs.setRandomSeedFragment();

    if (!useSyntheticData()) {
      // Right now just use the same random seed on each replica if the user set
      // it T9638 - to corrupt the seed for each replicant
      auto dataStream =
          graph().addHostToDeviceFIFO(h2dId(randomSeedId()),
                                      randomSeedTensor.elementType(),
                                      randomSeedTensor.numElements(),
                                      poplar::ReplicatedStreamMode::REPLICATE);

      sq.add(poplar::program::Copy(dataStream, randomSeedTensor));
    }

    poprand::setSeed(graph(),
                     randomSeedTensor,
                     0,
                     sq,
                     logging::format("{}/set", randomSeedId()));
  };

  return {
      +1e6,                   // high priority
      initRandomSeedTaskId(), // name of this task
      {},                     // depends on
      initRandomSeedTask      // what to run when the task is executed
  };
}

PriTask Devicex::incrementRandomSeedTask() {
  auto incrementRandomSeedTask = [this]() {
    popops::addInPlace(graph(),
                       getRandomSeedTensor(),
                       getConst(graph(), poplar::UNSIGNED_INT, {}, 1, "one"),
                       progs.preForwardFragment());
  };

  return {
      +1e6,                        // high priority
      incrementRandomSeedTaskId(), // name of this task
      {initRandomSeedTaskId()},    // depends on
      incrementRandomSeedTask      // what to run when the task is executed
  };
}

void Devicex::connectRandomSeedStream() {
  // Generate a separate random seed for each replicant.
  for (uint16_t replicaId = 0; replicaId < getReplicationFactor();
       ++replicaId) {

    auto callback = [this, replicaId](void *ptr) {
      logging::devicex::debug(
          "     Updating random seed for replica:{} to `{} + replicaId({})'",
          replicaId,
          randomSeed,
          replicaId);
      uint64_t *data = reinterpret_cast<uint64_t *>(ptr);
      data[0]        = randomSeed + replicaId;
    };

    pEngine->connectStreamToCallback(
        h2dId(randomSeedId()), replicaId, callback);
  }
}

void Devicex::setRandomSeed(uint64_t seedValue) {
  if (!prepareHasBeenCalled) {
    throw error("Devicex::prepare() must be called before "
                "Devicex::setRandomSeed(uint64_t) is called.");
  }

  setRandomSeedInternal(seedValue);
}

void Devicex::setRandomSeedInternal(uint64_t seedValue) {
  randomSeed = seedValue;

  if (useSyntheticData() == false) {
    logging::devicex::debug("Setting the random seed to {}", seedValue);
    pEngine->disableExecutionProfiling();
    run(PopPrograms::ProgramIndex::SETRANDOMSEED);
    logging::devicex::debug("done.");
  }
}

template <typename T> void Devicex::setInitVal(Tensor *tensor) {

  graph().setInitialValue<T>(
      tensors.get(tensor->id),
      poplar::ArrayRef<T>(static_cast<const T *>(tensor->tensorData()->data()),
                          tensor->info.nelms()));
}

// Using specialised poplar function for setting init val for FLOAT16
void Devicex::setInitValHalf(Tensor *tensor) {

  graph().setInitialValueHalf(
      tensors.get(tensor->id),
      poplar::ArrayRef<uint16_t>(
          static_cast<const uint16_t *>(tensor->tensorData()->data()),
          tensor->info.nelms()));
}

PriTask Devicex::setInitTensorValTask(Tensor *tensor) {
  // See T6254. Currently we just use setInitialValue for all constant tensors
  auto f = [this, tensor]() {
    logging::devicex::debug("Setting initial value for tensor {}",
                            tensor->str());

    // see T5925 for making a more compact way of matching
    // types than using this switch statement
    switch (tensor->info.dataType()) {
    case DataType::FLOAT: {
      setInitVal<float>(tensor);
      break;
    }
    case DataType::INT32: {
      setInitVal<int32_t>(tensor);
      break;
    }
    case DataType::FLOAT16: {
      setInitValHalf(tensor);
      break;
    }
    case DataType::BOOL: {
      setInitVal<bool>(tensor);
      break;
    }

    case DataType::UNDEFINED:
    case DataType::UINT8:
    case DataType::INT8:
    case DataType::INT64:
    case DataType::UINT16:
    case DataType::INT16:
    case DataType::STRING:
    case DataType::DOUBLE:
    case DataType::UINT32:
    case DataType::UINT64:
    case DataType::COMPLEX64:
    case DataType::COMPLEX128:
    case DataType::BFLOAT16:
    default: {
      throw error(
          "setInitTensorValTask not implemented for Tensor {} of Type {}. ",
          tensor->id,
          tensor->info.data_type());
    }
    }
  };

  return {// priority unimportant
          0,
          // name of this task
          setInitTensorValTaskId(tensor->id),
          // poplar::Tensor must exist. Other that this, this task can be
          // performed any time
          {initTensorTaskId(tensor->id)},
          f};
}

PriTask Devicex::streamFromHostTask(Tensor *tensor) {
  auto f = [this, tensor]() {
    std::vector<VGraphId> ipus;

    auto consumerOps = tensor->consumers.getOps();

    for (auto *op : consumerOps) {
      // Assume another op will copy the tensor for an ipucopy
      if (op->opid != Onnx::CustomOperators::IpuCopy) {
        auto &graph = getOpx(op->id)->graph();

        VGraphId vgid = -1;
        if (op->hasVirtualGraphId()) {
          // VirtualGraphId with subgraph call introspection
          // for the current tensor
          auto index = op->input->indicesMap().at(tensor)[0];
          vgid       = op->getIntrospectionInVirtualGraphId(index);
        }

        // Only stream the tensor once for all op's that consume it on an ipu
        if (std::find(ipus.begin(), ipus.end(), vgid) == ipus.end()) {

          logging::devicex::debug(
              "Creating host-to-device FIFO {} copied to ipu:{}",
              tensor->id,
              vgid);

          poplar::ReplicatedStreamMode mode;

          if (tensor->tensorType() == TensorType::Variable) {
            // If it is a variable we 'broadcast' the same tensor
            // to all replicants
            mode = poplar::ReplicatedStreamMode::BROADCAST;

          } else if (tensor->tensorType() == TensorType::Stream) {

            switch (tensor->getReplicatedStreamMode()) {
            case Tensor::ReplicatedStreamMode::Broadcast:
              mode = poplar::ReplicatedStreamMode::BROADCAST;
              break;
            case Tensor::ReplicatedStreamMode::Replicate:
              mode = poplar::ReplicatedStreamMode::REPLICATE;
              break;
            }

          } else {
            throw error("Tensor {} of type {} are not stream to device",
                        tensor->id,
                        tensor->tensorType());
          }

          fromHostStreams.emplace(
              tensor->id,
              graph.addHostToDeviceFIFO(h2dId(tensor->id),
                                        popType(tensor->info),
                                        tensor->info.nelms(),
                                        mode));

          ipus.push_back(vgid);
        }
      }
    }
  };

  return {
      0,                                // priority unimportant
      streamFromHostTaskId(tensor->id), // name of this task
      {initTensorTaskId(tensor->id)},   // poplar::Tensor must exist
      f                                 // what to run when the task is executed
  };
}

PriTask Devicex::streamToHostTask(Tensor *tensor, bool isAnchorStream) {
  auto f = [this, tensor, isAnchorStream]() {
    logging::devicex::debug("Creating device-to-host FIFO {}", tensor->id);

    auto pToHostStreams = &toHostAnchorStreams;
    if (!isAnchorStream) {
      pToHostStreams = &toHostWeightStreams;
    }

    pToHostStreams->emplace(
        tensor->id,
        graph().addDeviceToHostFIFO(d2hId(tensor->id, isAnchorStream),
                                    popType(tensor->info),
                                    tensor->info.nelms()));
  };

  return {
      0,                                              // priority unimportant
      streamToHostTaskId(tensor->id, isAnchorStream), // name of this task
      {taskWhichCreates(tensor->id)}, // poplar::Tensor must exist
      f                               // what to run when the task is executed
  };
}

bool Devicex::containsFragment(const Graph &graph) const {
  return progs.containsFragment(graph);
}

void Devicex::createFragment(const Graph &graph) {
  return progs.createFragment(graph);
}

poplar::Function &Devicex::getFragmentFunction(const Graph &called_graph) {
  logging::devicex::trace("[getFragmentFunction] Getting function for graph {}",
                          called_graph.id.str());
  return progs.getFragmentFunction(called_graph, graph());
}

void Devicex::addPipelinedCopyTasks(PriTasks &tasks) {
  auto schedule          = ir().getMainGraph().getOpSchedule({});
  std::string prevTaskId = "";

  for (auto iter = schedule.rbegin(); iter != schedule.rend(); iter++) {
    auto &op = *iter;
    if (op->isConvertibleTo<IpuCopyOp>() && !op->copiesOptimizerTensors()) {
      auto task = pipelinedCopyTask(op, prevTaskId);
      tasks.add(task);
      prevTaskId = task.name;
    }
  }
}

PriTask Devicex::pipelinedCopyTask(Op *op, TaskId prevTaskId) {
  auto copyOp  = dynamic_cast<IpuCopyOp *>(op);
  auto opx     = getOpx(copyOp->id);
  auto copyOpx = dynamic_cast<IpuCopyOpx *>(opx);

  auto f = [this, copyOp, copyOpx]() {
    logging::debug("Adding pipelined copies for op {}", copyOp->debugName());
    for (auto &prog : progs.pipelineIpuCopyFragments(
             copyOp->getPipelineStage(),
             logging::format("{}, {}, PipelineStage({})",
                             copyOp->debugName(),
                             copyOp->getFromToStr(),
                             copyOp->getPipelineStage()))) {
      copyOpx->growPipelined(*prog);
    }
  };

  std::vector<TaskId> deps;
  if (!prevTaskId.empty()) {
    // Ensure the ops are scheduled in the order we're iterating through them
    // here.
    deps.push_back(prevTaskId);
  }

  // The ops opTask needs to run first to create the destination tensor.
  deps.push_back(opTaskId(op));

  return {-100, pipelinedCopyTaskId(op), deps, f};
}

void Devicex::addOpTasks(PriTasks &tasks) {

  // Ensure there is a program fragment for every Ir Graph
  for (auto graph : ir().getGraphSchedule()) {
    if (!containsFragment(*graph)) {
      createFragment(*graph);
    }
  }

  auto mainGraphSchedule = ir().getMainGraph().getOpSchedule({});

  // repeating logic in Ir::getOpSchedule (can be simplified there?)
  std::vector<Op *> allOps;
  std::set<const Graph *> addedGraphs;
  std::function<void(const Graph *)> addGraph;
  addGraph = [&allOps, &addedGraphs, &addGraph](const Graph *graph) {
    if (addedGraphs.find(graph) != addedGraphs.end()) {
      return;
    }
    addedGraphs.insert(graph);

    // Add each op in the graph
    for (auto op : graph->getOpSchedule({})) {
      // If the op calls another graph, then
      // the ops in that graph should be scheduled first
      for (auto calledGraph : op->getCalledGraphs()) {
        addGraph(calledGraph);
      }
      if (op->settings.recomputeType != RecomputeType::CHECKPOINT) {
        throw error("ILE: non-main Graph Op which is not a CHECKPOINT");
      }
      allOps.push_back(op);
    }
  };

  for (auto op : mainGraphSchedule) {
    for (auto calledGraph : op->getCalledGraphs()) {
      addGraph(calledGraph);
    }
    allOps.push_back(op);
  }

  double priority     = 0.0;
  TaskId prevOpTaskId = "";

  // Iterate through Ops according to the Ir's schedule
  for (auto op : allOps) {
    for (auto graph : op->getCalledGraphs()) {
      auto opInputs = op->getInputsForGraph(*graph);
      for (int i = 0; i < opInputs.size(); i++) {
        auto graphInput = graph->getInputId(i);
        if (!tasks.contains(initTensorTaskId(graphInput))) {
          tasks.add(initTensorByCloningTask(op, opInputs.at(i), graphInput));
        }
      }
    }

    auto task = opTask(op, priority, prevOpTaskId);

    tasks.add(task);
    prevOpTaskId = task.name;
    priority -= 1.;
  }
}

PriTask
Devicex::initTensorByCloningTask(Op *op, TensorId srcId, TensorId dstId) {
  Opx *opx = getOpx(op->id);

  auto f = [srcId, dstId, opx, this]() {
    logging::debug("Cloning tensor {} to {}", srcId, dstId);
    auto src = opx->get(srcId);
    auto dst = opx->graph().clone(src);
    tensors.insert(dstId, dst);
  };

  std::vector<TaskId> deps;
  auto creatorTask = taskWhichCreates(srcId);
  deps.push_back(creatorTask);

  return {-1e6, initTensorTaskId(dstId), deps, f};
}

// This code should be refactored
PriTask Devicex::opTask(Op *op, double priority, TaskId prevOpTaskId) {
  Opx *opx = getOpx(op->id);

  // although priority should guarantee that this
  // task is only run after inputs are all created,
  // we add a dependency to the input tensors, just
  // in case someone plays with the priorities.
  // Moreover, we must state the copy-from-host deps
  std::vector<TaskId> deps;
  for (auto t_inds : op->input->indicesMap()) {
    Tensor *tensor = t_inds.first;

    auto creatorTask = taskWhichPopulates(tensor->id);

    // Make sure we only add the creatorTask once in the dependency list
    if (std::find(deps.begin(), deps.end(), creatorTask) == deps.end()) {
      deps.push_back(creatorTask);
    }
  }

  auto addGraphOpsToDeps = [&](const Graph &graph) {
    for (auto graphOp : graph.getOpSchedule({})) {
      auto taskId = opTaskId(graphOp);
      if (std::find(deps.begin(), deps.end(), taskId) == deps.end()) {
        deps.push_back(taskId);
      }
    }
  };

  // TODO This could probably be made generic in the future
  // for (auto &graph : op->getCalledGraphs()) { ... }
  if (op->isConvertibleTo<CallOp>()) {
    auto callOp = dynamic_cast<CallOp *>(op);
    addGraphOpsToDeps(callOp->getCalledGraph());
  } else if (op->isConvertibleTo<IfOp>()) {
    auto ifOp = dynamic_cast<IfOp *>(op);
    addGraphOpsToDeps(ifOp->getThenGraph());
    addGraphOpsToDeps(ifOp->getElseGraph());
  }

  // Depends on previous op task. This preserves op ordering from ir.
  // Note: the first opTask has no previous opTask
  if (prevOpTaskId != "") {
    // Add dependency only if not already added
    if (std::find(deps.begin(), deps.end(), prevOpTaskId) == deps.end()) {
      deps.push_back(prevOpTaskId);
    }
  }

  auto f = [op, opx, this]() {
    auto growOpx = [opx, this](poplar::program::Sequence &seq) {
      if (opxTrace) {
        seq.add(poplar::program::PrintTensor(opx->op_p->str() + "/enter",
                                             opxTraceTensor));
        opx->grow(seq);
        seq.add(poplar::program::PrintTensor(opx->op_p->str() + "/exit",
                                             opxTraceTensor));
      } else {
        opx->grow(seq);
      }
    };

    const auto &containingGraph = opx->op_p->getGraph();
    // if this Op is not in the main scope
    if (!containingGraph.id.str().empty()) {
      logging::devicex::debug("Creating output tensors for non-main " +
                              opx->op_p->debugName());
      growOpx(progs.scopeFragment(containingGraph));
    }

    // else if this Op is in the main scope
    else {
      if (op->copiesOptimizerTensors()) {
        growOpx(progs.streamOptimizerFromHostFragment());
      }

      // pre-loss : create vertices for all recompute types
      else if (op->scheduledPreLoss == ScheduledPreLoss::Yes) {

        // Pre-loss, not recompute
        if (op->settings.recomputeType == RecomputeType::CHECKPOINT) {
          logging::devicex::debug("Adding checkpoint Op {}", op->debugName());

          // Pre-loss, not recompute, pipelining
          if (ir().getSessionOptions().enablePipelining) {
            auto ipuCopyOp = dynamic_cast<IpuCopyOp *>(op);

            if (ipuCopyOp) {
              // IpuCopyOps are handled as a special case in pipelining. Here,
              // the destination tensor is created using the
              // `createPipelinedOutput` method. Later, for each pipeline cycle
              // the copy appears in, a new copy program is added to the cycles
              // sequence using `IpuCopyOpx::growPipelined`.
              dynamic_cast<IpuCopyOpx *>(opx)->createPipelinedOutput();
            } else {
              growOpx(progs.pipelineForwardFragment(op->getPipelineStage(),
                                                    op->str()));
            }
          }

          // Pre-loss, not recompute, no pipelining
          else {
            growOpx(progs.forwardFragment());
          }
        }

        // Pre-loss, recompute
        else if (op->settings.recomputeType == RecomputeType::RECOMPUTE) {
          logging::devicex::debug("Adding (first) recompute Op {}",
                                  op->debugName());

          growOpx(progs.recomputeFragment(op->id));

          // Pre-loss, recompute, pipelining
          if (ir().getSessionOptions().enablePipelining) {
            progs.pipelineForwardFragment(op->getPipelineStage(), op->str())
                .add(progs.recomputeFragment(op->id));
          }

          // Pre-loss, recompute, no-pipelining
          else {
            progs.forwardFragment().add(progs.recomputeFragment(op->id));
          }
        }

        // Pre-loss, not recompute or checkpoint
        else {
          throw error("ILE: Unrecognised recompute type");
        }
        mainGraphOpRegistery.push_back(op);
      }

      // post-loss
      else if (op->scheduledPreLoss == ScheduledPreLoss::No) {
        if (op->settings.recomputeType != RecomputeType::CHECKPOINT) {
          std::stringstream oss;
          op->append(oss);
          throw error("ILE: Non-checkpoint Op which is ScheduledPreLoss::No is "
                      "not permitted: \n{}",
                      oss.str());
        }

        // 2 special case Ops when their is a gradient accumulator / velocity.
        // If we are doing gradient accumulation, we need to ensure the reset
        // and var update aren't run every time. Instead, these fragments sit
        // outside the "main" loop of the fowards and backwards passes.
        // special case Op 1:
        if ((op->isConvertibleTo<SGD1VarUpdateOp>())) {
          outerLoopFragEmpty = false;
          growOpx(progs.varUpdateFromAccumulatorFragment());
        }
        // special case Op 2:
        else if ((op->isConvertibleTo<SGD1AcclUpdateOp>())) {
          outerLoopFragEmpty = false;
          growOpx(progs.resetWeightGradientAccumulatorFragment());
        }

        // post-loss, not special gradient accumulation case,
        else {

          // decide what needs to be re-run.
          std::set<Op *> toRerun;

          auto getRequiredProducers = [&toRerun, this](Op *toBackcheck) {
            std::vector<Op *> newOps;
            for (auto t : toBackcheck->input->tensors()) {
              if (t->hasProducer()) {
                Op *inProducer = t->getProducer();
                // recompute op, which hasn't been recomputed, and hasn't been
                // registered as required yet
                if (inProducer->settings.recomputeType ==
                        RecomputeType::RECOMPUTE &&
                    !progs.hasBeenRecomputed(inProducer->id) &&
                    toRerun.count(inProducer) == 0) {
                  newOps.push_back(inProducer);
                }
              }
            }
            return newOps;
          };

          std::vector<Op *> rerunFront = getRequiredProducers(op);

          while (!rerunFront.empty()) {
            Op *newRecomputeOp = rerunFront.back();
            rerunFront.resize(rerunFront.size() - 1);
            if (toRerun.count(newRecomputeOp) == 0) {
              toRerun.insert(newRecomputeOp);
              for (auto x : getRequiredProducers(newRecomputeOp)) {
                rerunFront.push_back(x);
              }
            }
          }

          // put the ops to rerun in a topological order
          auto aSchedule = op->getGraph().getOpSchedule({});
          std::vector<Op *> toRerunVector;
          toRerunVector.reserve(toRerun.size());
          for (auto schedOp : aSchedule) {
            if (toRerun.count(schedOp) > 0) {
              toRerunVector.push_back(schedOp);
              progs.recordRecomputed(schedOp->id);
            }
          }

          for (auto opToRerun : toRerunVector) {
            logging::devicex::debug("Adding (second) recompute Op {}",
                                    opToRerun->debugName());

            if (ir().getSessionOptions().enablePipelining) {

              progs
                  .pipelineForwardFragment(op->getPipelineStage(),
                                           "recompute of " + opToRerun->str())
                  .add(progs.recomputeFragment(opToRerun->id));

            }

            else {
              progs.backwardFragment().add(
                  progs.recomputeFragment(opToRerun->id));
            }

            mainGraphOpRegistery.push_back(opToRerun);
          }

          logging::devicex::debug("Adding post-turning check-point Op {}",
                                  op->debugName());

          // Post-loss, no pipelining.
          if (!ir().getSessionOptions().enablePipelining) {
            growOpx(progs.backwardFragment());
          }

          // post-loss, with pipelining.
          else {
            auto ipuCopyOp = dynamic_cast<IpuCopyOp *>(op);
            if (ipuCopyOp) {
              // IpuCopyOps are handled as a special case in pipelining. Here,
              // the destination tensor is created using the
              // `createPipelinedOutput` method. Later, for each pipeline cycle
              // the copy appears in, a new copy program is added to the cycles
              // sequence using `IpuCopyOpx::growPipelined`.
              dynamic_cast<IpuCopyOpx *>(opx)->createPipelinedOutput();
            } else {
              growOpx(progs.pipelineForwardFragment(op->getPipelineStage(),
                                                    op->str()));
            }
          }
          mainGraphOpRegistery.push_back(op);
        }
      }

      else {
        throw error("ILE: Unknown SchedulePreLoss in prepare, should "
                    "updateVertices have been called recently?");
      }
    }
    // main scope Ops complete
  };
  return {priority, opTaskId(op), deps, f};
}

ICreatorCandidate::ICreatorCandidate(int index_,
                                     const Opx *opx_,
                                     std::vector<OpxInAndOutIndex> path)
    : pathFromInput(path), index(index_), opx(opx_) {}

poplar::Tensor ICreatorCandidate::unwind(poplar::Tensor input) {

  // Reverse the path,
  // The first element is now the Opx producing a tensor consumed by
  // the candidate.
  // The last element is now the Opx consuming the input we are mapping.

  auto pathToInput = getPathFromInput();
  std::reverse(pathToInput.begin(), pathToInput.end());

  for (auto &opxOnPath : pathToInput) {
    input = opxOnPath.opx->unwindTensorLayout(
        input, opxOnPath.inIndex, opxOnPath.outIndex);
  }

  return input;
}

InputCreatorCandidate::InputCreatorCandidate(
    int index,
    const Opx *opx,
    std::vector<OpxInAndOutIndex> pathFromInput_)
    : ICreatorCandidate(index, opx, pathFromInput_) {}

double InputCreatorCandidate::getMaxCreatorPriority() {
  return getOpx()->inputCreatorPriority;
}

bool InputCreatorCandidate::createsEquivalent(
    const ICreatorCandidatePtr other) {

  const InputCreatorCandidate *otherCreator =
      dynamic_cast<const InputCreatorCandidate *>(other.get());
  if (otherCreator) {
    return getOpx()->createsEquiv(
        getIndex(), otherCreator->getOpx(), otherCreator->getIndex());
  } else {
    return false;
  }
}

poplar::Tensor InputCreatorCandidate::createInput(const std::string &name) {
  poplar::Tensor t = getOpx()->createInput(getIndex(), name);
  return unwind(t);
}

std::vector<TensorId> InputCreatorCandidate::mustExistBeforeCreate() {
  return getOpx()->mustExistBeforeCreate(getIndex());
}

std::string InputCreatorCandidate::str() {

  std::string result = getOpx()->op_p->str();

  result += "(";
  for (auto &i : pathFromInput) {
    result += "<- " + i.opx->op_p->str();
  }
  result += ")";

  return result;
}

InputMultiCreatorCandidate::InputMultiCreatorCandidate(
    int index,
    const Opx *opx,
    std::vector<OpxInAndOutIndex> pathFromInput_)
    : ICreatorCandidate(index, opx, pathFromInput_) {}

double InputMultiCreatorCandidate::getMaxCreatorPriority() {

  return (*std::max_element(
              candidates.begin(),
              candidates.end(),
              [](ICreatorCandidatePtr icc1, ICreatorCandidatePtr icc2) {
                return icc1->getMaxCreatorPriority() <
                       icc2->getMaxCreatorPriority();
              }))
      ->getMaxCreatorPriority();
}

bool InputMultiCreatorCandidate::createsEquivalent(
    const ICreatorCandidatePtr other) {

  // Test each candidate creator
  InputMultiCreatorCandidate *_other =
      dynamic_cast<InputMultiCreatorCandidate *>(other.get());

  if (_other != nullptr) {
    if (candidates.size() == _other->candidates.size()) {

      for (int i = 0; i < candidates.size(); ++i) {
        if (candidates[i]->createsEquivalent(_other->candidates[i]) == false) {
          return false;
        }
      }

    } else {
      return false;
    }
  } else {
    return false;
  }

  return true;
}

poplar::Tensor
InputMultiCreatorCandidate::createInput(const std::string &name) {

  std::vector<poplar::Tensor> tensors;

  for (auto &c : candidates) {
    tensors.push_back(c->createInput(name));
  }

  poplar::Tensor t = getOpx()->unwindTensorLayout(tensors, getIndex(), 0);

  return unwind(t);
}

std::vector<TensorId> InputMultiCreatorCandidate::mustExistBeforeCreate() {
  std::vector<TensorId> tensors;
  for (auto &c : candidates) {
    std::vector<TensorId> t2 = c->mustExistBeforeCreate();
    tensors.insert(tensors.end(), t2.begin(), t2.end());
  }
  return tensors;
}

std::string InputMultiCreatorCandidate::str() {
  std::string result;
  for (auto &c : candidates) {
    result = result + " " + c->str();
  }

  result += "  [";
  for (auto &i : pathFromInput) {
    result += "<- " + i.opx->op_p->str();
  }
  result += "]";

  return result;
}

unsigned Devicex::getReplicationFactor() const {

  unsigned replicationFactor = 1;
  if (ir().getSessionOptions().enableReplicatedGraphs) {
    replicationFactor =
        static_cast<unsigned>(ir().getSessionOptions().replicatedGraphCount);
  }

  else {
    // A check on user input consistency
    if (static_cast<unsigned>(ir().getSessionOptions().replicatedGraphCount) >
        1) {
      throw error(
          "enableReplicatedGraphs is false, but replicatedGraphCount > 1. "
          "Either enable replication, or set the replicated graph count to 1");
    }
  }

  return replicationFactor;
}

unsigned Devicex::getAccumulationFactor() const {

  unsigned accumulationFactor = 1;
  if (ir().getSessionOptions().enableGradientAccumulation) {
    accumulationFactor =
        static_cast<unsigned>(ir().getSessionOptions().accumulationFactor);
  }

  else {
    // A check on user input consistency
    if (static_cast<unsigned>(ir().getSessionOptions().accumulationFactor) >
        1) {
      throw error(
          "enableGradientAccumulation is false, but accumulationFactor > 1. "
          "Either enable gradient accumulation, or set the accumulation factor "
          "to 1");
    }
  }

  return accumulationFactor;
}

PipelineInfo Devicex::pipelineInfo() const { return pInfo; }

bool Devicex::isEngineLoaded() const { return engineIsLoaded; }

void Devicex::setEngineIsLoaded(bool isLoaded) { engineIsLoaded = isLoaded; }

void Devicex::loadEngineAndConnectStreams() {
  DevicexInfo &di = dynamic_cast<DevicexInfo &>(*deviceInfo);

  // Let the device info know that this devicex's engine
  // has most recently loaded its engine onto the poplar
  // device
  for (auto d : di.previouslyLoadedDevicexs) {
    d->setEngineIsLoaded(false);
  }
  di.previouslyLoadedDevicexs.insert(this);
  setEngineIsLoaded(true);

  pEngine->load(di.getDevice());
  logging::devicex::info("Engine loaded");

  if (useSyntheticData() == false) {
    logging::devicex::debug("Connecting initializer streams");
    for (auto id : ir().getTensorIds(TensorType::Variable)) {
      Tensor *tensor = ir().getTensor(id);
      logging::devicex::debug("   {}", tensor->str());
      pEngine->connectStream(h2dId(id), tensor->tensorData()->data());
    }

    // Random seed
    connectRandomSeedStream();

    logging::devicex::debug("Connecting optimizer streams");

    for (auto tensor : ir().optimizerTensors()) {
      logging::devicex::debug("   {}", tensor->str());
      pEngine->connectStream(h2dId(tensor->id), tensor->tensorData()->data());
    }

    auto engineToInputStreamWithCallback =
        [&pEngine = pEngine, this](Tensor *tensor, PopStreamId streamId) {
          std::shared_ptr<InputDatastream> ds =
              std::make_shared<InputDatastream>(tensor, streamId);
          this->inputStreams[tensor->id] = ds;

          auto replicationFactor = getReplicationFactor();
          for (auto replicationIndex = 0; replicationIndex < replicationFactor;
               ++replicationIndex) {

            auto callback = std::make_unique<PrefetchCallback>(ds);
            pEngine->connectStreamToCallback(
                streamId, replicationIndex, std::move(callback));
          }
        };

    auto engineToOutputStreamWithCallback = [&pEngine = pEngine,
                                             this](Tensor *tensor,
                                                   PopStreamId streamId) {
      std::shared_ptr<OutputDatastream> ds =
          std::make_shared<OutputDatastream>(tensor, streamId);
      this->outputStreams[tensor->id] = ds;

      auto callback = [ds](void *ptr) mutable { ds->write(ptr); };

      auto replicationFactor = getReplicationFactor();
      for (auto replicationIndex = 0; replicationIndex < replicationFactor;
           ++replicationIndex) {
        pEngine->connectStreamToCallback(streamId, replicationIndex, callback);
      }
    };

    // Special case for variables (i.e. weights). This should be the same on
    // every replicant so we only return one copy. The poplar api requires a
    // callback for every replicant. So here we will only return replicant 0.
    auto engineToStreamVariables =
        [&pEngine = pEngine, replicationFactor = getReplicationFactor()](
            char *data0, int64_t n_bytes, PopStreamId streamId) {
          for (uint16_t replicaId = 0; replicaId < replicationFactor;
               ++replicaId) {

            auto callback = [replicaId, data0, n_bytes](void *ptr) mutable {
              if (replicaId == 0) {
                char *data = reinterpret_cast<char *>(ptr);
                memcpy(data0, data, n_bytes);
              }
            };

            pEngine->connectStreamToCallback(streamId, replicaId, callback);
          }
        };

    logging::devicex::debug("Connected h2d input data streams");
    for (Tensor *tensor : ir().dataStreamTensors()) {
      logging::devicex::debug(" {}", tensor->id);
      engineToInputStreamWithCallback(tensor, h2dId(tensor->id));
    }

    logging::devicex::debug("Connected d2h anchor data streams");
    for (TensorId anchorId : ir().getDataFlow().anchors()) {

      bool isAnchorStream  = true;
      PopStreamId streamId = d2hId(anchorId, isAnchorStream);
      Tensor *tensor       = ir().getTensor(anchorId);
      logging::devicex::debug(" {}", tensor->id);
      engineToOutputStreamWithCallback(tensor, streamId);
    }

    logging::devicex::debug("Connected d2h weight data streams");
    for (auto initId : ir().getTensorIds(TensorType::Variable)) {

      bool isAnchorStream      = false;
      PopStreamId streamId     = d2hId(initId, isAnchorStream);
      Tensor *tensor           = ir().getTensor(initId);
      int64_t n_bytes          = tensor->info.nbytes();
      d2hWeightBuffers[initId] = std::vector<char>(n_bytes);
      char *data0              = d2hWeightBuffers[initId].data();

      logging::devicex::debug(" {}", initId);
      engineToStreamVariables(data0, n_bytes, streamId);
    }
  }

  uint64_t seed = std::chrono::system_clock::now().time_since_epoch().count();
  setRandomSeedInternal(seed);
}

// Floating point settings are not suported on CPU
void Devicex::setFloatingPointBehaviour(poplar::Graph &graph) {

  if (ir().getSessionOptions().enableFloatingPointChecks) {
    if (deviceInfo->getType() == DeviceType::Ipu) {
      logging::devicex::info("Enabling all floating point checks");
      // Not enabling stochasitc rounding, that is done in a seperate call
      poplar::FloatingPointBehaviour behaviour(true, true, true, false, true);
      poplar::setFloatingPointBehaviour(
          graph, progs.initFragment(), behaviour, "/init");
    } else {
      logging::devicex::warn(
          "Floating point checks cannot be enabled for non IPU devices");
    }
  }
}

// Stocastic rounding is only supported on the IPU
void Devicex::setStochasticRoundingBehaviour(poplar::Graph &graph) {

  if (ir().getSessionOptions().enableStochasticRounding) {
    if (deviceInfo->getType() == DeviceType::Ipu) {
      logging::devicex::info("Enabling stochastic rounding");
      bool behaviour = true;
      poplar::setStochasticRounding(
          graph, progs.initFragment(), behaviour, "/init");
    } else {
      logging::devicex::warn(
          "Stochastic rounding cannot be enabled for non IPU devices");
    }
  }
}

// go all the way to creating the engine and connecting streams
void Devicex::prepare() {
  logging::devicex::info("Poplar version: {}", poplar::versionString());
  logging::devicex::info("Poplar release githash: {}", poplar::packageHash());

  tryLoadExecutable();

  // Do not like the dynamic_cast is there a better way to handle this?
  auto &popDevice = dynamic_cast<DevicexInfo &>(*deviceInfo).getDevice();

  const unsigned numIOTilesPerIpu = 0;
  poplar::replication_factor rf(getReplicationFactor());

  logging::devicex::debug("Creating graph with replication factor {}",
                          getReplicationFactor());

  pGraph.reset(new poplar::Graph(popDevice, numIOTilesPerIpu, rf));

  popops::addCodelets(graph());
  poplin::addCodelets(graph());
  popnn::addCodelets(graph());
  poprand::addCodelets(graph());

  // Add custom codelets as per the user provided list of paths. Allow poplar to
  // infer the file type from the extension. Also feed through the compile
  // flags.
  for (auto codelet : ir().getSessionOptions().customCodelets) {
    logging::devicex::info("Adding codelet: {}", codelet);
    graph().addCodelets(codelet,
                        poplar::CodeletFileType::Auto,
                        ir().getSessionOptions().customCodeletCompileFlags);
  }

  setFloatingPointBehaviour(graph());
  setStochasticRoundingBehaviour(graph());

  if (ir().virtualGraphsEnabled()) {
    auto numIPUs     = graph().getTarget().getNumIPUs();
    auto tilesPerIPU = graph().getTarget().getTilesPerIPU();

    for (unsigned ipu = 0; ipu < numIPUs; ++ipu) {
      unsigned startTile = ipu * tilesPerIPU;
      unsigned endTile   = (ipu + 1) * tilesPerIPU;
      virtualGraphs.emplace_back(
          graph().createVirtualGraph(startTile, endTile));
      logging::devicex::info(
          "Created virtual graph {} from {} to {}", ipu, startTile, endTile);
    }

    // Make sure that the virtual graph information is valid
    for (Op *op : ir().getOpSchedule({})) {
      if (op->hasVirtualGraphId()) {
        VGraphId index = op->getVirtualGraphId();
        if (index < 0 || index >= numIPUs) {
          throw error("{} has been assigned to an invalid virtual graph {}. "
                      "numIPUs = {}.",
                      op->debugName(),
                      index,
                      numIPUs);
        }
      }
    }
  } else {
    auto numIPUs = graph().getTarget().getNumIPUs();
    if (numIPUs > 1 &&
        numIPUs != ir().getSessionOptions().replicatedGraphCount) {
      throw error("If virtual graphs are disabled, the replicated graph count "
                  "({}) needs to be equal to the number of IPUs ({})",
                  ir().getSessionOptions().replicatedGraphCount,
                  numIPUs);
    }
  }

  // Create a constant tensor which will be used if opxTrace is enabled
  if (opxTrace) {
    opxTraceTensor = getConst(graph(), poplar::HALF, {1}, 0, "traceTensor");
  }

  // create an Opx for every Op
  for (Op *op : ir().getOpSchedule({})) {
    opxs[op->id] = createOpx(op);
  }

  PriTasks tasks;

  // weights (variables):
  // 1) make tensor,
  // 2) make stream from host,
  // 3) create write prog,
  // 4) make stream to host,
  // 5) create read prog.
  for (auto id : ir().getTensorIds(TensorType::Variable)) {
    Tensor *tensor = ir().getTensor(id);
    // 1
    tasks.add(initTensorTask(tensor));

    if (useSyntheticData() == false) {
      // 2
      tasks.add(streamFromHostTask(tensor));
      // 3
      tasks.add(fromHostTask(tensor, progs.streamWeightsFromHostFragment()));
      // 4
      bool isAnchorStream = false;
      tasks.add(streamToHostTask(tensor, isAnchorStream));
      // 5
      tasks.add(
          toHostTask(tensor, progs.weightsToHostFragment(), isAnchorStream));
    }
  }

  // constants:
  // 1) make tensor,
  // 2) set initial value.
  for (auto id : ir().getTensorIds(TensorType::Const)) {
    Tensor *tensor = ir().getTensor(id);
    // 1
    tasks.add(initTensorTask(tensor));
    // 2
    tasks.add(setInitTensorValTask(tensor));
  }

  // stream-to-device tensors : 1)  make tensor 2) make stream
  for (auto id : ir().getTensorIds(TensorType::Stream)) {

    Tensor *tensor = ir().getTensor(id);
    // 1
    tasks.add(initTensorTask(tensor));
    logging::devicex::debug("Addings initTensorTask for {}", id);

    if (useSyntheticData() == false) {
      // 2
      tasks.add(streamFromHostTask(tensor));
    }
  }

  // Init the random seed(s)
  tasks.add(initRandomSeed());
  tasks.add(incrementRandomSeedTask());

  // Depending on anchor return types specified by the user, some
  // tensors may need to be added to the graph to keep track of
  // batch count.
  if (ir().getDataFlow().isBatchCountingRequired()) {
    tasks.add(initBatchCounterTensorsTask());
    tasks.add(updateBatchCountTask(progs.preForwardFragment()));
  }

  // stream-to-host tensors : 1) make streams 2) make copy programs
  // note that the order in which tasks are added does not matter,
  // they will be topologically sorted before running
  if (useSyntheticData() == false) {
    for (auto anchorId : ir().getDataFlow().anchors()) {
      Tensor *tensor = ir().getTensor(anchorId);

      bool isAnchorStream = true;
      tasks.add(streamToHostTask(tensor, isAnchorStream));

      // 2
      switch (ir().getDataFlow().art(anchorId).id()) {
      // Copy program runs after every batch
      case (AnchorReturnTypeId::ALL): {
        if (ir().getSessionOptions().enablePipelining) {
          auto isOptimizerTensorCopy = [&](Op *x) {
            return x->isConvertibleTo<IpuCopyOp>() &&
                   dynamic_cast<IpuCopyOp *>(x)->copiesOptimizerTensors();
          };

          PipelineStage ps;
          // Copies of optimizer tensors do not have a pipeline stage.
          if (tensor->hasProducer() &&
              !isOptimizerTensorCopy(tensor->getProducer())) {
            ps = tensor->getProducer()->getPipelineStage();
          } else if (tensor->tensorType() == TensorType::Stream) {
            ps = *tensor->consumers.findLowestPipelineStage();
          } else if (tensor->tensorType() == TensorType::Variable) {
            ps = *tensor->consumers.findHighestPipelineStage();
          }
          // Edge cases where we have a const or optimizer tensor.
          else {
            ps = *tensor->consumers.findHighestPipelineStage();
          }

          tasks.add(
              toHostTask(tensor,
                         progs.pipelineToHostStreamFragment(ps, tensor->str()),
                         isAnchorStream));
        } else {
          tasks.add(toHostTask(
              tensor,
              tensor->tensorType() == TensorType::Variable
                  ? progs.backwardFragment()
                  : progs.forwardOrBackwardFragment(tensor->scheduledPreLoss),
              isAnchorStream));
        }
        break;
      }
      // Copy program runs at the end of every N batches
      case (AnchorReturnTypeId::EVERYN): {
        if (ir().getSessionOptions().enablePipelining) {
          throw error(
              "AnchorReturnType::EVERYN is not valid for pipelined models");
        } else {
          tasks.add(toHostEveryNBatchesTask(
              tensor,
              ir().getDataFlow().art(anchorId).rp(),
              tensor->tensorType() == TensorType::Variable
                  ? progs.backwardFragment()
                  : progs.forwardOrBackwardFragment(tensor->scheduledPreLoss)));
        }
        break;
      }
      // Copy program runs at the end of the step
      case (AnchorReturnTypeId::FINAL): {
        tasks.add(toHostTask(
            tensor, progs.toHostFinalCopyFragment(), isAnchorStream));
        break;
      }
      }
    }

    // create Program to write optimizer tensors to device
    for (auto tensor : ir().optimizerTensors()) {
      tasks.add(fromHostTask(tensor, progs.streamOptimizerFromHostFragment()));
    }

    for (Tensor *tensor : ir().dataStreamTensors()) {
      if (ir().getSessionOptions().enablePipelining) {
        PipelineStage ps = *tensor->consumers.findLowestPipelineStage();
        auto &sq = progs.pipelineToDeviceStreamFragment(ps, tensor->str());
        tasks.add(fromHostTask(tensor, sq));
      } else {
        auto &sq = progs.forwardOrBackwardFragment(tensor->scheduledPreLoss);
        tasks.add(fromHostTask(tensor, sq));
      }
    }
  }

  addOpTasks(tasks);

  if (ir().getSessionOptions().enablePipelining) {
    addPipelinedCopyTasks(tasks);
  }

  for (auto &task : tasks.getLinearised()) {
    task.f();
  }

  logging::devicex::debug(getMainGraphOpString());

  if (ir().getSessionOptions().exportPoplarVertexGraph) {
    std::ofstream strm;
    strm.open("poplar_vertex_graph.dot", std::ios::out);
    graph().outputVertexGraph(strm, progs.progs());
  }

  if (ir().getSessionOptions().exportPoplarComputationGraph) {
    std::ofstream strm;
    strm.open("poplar_compute_graph.dot", std::ios::out);
    graph().outputComputeGraph(strm, progs.progs());
  }

  if (!ir().getSessionOptions().compileEngine) {
    logging::devicex::info("Not compiling engine by request");
    return;
  }

  logging::devicex::info("Starting Engine compilation");

  auto trySaveTensorTileMap = [this]() {
    auto popartTensorTileMap = getPopartEnvVar("TENSOR_TILE_MAP");
    if (popartTensorTileMap && strcmp(popartTensorTileMap, "") != 0) {
      saveTensorTileMap(popartTensorTileMap);
    }
  };

  try {
    auto executable = getExecutable();
    pEngine.reset(new poplar::Engine(std::move(executable), engineOptions));
  } catch (const poplar::graph_memory_allocation_error &e) {
    // If the creation of the engine throw an exception due to memory
    // allocation i.e. the program does not fit show graph profile and
    // re-throw the exception In certain cases poplar will throw the error
    // without a graph profile. The following engine option needs to be set to
    // enable the graph profile in this case "debug.allowOutOfMemory":"true"

    trySaveTensorTileMap();

    logging::devicex::err("Memory allocation error : {}", e.what());
    throw devicex_memory_allocation_err(e, reportOptions);
  }

  logging::devicex::info("Engine compiled");

  loadEngineAndConnectStreams();

  trySaveTensorTileMap();

  prepareHasBeenCalled = true;
}

int64_t Devicex::getStashSize(VGraphId vGraphId) {
  int64_t maxVGraphId = static_cast<int64_t>(ir().getMaxVirtualGraphId());
  return 2 * (maxVGraphId - vGraphId) - 1;
}

poplar::Executable Devicex::getExecutable() {
  auto progressLogger = [](int progress, int total) {
    if (total != 0) {
      float percentage = std::floor(100.0f * static_cast<float>(progress) /
                                    static_cast<float>(total));
      logging::devicex::info("Engine compilation {}% complete", percentage);
    }
  };

  if (cachedExecutable) {
    // return the executable in cachedExecutable while ensuring
    // cachedExecutable is set to boost::none
    optional<poplar::Executable> result = boost::none;
    boost::swap(cachedExecutable, result);
    return std::move(result.get());
  } else {
    auto executable = poplar::compileGraph(
        graph(), progs.progs(), engineOptions, progressLogger);
    trySaveExecutable(executable);
    return executable;
  }
}

namespace {

class SavedInfo {
public:
  SavedInfo(const Devicex &devicex) : irHash(std::hash<Ir>{}(devicex.ir())) {}

  void serialize(std::ostream &os) { os << irHash; }

  static SavedInfo deserialize(std::istream &is) {
    SavedInfo result;
    is >> result.irHash;
    return result;
  }

  bool operator==(const SavedInfo &rhs) { return irHash == rhs.irHash; }

  std::size_t irHash;

private:
  SavedInfo() : irHash(0) {}
};

} // namespace

std::string Devicex::getPoplarCachePath() {
  return ir().getSessionOptions().cachePath + ".poplar";
}

std::string Devicex::getPopartCachePath() {
  return ir().getSessionOptions().cachePath + ".popart";
}

void Devicex::trySaveExecutable(poplar::Executable &executable) {

  auto cachePath    = ir().getSessionOptions().cachePath;
  auto cacheEnabled = ir().getSessionOptions().enableEngineCaching;

  if (cacheEnabled && !cachePath.empty() &&
      deviceInfo->getType() == DeviceType::Ipu) {
    // save the poplar executable
    auto poplarCachePath = getPoplarCachePath();
    std::ofstream poplarFs(poplarCachePath, std::ofstream::binary);
    logging::devicex::debug("Saving poplar Executable to '{}'",
                            poplarCachePath);
    executable.serialize(poplarFs);

    // save the popart ir hash
    auto popartCachePath = getPopartCachePath();
    std::ofstream popartFs(popartCachePath, std::ofstream::binary);
    logging::devicex::debug("Saving popart ir hash to '{}'", popartCachePath);
    SavedInfo savedInfo(*this);
    savedInfo.serialize(popartFs);
  }
}

void Devicex::tryLoadExecutable() {
  auto warn = [&](const std::string &msg) {
    logging::devicex::warn("Unable to load cached poplar::Executable, {}", msg);
  };

  auto cachePath    = ir().getSessionOptions().cachePath;
  auto cacheEnabled = ir().getSessionOptions().enableEngineCaching;

  if (cacheEnabled && !cachePath.empty() &&
      deviceInfo->getType() == DeviceType::Ipu) {
    // load the popart ir hash
    auto popartCachePath = getPopartCachePath();
    std::ifstream popartFs(popartCachePath, std::ifstream::binary);
    if (popartFs.is_open()) {
      if (SavedInfo(*this) == SavedInfo::deserialize(popartFs)) {
        auto poplarCachePath = getPoplarCachePath();
        std::ifstream poplarFs(poplarCachePath, std::ifstream::binary);
        if (poplarFs.is_open()) {
          logging::devicex::trace("Loading poplar Executable from '{}'",
                                  cachePath);
          cachedExecutable.emplace(poplar::Executable::deserialize(poplarFs));
          usingCachedExecutable = true;
        } else {
          warn(logging::format("could not open file `{}'", poplarCachePath));
        }
      } else {
        warn("ir hashes differ");
      }
    } else {
      warn(logging::format("could not open file `{}'", popartCachePath));
    }
  }
}

TaskId Devicex::streamFromHostTaskId(TensorId id) const {
  return "streamFromHostTask_" + id;
}

TaskId Devicex::setInitTensorValTaskId(TensorId id) const {
  return "setInitTensorValTask_" + id;
}

TaskId Devicex::streamToHostTaskId(TensorId id, bool isAnchorStream) const {
  std::string anchorPrefix = isAnchorStream ? "anchor" : "weight";
  return anchorPrefix + "StreamToHostTask_" + id;
}

TaskId Devicex::fromHostTaskId(TensorId id) const {
  return "fromHostTask_" + id;
}

TaskId Devicex::toHostTaskId(TensorId id, bool isAnchorStream) const {
  if (isAnchorStream) {
    return "anchorToHostTask_" + id;
  }
  return "weightToHostTask_" + id;
}

TaskId Devicex::initBatchCounterTensorsTaskId() const {
  return "initBatchCounterTensorsTask";
}

TaskId Devicex::updateBatchCountTaskId() const {
  return "updateBatchCountTask";
}

TaskId Devicex::initRandomSeedTaskId() const { return "initRandomSeedTask"; }

TaskId Devicex::incrementRandomSeedTaskId() const {
  return "incrementRandomSeed";
}

TaskId Devicex::initTensorTaskId(TensorId id) const {
  return "initTensorTaskId_" + id;
}

TaskId Devicex::opTaskId(Op *op) const {

  std::stringstream ss;
  ss << "fromOpTask_" << op->id << '_' << op->opid;
  return ss.str();
}

TaskId Devicex::pipelinedCopyTaskId(Op *op) const {

  std::stringstream ss;
  ss << "pipelinedCopyTask_" << op->id << "_" << op->opid;
  return ss.str();
}

PopStreamId Devicex::h2dId(TensorId id) const { return "h2d_" + id; }

PopStreamId Devicex::d2hId(TensorId id, bool isAnchorStream) const {

  std::string anchorPrefix = isAnchorStream ? "anchor" : "weight";

  return anchorPrefix + "_d2h_" + id;
}

PriTask Devicex::fromHostTask(Tensor *tensor,
                              poplar::program::Sequence &streamSq) const {

  auto f = [&streamSq, tensor, this]() {
    logging::devicex::debug("Adding poplar::program::Copy from host " +
                            tensor->id);

    streamSq.add(poplar::program::Copy(fromHostStreams.at(tensor->id),
                                       tensors.get(tensor->id),
                                       doRearrangeOnHost(tensor)));
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
                            poplar::program::Sequence &sq,
                            bool isAnchorStream) const {

  auto f = [&sq, tensor, this, isAnchorStream]() {
    logging::devicex::debug(
        "Adding poplar::program::Copy to host (isAnchorStream = {}) " +
            tensor->id,
        isAnchorStream);

    auto pToHostStreams = &toHostAnchorStreams;
    if (!isAnchorStream) {
      pToHostStreams = &toHostWeightStreams;
    }

    sq.add(poplar::program::Copy(tensors.get(tensor->id),
                                 pToHostStreams->at(tensor->id),
                                 doRearrangeOnHost(tensor)));
  };

  auto finalPopulator = taskWhichPopulates(tensor->id);
  if (isAnchorStream && tensor->tensorType() == TensorType::Variable) {
    for (auto op : tensor->consumers.getOps()) {
      if (dynamic_cast<VarUpdateOp *>(op)) {
        finalPopulator = opTaskId(op);
      }
    }
  }

  auto taskId = toHostTaskId(tensor->id, isAnchorStream);

  logging::devicex::debug(
      "Final populator for {} is {} ", taskId, finalPopulator);

  return {+1e6, // writes to host: always as early as possible
          taskId,
          {
              // the dependencies:
              streamToHostTaskId(
                  tensor->id, isAnchorStream), // poplar::Stream creation task,
              finalPopulator // poplar::Tensor has its final values
          },
          f};
}

PriTask Devicex::initBatchCounterTensorsTask() {

  auto f = [this]() {
    logging::devicex::debug("Adding batch counter tensors");

    // Add scalar tensors outside of the ir to track the batch
    // Id and decide when to execute the copy to the host
    for (ReturnPeriod N : ir().getDataFlow().rps()) {
      // Add to map so copy task can access
      batchCountingTensors[N]      = graph().addVariable(poplar::INT, {});
      batchCountCheckingTensors[N] = graph().addVariable(poplar::BOOL, {});

      getConst(graph(), poplar::INT, {}, N, "batchCounter");

      poputil::mapTensorLinearly(graph(), batchCountingTensors[N]);
      poputil::mapTensorLinearly(graph(), batchCountCheckingTensors[N]);
    }

    // Make sure const 1 tensor exists
    getConst(graph(), poplar::INT, {}, 1, "one");
  };

  return {+1e6, // followed by writes to host: always as early as possible
          initBatchCounterTensorsTaskId(),
          {},
          f};
}

PriTask Devicex::updateBatchCountTask(poplar::program::Sequence &sq) {

  auto f = [&sq, this]() {
    logging::devicex::debug("Adding batch count checker program");

    // Placeholder 'do nothing' branch if not running assign program
    poplar::program::Sequence emptyseq;

    // Increment the batch count at the at the earliest point
    // the anchor tensor is required, and check if it is a
    // copy batch
    for (ReturnPeriod N : ir().getDataFlow().rps()) {
      popops::addInPlace(
          graph(),
          batchCountingTensors[N],
          getConst(graph(), poplar::INT, {}, 1, "batchCount/one"),
          sq);

      batchCountCheckingTensors[N] =
          popops::eq(graph(),
                     batchCountingTensors[N],
                     getConst(graph(), poplar::INT, {}, N, "batchCount/n"),
                     sq);

      // Reset batch count once it has reached N
      auto zero = getConst(graph(), poplar::INT, {}, 0, "batchCount/zero");
      sq.add(poplar::program::If(
          batchCountCheckingTensors[N],
          poplar::program::Copy(zero, batchCountingTensors[N]),
          emptyseq));
    }
  };

  return {+1e6, // followed by writes to host: always as early as possible
          updateBatchCountTaskId(),
          {
              initBatchCounterTensorsTaskId() // poplar::Tensor creation task
          },
          f};
}

std::map<PipelineStage, VGraphId> Devicex::getPipelineToVGraphIdMap() const {
  // Create a map of pipeline stage to virtual graph ids
  std::map<PipelineStage, VGraphId> pipeline_vgraph_map;
  for (auto &id_op : ir().getMainGraph().getOps()) {
    auto op = id_op.second.get();

    // Not sure why, but in test
    // 'pipeline_test.py::test_acts_match_restored_acts', an IpuCopy did not
    // have a virtual graph id set
    if (!op->isConvertibleTo<IpuCopyOp>()) {
      auto ps       = op->getPipelineStage();
      auto vgraphid = op->getVirtualGraphId();

      pipeline_vgraph_map[ps] = vgraphid;
    }
  }

  std::stringstream ss;
  ss << "Pipeline stages running on virtual graphs:";
  for (auto &ps_vgraph : pipeline_vgraph_map) {
    auto ps       = ps_vgraph.first;
    auto vgraphid = ps_vgraph.second;
    ss << logging::format("\n  ps {} on virtual graph {}", ps, vgraphid);
  }
  logging::devicex::debug(ss.str());

  return pipeline_vgraph_map;
}

PipelineStage Devicex::getMaxPipelineStage() const {
  PipelineStage max_ps = 0;

  for (auto &id_op : ir().getMainGraph().getOps()) {
    auto op = id_op.second.get();

    if (!op->isConvertibleTo<IpuCopyOp>()) {
      max_ps = std::max(max_ps, op->getPipelineStage());
    }
  }

  return max_ps;
}

PriTask Devicex::toHostEveryNBatchesTask(Tensor *tensor,
                                         int N,
                                         poplar::program::Sequence &sq) {

  auto f = [&sq, tensor, N, this]() {
    logging::devicex::debug(
        "Adding conditional poplar::program::Copy to host " + tensor->id);

    poplar::Tensor isNthBatch = batchCountCheckingTensors.at(N);

    poplar::program::Sequence copyseq;
    copyseq.add(poplar::program::Copy(tensors.get(tensor->id),
                                      toHostAnchorStreams.at(tensor->id),
                                      doRearrangeOnHost(tensor)));

    // Placeholder 'do nothing' branch if not running copy program
    poplar::program::Sequence emptyseq;

    sq.add(poplar::program::If(isNthBatch, copyseq, emptyseq));
  };

  bool isAnchorStream = true;
  return {
      +1e6, // writes to host: always as early as possible
      toHostTaskId(tensor->id, isAnchorStream),
      {
          // the dependencies:
          updateBatchCountTaskId(), // updating poplar::Tensor task,
          streamToHostTaskId(tensor->id,
                             isAnchorStream), // poplar::Stream creation task,
          taskWhichPopulates(tensor->id) // poplar::Tensor value setting task
      },
      f};
}

bool Devicex::doRearrangeOnHost(Tensor *tensor) const {
  if (tensor->tensorType() == TensorType::Variable) {
    return true;
  } else if (tensor->tensorType() == TensorType::Stream) {
    return false;
  } else if (ir().isAnchored(tensor->id)) {
    return ir().getSessionOptions().rearrangeAnchorsOnHost;
  }
  return true;
}

void Devicex::doProfileChecks() const {
  if (pEngine == nullptr) {
    throw error(
        "Session must have been prepared before a report can be fetched");
  }
  if (usingCachedExecutable) {
    throw error("Unable to get reports when using a cached executable.\n"
                "Either remove the cache file ({}), or \ndisable engine "
                "caching (userOptions.enableEngineCaching = false)",
                ir().getSessionOptions().cachePath);
  }
}

std::string Devicex::getSummaryReport() const {
  doProfileChecks();
  const auto &g_prof = pEngine->getGraphProfile();
  const auto &e_prof = pEngine->getExecutionProfile();

  std::stringstream ss;
  printProfileSummary(ss, g_prof, e_prof, reportOptions);

  pEngine->resetExecutionProfile();
  return ss.str();
}

std::string Devicex::getGraphReport(bool use_cbor) const {
  doProfileChecks();
  std::stringstream ss;
  auto report = pEngine->getGraphProfile();
  if (use_cbor) {
    serializeToCBOR(ss, report);
  } else {
    serializeToJSON(ss, report);
  }

  return ss.str();
}

std::string Devicex::getExecutionReport(bool use_cbor) const {
  doProfileChecks();
  std::stringstream ss;
  auto report = pEngine->getExecutionProfile();

  if (use_cbor) {
    serializeToCBOR(ss, report);
  } else {
    serializeToJSON(ss, report);
  }

  pEngine->resetExecutionProfile();
  return ss.str();
}

std::string Devicex::getSerializedGraph() const {
  doProfileChecks();
  std::stringstream ss;
  graph().serialize(ss, poplar::SerializationFormat::Binary);
  return ss.str();
}

TensorTileMap Devicex::getTensorTileMap() const {
  TensorTileMap map;

  for (const auto &t : tensors.getTensors()) {
    std::vector<TensorIntervalList> mapping;
    for (auto tile : graph().getTileMapping(t.second)) {
      TensorIntervalList intervalList;
      std::transform(tile.begin(),
                     tile.end(),
                     std::back_inserter(intervalList),
                     [](poplar::Interval i) {
                       return std::pair<size_t, size_t>(i.begin(), i.end());
                     });
      mapping.emplace_back(intervalList);
    }
    map.insert(std::make_pair(t.first, mapping));
  }
  return map;
}

void Devicex::saveTensorTileMap(const std::string &mapFileName) const {
  auto tt = getTensorTileMap();

  std::string finalPath =
      io::appendDirFn(ir().getSessionOptions().logDir, mapFileName);

  std::ofstream ofs(finalPath, std::ofstream::out);
  if (!ofs.is_open()) {
    throw error("Unable to open file '{}'", finalPath);
  }

  writeJSON(tt, ofs);
};

poplar::Type popType(const TensorInfo &info) {
  switch (info.dataType()) {
  case DataType::FLOAT: {
    return poplar::FLOAT;
  }
  case DataType::INT32: {
    return poplar::INT;
  }
  case DataType::FLOAT16: {
    return poplar::HALF;
  }
  case DataType::BOOL: {
    return poplar::BOOL;
  }
  case DataType::UINT32: {
    return poplar::UNSIGNED_INT;
  }

  case DataType::UNDEFINED:
  case DataType::UINT8:
  case DataType::INT8:
  case DataType::UINT16:
  case DataType::INT16:
  case DataType::INT64:
  case DataType::STRING:
  case DataType::BFLOAT16:
  case DataType::DOUBLE:
  case DataType::UINT64:
  case DataType::COMPLEX64:
  case DataType::COMPLEX128:
  default:
    throw error("Is there a poplar type for " + info.data_type() + "?");
  }
}

// using TensorInfo's data_type() function to get a string of the DataType
poplar::Type popType(DataType type) { return popType(TensorInfo(type, {1})); }

std::set<TensorId> Devicex::getLinearlyCreatedInputTensors() const {
  return linearlyCreatedInputTensors;
}
std::set<TensorId> Devicex::getEfficientlyCreatedInputTensors() const {
  return efficientlyCreatedInputTensors;
}

bool Devicex::useSyntheticData() const {
  return (ir().getSessionOptions().ignoreData);
}

std::string Devicex::randomSeedId() const { return "randomSeed"; }

const poplar::Tensor &Devicex::getRandomSeedTensor() const {
  return randomSeedTensor;
}

} // namespace popx
} // namespace popart
