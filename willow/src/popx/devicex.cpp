#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <tuple>

#include <boost/lexical_cast.hpp>
#include <boost/range/algorithm/find.hpp>
#include <boost/range/algorithm_ext.hpp>

#include <poplar/CSRFunctions.hpp>
#include <poplar/CycleCount.hpp>
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
#include <popart/op/cache.hpp>
#include <popart/op/call.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/op/if.hpp>
#include <popart/op/init.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/restore.hpp>
#include <popart/op/subgraphop.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/devicexmanager.hpp>
#include <popart/popx/op/callx.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/poplaroptionsx.hpp>
#include <popart/popx/pritask.hpp>
#include <popart/recompute.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensors.hpp>
#include <popart/tojson.hpp>
#include <popart/topocons.hpp>

#include <popart/op/hostreducevarupdate.hpp>
#include <popart/op/sgd1acclreduce.hpp>
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

  std::string getGraphReport(bool useCbor) const {

    if (exception.graphProfile.type() == poplar::ProfileValue::Type::MAP &&
        exception.graphProfile.size() != 0) {

      std::stringstream ss;
      if (useCbor) {
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

std::map<Op *, int, POpCmp> Devicex::getMainGraphOpSeriesNums() const {
  std::map<Op *, int, POpCmp> nums;
  int num = 0;
  for (auto entry : mainGraphOpRegistry) {
    for (auto op : entry.second) {
      auto found = nums.find(op);
      if (found == nums.end()) {
        nums.insert({op, num});
        ++num;
      }
    }
  }
  return nums;
}

std::string
Devicex::getMainGraphOpString(const std::vector<TaskId> &taskOrder) const {

  std::stringstream ss;
  auto seriesNums = getMainGraphOpSeriesNums();
  std::set<Op *> seen;
  for (auto taskId : taskOrder) {
    auto task_ops = mainGraphOpRegistry.find(taskId);
    if (task_ops == mainGraphOpRegistry.end())
      continue;
    for (auto op : task_ops->second) {
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
      if (logging::shouldLog(logging::Module::devicex, logging::Level::Trace)) {
        ss << type << "  " << seriesNums[op] << "  " << op->debugName()
           << "  PingPong: "
           << (op->hasPingPongPhase() ? op->getPingPongPhase() : -1)
           << "  Pipeline: "
           << (op->hasPipelineStage() ? op->getPipelineStage() : -1)
           << "  VGID: "
           << (op->hasVirtualGraphId() ? op->getVirtualGraphId() : -1)
           << "  priority: " << op->priority << std::endl;
      } else {
        ss << type << "  " << seriesNums[op] << "  " << op->str() << std::endl;
      }
    }
  }
  return ss.str();
}

std::map<Op *, int, POpCmp> Devicex::getMainGraphOpCounts() const {
  std::map<Op *, int, POpCmp> counts;
  for (auto entry : mainGraphOpRegistry) {
    for (auto op : entry.second) {
      auto found = counts.find(op);
      if (found == counts.end()) {
        counts.insert({op, 1});
      } else {
        ++found->second;
      }
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

  if (ir().useSyntheticData() == false) {
    logging::devicex::debug("Writing weights to host");
    pEngine->disableExecutionProfiling();
    // Weights on the IPU
    run(PopPrograms::ProgramIndex::WEIGHTSTOHOST);
    // Weights in the remote buffers
    remoteBufferWeightsToHost();
    logging::devicex::debug("Writing weights to host complete.");
  }
}

void Devicex::remoteBufferWeightsToHost() {
  for (auto initId : ir().getTensorIds(TensorType::Variable)) {
    Tensor *tensor = ir().getTensor(initId);
    if (tensor->isCached()) {
      logging::devicex::debug("remoteBufferWeightsToHost: {}", initId);
      auto remoteBufferInfo = tensor->getRemoteBufferInfo();
      char *data0           = d2hWeightBuffers[initId].data();
      // Weight should be the same for each replica, only return 0
      pEngine->copyFromRemoteBuffer(getRemoteBuffer(remoteBufferInfo.first),
                                    data0,
                                    remoteBufferInfo.second,
                                    0);
    }
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
  // Better to do this the other way round
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

  if (!prepareHasBeenCalled()) {
    throw error(
        "Devicex::prepare() must be called before Devicex::weightsToHost(const "
        "std::map<TensorId, MutableVoidData> &) is called.");
  }

  if (ir().useSyntheticData() == false) {
    logging::devicex::debug("Writing weights to host");
    // write weights from IPU to host stream memory points

    pEngine->disableExecutionProfiling();
    // Weights on the IPU
    run(PopPrograms::ProgramIndex::WEIGHTSTOHOST);
    // Weights in the remote buffers
    remoteBufferWeightsToHost();

    logging::devicex::debug("Writing weights to ONNX ModelProto");
    // copy from the host stream memory points to the
    // addresses on onnxModelData
    for (auto id : ir().getTensorIds(TensorType::Variable)) {
      if (!ir().streamingIsDisabledForTensor(id)) {
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
}

const std::string Devicex::cycleCountStreamId() const {
  return "d2h_" + std::string(cycleCountPrefix());
}

void Devicex::instrumentWithHardwareCycleCounter(
    poplar::program::Sequence &sq) {
  poplar::Tensor cycleCountTensor =
      poplar::cycleCount(graph(), sq, 0, cycleCountPrefix());

  // Create stream
  auto st = graph().addDeviceToHostFIFO(cycleCountStreamId(),
                                        cycleCountTensor.elementType(),
                                        cycleCountTensor.numElements());

  // Add program fragment to copy to host stream
  auto cyclesToHostStream = poplar::program::Copy(cycleCountTensor, st, true);
  progs.cycleCountTensorToHostFragment().add(cyclesToHostStream);
}

uint64_t Devicex::cycleCountTensorToHost() {
  if (ir().getSessionOptions().instrumentWithHardwareCycleCounter) {
    // Calls the copy from device to host
    logging::devicex::debug("Writing cycle count to host");
    pEngine->disableExecutionProfiling();
    run(PopPrograms::ProgramIndex::CYCLECOUNTTENSORTOHOST);
    logging::devicex::debug("Writing cycle count to host complete.");

    return cycleCount;
  } else {
    throw error("SessionOption 'instrumentWithHardwareCycleCounter' must be "
                "set to true in order to measure cycle count");
  }
}

poplar::Tensor Devicex::getConst(poplar::Graph &graph,
                                 const poplar::Type &type,
                                 const std::vector<size_t> &shape,
                                 double val,
                                 const std::string &name) {
  static unsigned tileCounter = 0;

  auto tensor     = graph.addConstant(type, shape, val, name);
  auto tilesTotal = graph.getTarget().getTilesPerIPU();
  auto tile       = tileCounter % tilesTotal;
  tileCounter++;

  graph.setTileMapping(tensor, tile);
  return tensor;
}

poplar::Tensor Devicex::getScalarVariable(poplar::Graph &graph,
                                          const poplar::Type &type,
                                          const std::string &name) {
  static int tileCounter = -1;

  auto tensor     = graph.addVariable(type, {}, name);
  auto tilesTotal = graph.getTarget().getTilesPerIPU();
  auto tile       = (tilesTotal + (tileCounter % tilesTotal)) % tilesTotal;
  tileCounter--;

  graph.setTileMapping(tensor, tile);
  return tensor;
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
    throw internal_error(
        "mainCycles should not be less than 1. Current value is {}.",
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
      prepareHasBeenCalled_(false) {

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
  if (ir().useSyntheticData() == false) {
    logging::devicex::debug("Writing weights from host, ");
    pEngine->disableExecutionProfiling();
    // Weights on the IPU
    run(PopPrograms::ProgramIndex::WEIGHTSFROMHOST);
    // Weights in the remote buffers
    remoteBufferWeightsFromHost();
    logging::devicex::debug("done.");
  }
}

void Devicex::remoteBufferWeightsFromHost() {
  for (auto initId : ir().getTensorIds(TensorType::Variable)) {
    Tensor *tensor = ir().getTensor(initId);
    if (tensor->isCached()) {
      logging::devicex::debug("remoteBufferWeightsFromHost: {}", initId);
      char *data0           = static_cast<char *>(tensor->tensorData()->data());
      auto remoteBufferInfo = tensor->getRemoteBufferInfo();
      for (unsigned replica_id = 0; replica_id < getReplicationFactor();
           ++replica_id) {
        // Weights to every replica
        pEngine->copyToRemoteBuffer(data0,
                                    getRemoteBuffer(remoteBufferInfo.first),
                                    remoteBufferInfo.second,
                                    replica_id);
      }
    }
  }
}

void Devicex::optimizerFromHost() {
  if (ir().useSyntheticData() == false) {
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

  if (ir().useSyntheticData() == false) {
    std::string prefix = "     ";
    logging::devicex::debug(prefix + "Copying to h2d stream address(es) ");
    for (Tensor *tensor : ir().dataStreamTensors()) {
      inputStreams.at(tensor->id)->setStepIO(&stepio);
    }
  }
}

void Devicex::anchorsHostFromHostStreams(IStepIO &stepio) {

  if (ir().useSyntheticData() == false) {
    std::string prefix = "     ";
    logging::devicex::debug(prefix + "Copying from d2h stream address(es) ");
    for (TensorId anchorId : ir().getDataFlow().anchors()) {
      outputStreams.at(anchorId)->setStepIO(&stepio);
    }
  }
}

void Devicex::run(IStepIO &stepio) {
  if (!prepareHasBeenCalled()) {
    throw error("Devicex::prepare() must be called before"
                " Devicex::run(const IStepIO &) is called.");
  }

  // Check that the input and output buffers have the correct number of
  // elements. As run(.) is called multiple times during a user's session, the
  // check is only performed in the first call to run, under the assumption that
  // the user is unlikely to change the size of buffers between runs.
  if (nCallsToRun == 0 && stepio.runtimeAssertsEnabled()) {
    stepio.assertNumElements(ir());
  }

  logging::devicex::debug("Performing one step: ");

  // Reconnect input streams.
  reconnectInputStreams();

  // Configure the inputstreams
  anchorsHostToHostStreams(stepio);

  // Configure the outputstreams
  anchorsHostFromHostStreams(stepio);

  pEngine->enableExecutionProfiling();
  run(PopPrograms::ProgramIndex::PROGRAM);

  ++nCallsToRun;
}

std::unique_ptr<Opx> Devicex::createOpx(Op *op) {

  auto opx = OpxManager::createOpx(op, this);

  if (!opx) {
    if (op->opid == Onnx::Operators::Constant_1 ||
        op->opid == Onnx::Operators::Constant_9) {
      throw internal_error("No Opx for {}", op->opid);
    } else {
      auto pattern = PreAliasPatternManager::opReplacementPattern(op);
      if (pattern != "") {
        throw error("Could not create opx for '{}'. This op should have been "
                    "removed by pattern {}",
                    op->opid,
                    pattern);
      } else {
        throw error("Could not create opx for '{}' and there were no patterns "
                    "intended to remove it. Please check it is defined and "
                    "registered correctly",
                    op->opid);
      }
    }
  }

  return opx;
}

Opx *Devicex::getOpx(OpId id) { return opxs.at(id).get(); }

const Opx *Devicex::getOpx(OpId id) const { return opxs.at(id).get(); }

// The Id of the task which adds a Tensor to a poplar::Graph
std::pair<TaskId, DependencyType> Devicex::taskWhichCreates(TensorId id) {
  Tensor *tensor = ir().getTensor(id);
  // Tensors without producers, or produced by an InitOp are created by special
  // tasks These tasks are a TENSOR rather than an OUTPUT dependency.
  if (!tensor->hasProducer() || dynamic_cast<InitOp *>(tensor->getProducer())) {
    return {initTensorTaskId(id), DependencyType::TENSOR};
  }

  // Tensors with producer Ops are created (added to a Graph) by their
  // producer's OpTask
  else {
    return {opTaskId(tensor->getProducer()), DependencyType::OUTPUT};
  }
}

TaskId Devicex::taskWhichPopulates(TensorId id) const {
  Tensor *tensor = ir().getTensor(id);

  // OpTasks both initialize a Tensor, and generate the code to set its value
  if (tensor->hasProducer()) {
    return opTaskId(tensor->getProducer());
  }

  // if a Tensor is of type Stream, the Copy from host to device populates it
  else if (!ir().useSyntheticData() &&
           tensor->tensorType() == TensorType::Stream) {
    return fromHostTaskId(tensor->id);
  }

  // default:
  else {
    return initTensorTaskId(id);
  }
}

std::pair<std::vector<ICreatorCandidatePtr>, std::vector<UnwindEndpointPtr>>
Devicex::getCreatorEndpoints(Tensor *tensor,
                             std::vector<OpxInAndOutIndex> pathFromInput,
                             bool excludeEndpointsFromPath,
                             bool includeDeadends) const {

  std::vector<ICreatorCandidatePtr> endpoints;
  std::vector<UnwindEndpointPtr> endpointsUnwind;

  // For tensors created within a subgraph:
  // Detect graph outputs and try to propagate out through the first call site
  for (OutIndex o = 0; o < tensor->getGraph().getOutputIds().size(); ++o) {
    TensorId graph_output_tensor_id = tensor->getGraph().getOutputId(o);
    if (graph_output_tensor_id == tensor->id) {
      Op *op       = tensor->getGraph().getCallSiteOps(1).front();
      auto opx     = getOpx(op->id);
      bool visited = false;
      // Escape route only allowed if this call op has not been visited yet
      for (auto &elem : pathFromInput) {
        if (elem.isDelegate && opx == elem.opx) {
          visited = true;
        }
      }
      if (!visited) {
        auto updatedPath = pathFromInput;

        // Mark as delegate visited Opx on path
        updatedPath.push_back({opx});

        Tensor *nextOutputTensor = op->output->tensor(o);

        logging::devicex::trace("Subgraph escape path: {} -> {}",
                                graph_output_tensor_id,
                                nextOutputTensor->id);

        // Continue path recursion behind the subgraph
        auto candidates = getCreatorEndpoints(nextOutputTensor, updatedPath);
        for (auto &candidate : candidates.first) {
          endpoints.push_back(candidate);
        }
        for (auto &candidate : candidates.second) {
          endpointsUnwind.push_back(candidate);
        }
      }
    }
  }

  for (Op *op : tensor->consumers.getOps()) {
    auto conOpId   = op->id;
    const Opx *opx = getOpx(conOpId);

    for (InIndex inIndex : op->input->indices(tensor)) {
      auto updatedPath = pathFromInput;

      auto f_create = [&]() {
        if (!excludeEndpointsFromPath) {
          // note: no valid outIndex
          updatedPath.push_back({opx, inIndex, -1});
        }
        endpoints.push_back(
            std::make_shared<InputCreatorCandidate>(inIndex, opx, updatedPath));
      };

      auto f_unwind = [&]() {
        for (auto &ind_ten : op->output->tensorMap()) {
          auto nextOutputTensor = ind_ten.second;
          auto outIndex         = ind_ten.first;
          if (opx->canUnwind(inIndex, outIndex)) {
            updatedPath.push_back({opx, inIndex, outIndex});
            auto candidates =
                getCreatorEndpoints(nextOutputTensor, updatedPath);
            for (auto &candidate : candidates.first) {
              endpoints.push_back(candidate);
            }
            // Record unwind endpoints which are graph outputs
            for (TensorId tensor_id : op->getGraph().getOutputIds()) {
              if (tensor_id == nextOutputTensor->id) {
                endpointsUnwind.push_back(std::make_shared<UnwindEndpoint>(
                    op->getGraph(), nextOutputTensor, updatedPath));
              }
            }
          }
        }
      };

      auto f_deadend = [&]() {
        if (includeDeadends) {
          if (!excludeEndpointsFromPath) {
            updatedPath.push_back(
                {opx, inIndex, -1}); // note: no valid outIndex
          }
          endpoints.push_back(std::make_shared<InputCreatorCandidate>(
              inIndex, opx, updatedPath));
        }
      };

      // TODO: T13654 Generalize for other subgraphing ops (if, loop).
      // Create common base class for Loop, If, Call
      auto f_delegate = [&]() {
        // Mark as delegate visited Opx on path
        updatedPath.push_back({opx});

        const CallOpx *callopx = dynamic_cast<const CallOpx *>(opx);
        // Get delegated endpoints
        auto delegateEndpoints = callopx->getEndpoints(inIndex, updatedPath);

        // Endpoints contributing to tensor creation
        for (auto endpoint : delegateEndpoints.first) {
          endpoints.push_back(endpoint);
        }

        // Endpoints that unwind, but don't create -> propagate recursively
        for (auto endpoint : delegateEndpoints.second) {

          // This op is the last unwinder along the delegate path
          // auto last = endpoint->getPathsFromInput().front().back();
          // Op* lastOp = last.opx->getOp<Op>();

          // The delegate path is inside a subgraph, check subgraph outputs
          auto graphOutputIds = endpoint->graph.getOutputIds();
          for (OutIndex o = 0; o < graphOutputIds.size(); ++o) {
            if (graphOutputIds[o] == endpoint->tensor->id) {
              // Get delegate output tensor corresponding to subgraph output
              Tensor *nextOutputTensor = op->output->tensor(o);
              // Continue path recursion behind the subgraph
              auto candidates = getCreatorEndpoints(nextOutputTensor,
                                                    endpoint->pathFromInput);
              for (auto &candidate : candidates.first) {
                endpoints.push_back(candidate);
              }

              // Record delegate endpoints which are graph outputs
              for (TensorId tensor_id : op->getGraph().getOutputIds()) {
                if (tensor_id == nextOutputTensor->id) {
                  endpointsUnwind.push_back(std::make_shared<UnwindEndpoint>(
                      op->getGraph(),
                      nextOutputTensor,
                      endpoint->pathFromInput));
                }
              }
            }
          }
        }
      };

      switch (opx->getInputCreatorType(inIndex)) {
      // Opx has poplar call to layout tensor at this
      // inIndex
      case InputCreatorType::CANCREATE: {
        logging::devicex::trace("{} can create, path depth {}",
                                op->debugName(),
                                updatedPath.size());
        f_create();
        break;
      }
      case InputCreatorType::CANDELEGATE: {
        logging::devicex::trace("{} can delegate, path depth {}",
                                op->debugName(),
                                updatedPath.size());
        f_delegate();
        break;
      }
      // Recursively search the DAG downstream of the op until we
      // have set of endpoints that can create the tensor
      case InputCreatorType::CANUNWIND: {
        logging::devicex::trace("{} can unwind, path depth {}",
                                op->debugName(),
                                updatedPath.size());
        f_unwind();
        break;
      }
      case InputCreatorType::CANCREATE_OR_UNWIND: {
        logging::devicex::trace("{} can create or unwind , path depth {}",
                                op->debugName(),
                                updatedPath.size());
        f_create();
        f_unwind();
        break;
      }
      // Consuming op can't create tensor
      case InputCreatorType::DEADEND: {
        f_deadend();
        break;
      }
      default: {
        throw error("InputCreatorType not implemented for Opx of OpId {}",
                    op->id);
      }
      }
    }
  }
  return {endpoints, endpointsUnwind};
}

ICreatorCandidatePtr Devicex::getTensorCreator(Tensor *tensor) const {
  // Search of the graph to get the candidate Opxs that
  // know how to create this tensor.
  // The pathFromInput argument is an empty vector, as
  // we are starting the search from the root (input)

  logging::devicex::trace("Get tensor creator for {}, {} elements",
                          tensor->id,
                          tensor->info.nelms());

  std::vector<ICreatorCandidatePtr> candidates =
      getCreatorEndpoints(tensor, {}).first;

  logging::devicex::trace(
      "{} creator candidate(s) for {}", candidates.size(), tensor->id);

  if (candidates.size() > 0) {
    std::sort(
        candidates.begin(), candidates.end(), ICreatorCandidate::greaterThan);

    if (candidates.front()->getNumElems() == tensor->info.nelms()) {
      logging::devicex::trace("Candidate {} creates tensor alone.",
                              candidates.front()->str());
      // A single top-priority candidate can initialize the tensor fully.
      return candidates.front();
    } else {
      logging::devicex::trace("Multiple candidates needed.");
      // Multiple creators need to be concatenated to form the full tensor.
      std::shared_ptr<InputMultiCreatorCandidate> multiCandidate =
          std::make_shared<InputMultiCreatorCandidate>();
      for (auto candidate : candidates) {
        // Important to add candidates sorted by priorty.
        // Highest first - ICreatorCandidate::greaterThan.
        multiCandidate->addCreatorCandidate(candidate);
      }
      logging::devicex::trace("Using multi-candidate {}.",
                              multiCandidate->str());
      return multiCandidate;
    }
  } else {
    logging::devicex::trace("No suitable candidate.");
    return nullptr;
  }
}

// Design decision : leave the option for a Tensor to be
// created based on complex global criteria open.
PriTask Devicex::initTensorTask(Tensor *tensor) {
  auto candidate = getTensorCreator(tensor);

  // Initialising priMod to 0.0 since there is no guarantee that the tensor
  // to be initialised has a consumer in it's graph.
  double priMod = 0.0;

  // Design decision:
  // The order in which these initTensorTasks are run affects tensor
  // layout. This can affect the behaviour of random operations, as well
  // as overall memory consumption.
  // Let's fix the order of execution of a set of initTensorTasks, based
  // on some condition. We do this here by giving it a unique priority
  // based on:
  // - 0th consumer id
  // - the index at which it is consumed by this consumer
  if (tensor->consumers.getOps().size() > 0) {
    // Tensor does have consumers
    auto firstConsumer = tensor->consumers.getOps().front();
    auto priMod0       = firstConsumer->id;
    auto priMod1       = firstConsumer->input->indices(tensor).front();
    // assumes an op has max 1000 inputs
    priMod =
        static_cast<double>(priMod0) + static_cast<double>(priMod1) / 1000.0;
  }

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
      return SequenceMap();
    };
    // the inputs of creator which must have poplar::Tensors
    // before creator creates input tensor at index inIndex.
    std::vector<std::pair<TaskId, DependencyType>> deps;
    for (TensorId tenId : candidate->mustExistBeforeCreate()) {
      auto dep = taskWhichCreates(tenId);
      deps.push_back(dep);
    }

    // Discussion with David Norman suggests creating tensors as
    // late as possible gives better IPU memory use, so
    // giving this low priority.
    return {-1e6 + priMod,
            initTensorTaskId(tensor->id), // the task name
            deps,
            f};
  } else {

    auto f = [this, tensor]() {
      logging::devicex::debug("Creating poplar::Tensor '{}' linearly. No "
                              "operator specific allocator found",
                              tensor->id);

      // Find the ipu the op that consumes with tensor is on and create the
      // tensor on that graph
      auto consumerOps = tensor->consumers.getOps();

      std::vector<std::pair<Op *, bool>> relatedOps;
      relatedOps.reserve(consumerOps.size() + 1);

      for (auto *op : consumerOps) {
        relatedOps.emplace_back(op, true);
      }

      if (tensor->hasProducer()) {
        relatedOps.emplace_back(tensor->getProducer(), false);
      }

      if (logging::shouldLog(logging::Module::devicex, logging::Level::Trace)) {
        std::stringstream ss;
        for (auto &op : relatedOps) {
          if (op.second) {
            auto index = op.first->input->indicesMap().at(tensor)[0];
            ss << std::endl
               << "    {" << op.first->debugName() << ", VGID: "
               << (op.first->hasVirtualGraphId()
                       ? op.first->getIntrospectionInVirtualGraphId(index)
                       : -1)
               << "}";
          }
        }
        logging::devicex::trace(
            "[initTensorTask] Tensor: {}, type: {}, consumed by ops: [{}]",
            tensor->id,
            tensor->getTensorTypeInfo()->type_s(),
            ss.str());
      }

      std::vector<VGraphId> ipus;
      for (auto &op : relatedOps) {
        VGraphId vgid = -1;
        if (op.first->hasVirtualGraphId()) {
          if (op.second) {
            // Consumer OP
            // VirtualGraphId with subgraph call introspection
            // for the current tensor
            auto index = op.first->input->indicesMap().at(tensor)[0];
            vgid       = op.first->getIntrospectionInVirtualGraphId(index);
          } else {
            // Producer OP
            vgid = op.first->getVirtualGraphId();
          }
        }

        // The copyToIpu op assume that the tensor will already
        // have been copied to the ipu from another op
        if (op.first->opid != Onnx::CustomOperators::IpuCopy) {

          if (ipus.end() == std::find(ipus.begin(), ipus.end(), vgid)) {

            auto &graph = vgid > -1 ? getVirtualGraph(vgid)
                                    : getOpx(op.first->id)->graph();

            auto newTensor = graph.addVariable(
                popType(tensor->info), tensor->info.shape_szt(), tensor->str());
            linearMapper.mapTensor(graph, newTensor);

            tensors.insert(tensor->id, newTensor);
            linearlyCreatedInputTensors.insert(tensor->id);
            ipus.push_back(vgid);
          }
        }
      }
      return SequenceMap();
    };

    // Discussion with David Norman suggests creating tensors as
    // late as possible gives better IPU memory use, so
    // giving this low priority.
    return {-1e6 + priMod, initTensorTaskId(tensor->id), {}, f};
  }
}

PriTask Devicex::initRandomSeed() {
  auto streamedSeedId = GetRandomSeedOp::getStreamedSeedTensorId();
  auto updatedSeedId  = GetRandomSeedOp::getUpdatedSeedTensorId();

  auto initRandomSeedTask = [this, updatedSeedId]() {
    logging::devicex::debug("Initializing random seed.");
    SequenceMap seqs;
    poprand::setSeed(graph(),
                     tensors.get(updatedSeedId),
                     0,
                     seqs[&progs.setRandomSeedFromHostFragment()],
                     logging::format("{}/set", updatedSeedId));
    return seqs;
  };

  std::vector<std::pair<TaskId, DependencyType>> deps;
  deps.push_back(taskWhichCreates(updatedSeedId));
  // Stream the seed tensor to device before using to set PRNGs
  deps.push_back({fromHostTaskId(streamedSeedId), DependencyType::SCHEDULER});

  return {
      +1e6,                   // high priority
      initRandomSeedTaskId(), // name of this task
      deps,                   // depends on
      initRandomSeedTask      // what to run when the task is executed
  };
}

void Devicex::connectRandomSeedStream() {
  // Generate a separate random seed for each replicant.
  for (uint16_t replicaId = 0; replicaId < getReplicationFactor();
       ++replicaId) {

    auto callback = [this, replicaId](void *ptr) {
      TensorId seedId    = GetRandomSeedOp::getStreamedSeedTensorId();
      Tensor *seedTensor = ir().getTensor(seedId);
      uint64_t *seedVal =
          reinterpret_cast<uint64_t *>(seedTensor->tensorData()->data());
      logging::devicex::debug(
          "Updating random seed for replica:{} to `{} + replicaId({})'",
          replicaId,
          *seedVal,
          replicaId);
      uint64_t *data = reinterpret_cast<uint64_t *>(ptr);
      data[0]        = *seedVal + replicaId;
    };

    pEngine->connectStreamToCallback(
        h2dId(GetRandomSeedOp::getStreamedSeedTensorId()), replicaId, callback);
  }
}

void Devicex::setRandomSeedFromHost() {
  if (ir().useSyntheticData() == false) {
    pEngine->disableExecutionProfiling();
    run(PopPrograms::ProgramIndex::SETRANDOMSEEDFROMHOST);
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
    case DataType::UINT32: {
      setInitVal<uint32_t>(tensor);
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
    return SequenceMap();
  };

  return {// priority unimportant
          0,
          // name of this task
          setInitTensorValTaskId(tensor->id),
          // poplar::Tensor must exist. Other that this, this task can be
          // performed any time
          {{initTensorTaskId(tensor->id), DependencyType::TENSOR}},
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
    return SequenceMap();
  };

  return {
      0,                                // priority unimportant
      streamFromHostTaskId(tensor->id), // name of this task
      // poplar::Tensor must exist
      {{initTensorTaskId(tensor->id), DependencyType::TENSOR}},
      f // what to run when the task is executed
  };
}

PriTask Devicex::streamToHostTask(Tensor *tensor, bool isAnchorStream) {
  auto f = [this, tensor, isAnchorStream]() {
    logging::devicex::debug("Creating device-to-host FIFO for poplar::Tensor "
                            "{} (isAnchorStream = {}) with {} elements",
                            tensor->id,
                            isAnchorStream,
                            tensor->info.nelms());

    auto pToHostStreams = &toHostAnchorStreams;
    if (!isAnchorStream) {
      pToHostStreams = &toHostWeightStreams;
    }

    pToHostStreams->emplace(
        tensor->id,
        graph().addDeviceToHostFIFO(d2hId(tensor->id, isAnchorStream),
                                    popType(tensor->info),
                                    tensor->info.nelms()));
    return SequenceMap();
  };

  return {
      0,                                              // priority unimportant
      streamToHostTaskId(tensor->id, isAnchorStream), // name of this task
      {taskWhichCreates(tensor->id)}, // poplar::Tensor must exist
      f                               // what to run when the task is executed,
  };
}

const poplar::RemoteBuffer &Devicex::getRemoteBuffer(RemoteBufferId id) const {
  return remoteBuffers.at(id);
}

void Devicex::createRemoteBuffers() {
  for (auto info : ir().getAllRemoteBufferInfos()) {
    auto name    = "RB_" + std::to_string(info.first);
    auto type    = popType(info.second.info);
    auto size    = info.second.info.nelms();
    auto repeats = info.second.repeats;

    logging::devicex::info(
        "Creating remote buffer {}, type {}, size {}, repeats {}",
        name,
        type,
        size,
        repeats);

    remoteBuffers.insert(
        {info.first, graph().addRemoteBuffer(name, type, size, repeats, true)});
  }
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
    SequenceMap seqs;
    logging::debug("Adding pipelined copies for op {}", copyOp->debugName());
    for (auto &prog : progs.pipelineIpuCopyFragments(
             copyOp->getPipelineStage(),
             logging::format("{}, {}, PipelineStage({})",
                             copyOp->debugName(),
                             copyOp->getFromToStr(),
                             copyOp->getPipelineStage()))) {
      copyOpx->growPipelined(seqs[prog]);
    }
    return seqs;
  };

  std::vector<std::pair<TaskId, DependencyType>> deps;
  if (!prevTaskId.empty()) {
    // Ensure the ops are scheduled in the order we're iterating through them
    // here.
    deps.push_back({prevTaskId, DependencyType::SCHEDULER});
  }

  // The ops opTask needs to run first to create the destination tensor.
  deps.push_back({opTaskId(op), DependencyType::OUTPUT});

  return {-100, pipelinedCopyTaskId(op), deps, f};
}

void Devicex::addOpTasks(PriTasks &tasks) {

  // Ensure there is a program fragment for every Ir Graph
  logging::devicex::trace("[addOpTasks] Graphs: {}",
                          ir().getGraphSchedule().size());
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
    logging::devicex::trace("[addOpTasks] Adding graph {}", graph->id);
    addedGraphs.insert(graph);

    // Add each op in the graph
    for (auto op : graph->getOpSchedule({})) {
      // If the op calls another graph, then
      // the ops in that graph should be scheduled first
      for (auto calledGraph : op->getCalledGraphs()) {
        addGraph(calledGraph);
      }
      if (op->settings.recomputeType == RecomputeType::RECOMPUTE) {
        throw internal_error("non-main Graph Op which is RECOMPUTE");
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
          if (graph->isMarkedAsZeroCopy(graphInput)) {
            tasks.add(initTensorByAliasingTask(op, opInputs.at(i), graphInput));
          } else {
            tasks.add(initTensorByCloningTask(op, opInputs.at(i), graphInput));
          }
        }
      }

      auto opOutputs = getOpx(op->id)->getOutputsToPrepare();
      for (int i = 0; i < opOutputs.size(); i++) {
        auto opOutput = opOutputs[i];
        if (!tasks.contains(initTensorTaskId(std::get<1>(opOutput)))) {
          if (std::get<2>(opOutput)) {
            tasks.add(initTensorByAliasingTask(
                op, std::get<0>(opOutput), std::get<1>(opOutput)));
          } else {
            tasks.add(initTensorByCloningTask(
                op, std::get<0>(opOutput), std::get<1>(opOutput)));
          }
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
    auto dst = opx->graph().clone(src, dstId);
    tensors.insert(dstId, dst);
    return SequenceMap();
  };

  std::vector<std::pair<TaskId, DependencyType>> deps;
  auto creatorTask = taskWhichCreates(srcId);
  deps.push_back(creatorTask);

  return {-1e6, initTensorTaskId(dstId), deps, f};
}

PriTask
Devicex::initTensorByAliasingTask(Op *op, TensorId srcId, TensorId dstId) {
  Opx *opx = getOpx(op->id);

  auto f = [srcId, dstId, opx, this]() {
    logging::debug("Aliasing tensor {} to {}", srcId, dstId);
    auto src = opx->get(srcId);
    tensors.insert(dstId, src);
    return SequenceMap();
  };

  std::vector<std::pair<TaskId, DependencyType>> deps;
  auto creatorTask = taskWhichCreates(srcId);
  deps.push_back(creatorTask);

  return {-1e6, initTensorTaskId(dstId), deps, f};
}

PriTask Devicex::opTask(Op *op, double priority, TaskId prevOpTaskId) {
  // although priority should guarantee that this
  // task is only run after inputs are all created,
  // we add a dependency to the input tensors, just
  // in case someone plays with the priorities.
  // Moreover, we must state the copy-from-host deps
  std::vector<std::pair<TaskId, DependencyType>> deps;
  for (auto t_inds : op->input->indicesMap()) {
    Tensor *tensor = t_inds.first;

    std::pair<TaskId, DependencyType> creatorTask =
        taskWhichCreates(tensor->id);

    std::pair<TaskId, DependencyType> populatorTask = {
        taskWhichPopulates(tensor->id), DependencyType::SCHEDULER};

    // Make sure we only add the creatorTask once in the dependency list
    if (std::find(deps.begin(), deps.end(), creatorTask) == deps.end()) {
      deps.push_back(creatorTask);
    }
    if (std::find(deps.begin(), deps.end(), populatorTask) == deps.end()) {
      deps.push_back(populatorTask);
    }
  }

  // For InitOp and SubgraphOp,
  // the output tensor is created externally, and must
  // thereby exist before InitOp/SubgraphOp is grown.
  if (dynamic_cast<InitOp *>(op) || dynamic_cast<SubgraphOp *>(op)) {
    for (auto t_inds : op->output->indicesMap()) {
      Tensor *tensor = t_inds.first;

      logging::devicex::trace("Operation {} depends on it's output tensor {} "
                              "being externally created.",
                              op->debugName(),
                              tensor->id);

      std::pair<TaskId, DependencyType> creatorTask = {
          initTensorTaskId(tensor->id), DependencyType::TENSOR};

      // Make sure we only add the creatorTask once in the dependency list
      if (std::find(deps.begin(), deps.end(), creatorTask) == deps.end()) {
        deps.push_back(creatorTask);
      }
    }
  }

  auto addGraphOpsToDeps = [&](const Graph *graph) {
    for (auto graphOp : graph->getOpSchedule({})) {
      std::pair<TaskId, DependencyType> taskId = {opTaskId(graphOp),
                                                  DependencyType::SUBGRAPH};
      if (std::find(deps.begin(), deps.end(), taskId) == deps.end()) {
        deps.push_back(taskId);
      }
    }
  };

  for (auto &graph : op->getCalledGraphs()) {
    addGraphOpsToDeps(graph);
  }

  // Depends on previous op task. This preserves op ordering from ir.
  // Note: the first opTask has no previous opTask
  if (prevOpTaskId != "") {
    std::pair<TaskId, DependencyType> prevTask = {prevOpTaskId,
                                                  DependencyType::SCHEDULER};
    // Add dependency only if not already added
    if (std::find(deps.begin(), deps.end(), prevTask) == deps.end()) {
      deps.push_back(prevTask);
    }
  }

  auto f = [op, this]() {
    SequenceMap seqs;
    const auto &containingGraph = op->getGraph();
    // if this Op is not in the main scope
    if (!containingGraph.id.str().empty()) {
      Opx *opx = getOpx(op->id);
      logging::devicex::debug("Creating output tensors for non-main {} in {}",
                              opx->op_p->debugName(),
                              containingGraph.id.str());
      // Record each scope task fragment separately first.
      growOpx(opx, seqs[&progs.scopeFragment(containingGraph)]);
    } else if (ir().getSessionOptions().enablePipelining) {
      pipelinedOpTaskFunc(opTaskId(op), op, seqs);
    } else {
      opTaskFunc(opTaskId(op), op, seqs);
    }
    return seqs;
  };

  return {priority, opTaskId(op), deps, f};
}

namespace {

// Walk the producers of an ops inputs, applying function f to every producer.
// The producers are walked in a top down fashion. If f returns false of an op,
// then further producers below it are not traversed.
template <typename Predicate> void walkProducers(Op *op, Predicate f) {
  std::vector<Op *> toCheck;
  std::set<Op *> seen;

  auto addProducers = [&toCheck, &seen](Op *x) {
    for (auto t : x->input->tensors()) {
      if (t->hasProducer()) {
        auto p = t->getProducer();
        if (seen.find(p) == seen.end()) {
          toCheck.push_back(p);
          seen.insert(p);
        }
      }
    }
  };

  addProducers(op);
  while (!toCheck.empty()) {
    auto x = toCheck.back();
    toCheck.pop_back();
    if (f(x)) {
      addProducers(x);
    }
  }
}

} // namespace

void Devicex::growOpx(Opx *opx, poplar::program::Sequence &seq) {
  logging::devicex::trace("Calling growOpx for Op {} with debugName {}",
                          opx->op_p->str(),
                          opx->op_p->debugName());

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

void Devicex::opTaskFunc(TaskId taskId, Op *op, SequenceMap &seqs) {
  Opx *opx = getOpx(op->id);

  if (op->copiesOptimizerTensors()) {
    growOpx(opx, seqs[&progs.streamOptimizerFromHostFragment()]);
  }

  // pre-loss : create vertices for all recompute types
  else if (op->scheduledPreLoss == ScheduledPreLoss::Yes) {

    // Pre-loss, not recompute
    if (op->settings.recomputeType == RecomputeType::CHECKPOINT ||
        op->settings.recomputeType == RecomputeType::UNDEFINED) {
      logging::devicex::debug("Adding checkpoint Op {}", op->debugName());
      growOpx(opx, seqs[&progs.forwardFragment()]);
    }

    // Pre-loss, recompute
    else if (op->settings.recomputeType == RecomputeType::RECOMPUTE) {
      logging::devicex::debug("Adding (first) recompute Op {}",
                              op->debugName());

      growOpx(opx, progs.recomputeFragment(op->id));
      seqs[&progs.forwardFragment()].add(progs.recomputeFragment(op->id));
    }

    // Pre-loss, not recompute or checkpoint
    else {
      throw internal_error("Unrecognised recompute type");
    }
    mainGraphOpRegistry[taskId].push_back(op);
  }

  // post-loss
  else if (op->scheduledPreLoss == ScheduledPreLoss::No) {
    if (op->settings.recomputeType == RecomputeType::RECOMPUTE) {
      std::stringstream oss;
      op->append(oss);
      throw internal_error("Recompute Op which is ScheduledPreLoss::No is "
                           "not permitted: \n{}",
                           oss.str());
    }

    // 2 special case Ops when their is a gradient accumulator / velocity.
    // If we are doing gradient accumulation, we need to ensure the reset
    // and var update aren't run every time. Instead, these fragments sit
    // outside the "main" loop of the fowards and backwards passes.
    // special case Op 1:
    if (ir().getSessionOptions().enableGradientAccumulation &&
        (dynamic_cast<SGD1AcclReduceOp *>(op) ||
         dynamic_cast<SGD1VarUpdateOp *>(op) ||
         dynamic_cast<SGD1AcclUpdateOp *>(op) ||
         dynamic_cast<GradCopyToHostOp *>(op) ||
         dynamic_cast<GradCopyFromHostOp *>(op))) {
      outerLoopFragEmpty = false;
      growOpx(opx, seqs[&progs.accumulateOuterFragment()]);
    }

    // post-loss, not special gradient accumulation case,
    else {
      std::set<Op *> toRerun;
      walkProducers(op, [&toRerun, this, op](Op *x) {
        bool samePingPongPhase =
            (this->ir().getSessionOptions().pingPongPhases >= 2 &&
             op->hasPingPongPhase() && x->hasPingPongPhase() &&
             op->getPingPongPhase() == x->getPingPongPhase());
        PingPongPhase phase =
            op->hasPingPongPhase() &&
                    this->ir().getSessionOptions().pingPongPhases >= 2
                ? op->getPingPongPhase()
                : -1;
        if (x->settings.recomputeType == RecomputeType::RECOMPUTE &&
            !progs.hasBeenRecomputed(x->id, phase) && !samePingPongPhase) {
          toRerun.insert(x);
          return true;
        } else {
          return false;
        }
      });

      // The ops to rerun in topological order.
      auto rerunSchedule = op->getGraph().getOpSchedule({});
      boost::remove_erase_if(rerunSchedule, [&toRerun](Op *x) {
        return toRerun.find(x) == toRerun.end();
      });

      for (auto opToRerun : rerunSchedule) {
        logging::devicex::debug("Adding (second) recompute Op {}",
                                opToRerun->debugName());

        seqs[&progs.backwardFragment()].add(
            progs.recomputeFragment(opToRerun->id));
        mainGraphOpRegistry[taskId].push_back(opToRerun);
        PingPongPhase phase =
            op->hasPingPongPhase() &&
                    this->ir().getSessionOptions().pingPongPhases >= 2
                ? op->getPingPongPhase()
                : -1;
        progs.recordRecomputed(opToRerun->id, phase);
      }

      logging::devicex::debug("Adding post-turning check-point Op {}",
                              op->debugName());

      growOpx(opx, seqs[&progs.backwardFragment()]);

      mainGraphOpRegistry[taskId].push_back(op);
    }
  }

  else {
    throw internal_error("Unknown SchedulePreLoss in prepare, should "
                         "updateVertices have been called recently?");
  }
}

void Devicex::pipelinedOpTaskFunc(TaskId taskId, Op *op, SequenceMap &seqs) {
  Opx *opx = getOpx(op->id);

  if (op->copiesOptimizerTensors()) {
    growOpx(opx, seqs[&progs.streamOptimizerFromHostFragment()]);
  } else if (ir().getSessionOptions().enableGradientAccumulation &&
             (dynamic_cast<SGD1AcclReduceOp *>(op) ||
              dynamic_cast<SGD1VarUpdateOp *>(op) ||
              dynamic_cast<SGD1AcclUpdateOp *>(op) ||
              dynamic_cast<GradCopyToHostOp *>(op) ||
              dynamic_cast<GradCopyFromHostOp *>(op))) {
    outerLoopFragEmpty = false;
    growOpx(opx, seqs[&progs.accumulateOuterFragment()]);
  } else {
    std::set<Op *> toRerun;
    walkProducers(op, [&toRerun, &op, this](Op *x) {
      if (x->settings.recomputeType == RecomputeType::RECOMPUTE &&
          !progs.hasBeenRecomputed(x->id, -1) &&
          x->getPipelineStage() != op->getPipelineStage()) {
        toRerun.insert(x);
        return true;
      } else {
        return false;
      }
    });

    // The ops to rerun in topological order.
    auto rerunSchedule = op->getGraph().getOpSchedule({});
    boost::remove_erase_if(rerunSchedule, [&toRerun](Op *x) {
      return toRerun.find(x) == toRerun.end();
    });

    // Add the recomputations.
    for (auto opToRerun : rerunSchedule) {
      logging::devicex::debug("Adding (second) recompute Op {}",
                              opToRerun->debugName());
      progs.recordRecomputed(opToRerun->id, -1);
      seqs[&progs.pipelineForwardFragment(op->getPipelineStage(),
                                          "recompute of " + opToRerun->str())]
          .add(progs.recomputeFragment(opToRerun->id));

      mainGraphOpRegistry[taskId].push_back(opToRerun);
    }

    if (op->isConvertibleTo<IpuCopyOp>()) {
      // IpuCopyOps are handled as a special case in pipelining. Here,
      // the destination tensor is created using the
      // `createPipelinedOutput` method. Later, for each pipeline cycle
      // the copy appears in, a new copy program is added to the cycles
      // sequence using `IpuCopyOpx::growPipelined`.
      dynamic_cast<IpuCopyOpx *>(opx)->createPipelinedOutput();
    } else if (op->settings.recomputeType == RecomputeType::CHECKPOINT ||
               op->settings.recomputeType == RecomputeType::UNDEFINED) {
      logging::devicex::debug(
          "Adding post-turning check-point Op {} {} in pipelinedOpTaskFunc",
          op->str(),
          op->debugName());
      auto seqsKey =
          &progs.pipelineForwardFragment(op->getPipelineStage(), op->str());
      logging::devicex::debug("Obtained pipeline forward frag for ",
                              op->debugName());
      auto found = seqs.find(seqsKey);
      if (found == seqs.end()) {
        seqs[seqsKey] = poplar::program::Sequence{};
        found         = seqs.find(seqsKey);
      }
      logging::devicex::debug(
          "Growing {} {} in pipelinedOpTaskFunc", op->str(), op->debugName());

      growOpx(opx, found->second);
    } else if (op->settings.recomputeType == RecomputeType::RECOMPUTE) {
      logging::devicex::debug("Adding (first) recompute Op {}",
                              op->debugName());

      growOpx(opx, progs.recomputeFragment(op->id));

      seqs[&progs.pipelineForwardFragment(op->getPipelineStage(), op->str())]
          .add(progs.recomputeFragment(op->id));
    }
    mainGraphOpRegistry[taskId].push_back(op);
  }
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

// TODO consider moving the test in this function into the Ir (T12636)
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

  if (ir().useSyntheticData() == false) {
    logging::devicex::debug("Connecting initializer streams");

    for (auto id : ir().getTensorIds(TensorType::Variable)) {
      Tensor *tensor = ir().getTensor(id);
      if (!ir().streamingIsDisabledForTensor(id)) {
        logging::devicex::debug("   {}", tensor->str());
        pEngine->connectStream(h2dId(id), tensor->tensorData()->data());
      }
    }

    // Random seed
    if (ir().requiresRandomSeed()) {
      connectRandomSeedStream();
    }

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
      Tensor *tensor           = ir().getTensor(initId);
      int64_t n_bytes          = tensor->info.nbytes();
      d2hWeightBuffers[initId] = std::vector<char>(n_bytes);
      char *data0              = d2hWeightBuffers[initId].data();
      if (!ir().streamingIsDisabledForTensor(initId)) {
        // Only connect non-cached tensor streams,
        // RemoteBuffer handled separately
        bool isAnchorStream  = false;
        PopStreamId streamId = d2hId(initId, isAnchorStream);
        logging::devicex::debug(" {}", initId);
        engineToStreamVariables(data0, n_bytes, streamId);
        logging::devicex::debug("Created buffer (size {} B) and stream for {}",
                                n_bytes,
                                tensor->id);
      }
    }
  }

  // Hardware cycle counter - connect stream even if synthetic data mode is
  // not off
  if (ir().getSessionOptions().instrumentWithHardwareCycleCounter) {
    pEngine->connectStream(cycleCountStreamId(),
                           static_cast<void *>(&cycleCount));
  }
}

void Devicex::reconnectInputStreams() {
  logging::devicex::debug(
      "Reconnecting input streams, invalidating prefetches.");
  auto engineToInputStreamWithCallback =
      [&pEngine = pEngine, this](Tensor *tensor, poplar::DataStream &stream) {
        auto replicationFactor = getReplicationFactor();
        for (auto replicationIndex = 0; replicationIndex < replicationFactor;
             ++replicationIndex) {

          auto callback = std::make_unique<PrefetchCallback>(
              this->inputStreams[tensor->id]);
          pEngine->connectStreamToCallback(
              stream, replicationIndex, std::move(callback));
        }
      };

  for (Tensor *tensor : ir().dataStreamTensors()) {
    // The data stream for a tensor won't exist if using synthetic data, so
    // don't try and recreate them.
    if (!ir().useSyntheticData() && !tensor->isCached()) {
      engineToInputStreamWithCallback(tensor, fromHostStreams.at(tensor->id));
    }
  }
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

  const unsigned sharedStructureTilesPerIPU = 0;
  poplar::replication_factor rf(getReplicationFactor());

  logging::devicex::debug("Creating graph with replication factor {}",
                          getReplicationFactor());

  pGraph.reset(new poplar::Graph(popDevice, sharedStructureTilesPerIPU, rf));

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

  // Initialize remote buffer objects
  createRemoteBuffers();

  if (ir().virtualGraphsEnabled()) {
    auto numIPUs     = graph().getTarget().getNumIPUs();
    auto tilesPerIPU = graph().getTarget().getTilesPerIPU();

    int numIOTiles = 0;

    if (const char *env_p = std::getenv("GCL_NUM_IO_TILES")) {
      numIOTiles = boost::lexical_cast<int>(env_p);
      if (numIOTiles <= 0 || numIOTiles > 128 || (numIOTiles % 2 != 0)) {
        throw error(
            "{} is an invalid number of IO tiles. "
            "Number of IO tiles must be a positive even number up to 128",
            numIOTiles);
      }
      logging::devicex::info(
          "Reserving {} IO tiles for GCL collective operations on each IPU",
          numIOTiles);
    }

    for (unsigned ipu = 0; ipu < numIPUs; ++ipu) {
      unsigned startTile = ipu * tilesPerIPU;
      unsigned endTile   = (ipu + 1) * tilesPerIPU;
      virtualGraphs.emplace_back(
          graph().createVirtualGraph(startTile + numIOTiles, endTile));
      logging::devicex::info("Created virtual graph {} from {} to {}",
                             ipu,
                             startTile + numIOTiles,
                             endTile);
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
    logging::devicex::trace("Creating OPX for {}", op->debugName());
    opxs[op->id] = createOpx(op);
  }

  PriTasks tasks;

  // weights and accl tensors (i.e. variables):
  // 1) make tensor,
  // THEN
  // 2) make stream from host,
  // 3) create write prog,
  // 4) make stream to host,
  // 5) create read prog.
  // OR
  // 2) set initial value (if using synthetic data).
  for (auto id : ir().getTensorIds(TensorType::Variable)) {
    Tensor *tensor = ir().getTensor(id);
    if (tensor->isCached())
      continue;

    // 1
    tasks.add(initTensorTask(tensor));

    if (!ir().streamingIsDisabledForTensor(id)) {
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
    } else {
      // 2
      tasks.add(setInitTensorValTask(tensor));
    }
  }

  // constants:
  // 1) make tensor,
  // 2) set initial value.
  for (auto id : ir().getTensorIds(TensorType::Const)) {
    logging::devicex::debug("Adding initTensorTask for Const {}", id);
    Tensor *tensor = ir().getTensor(id);
    // 1
    tasks.add(initTensorTask(tensor));
    // 2
    tasks.add(setInitTensorValTask(tensor));
  }

  // InitOp outputs:
  // 1) ActGrad tensors
  for (Op *op : ir().getAllOps()) {
    if (InitOp *init = dynamic_cast<InitOp *>(op)) {
      logging::devicex::trace("Adding InitOp {} output initTensorTask for {}",
                              init->debugName(),
                              init->output->tensor(InitOp::getOutIndex())->id);
      tasks.add(initTensorTask(init->output->tensor(InitOp::getOutIndex())));
    }
  }

  // stream-to-device tensors :
  // 1) make tensor
  // THEN
  // 2) make stream
  // OR
  // 2) set initial value (if using synthetic data).
  for (auto id : ir().getTensorIds(TensorType::Stream)) {
    Tensor *tensor = ir().getTensor(id);

    // 1
    tasks.add(initTensorTask(tensor));
    logging::devicex::debug("Adding initTensorTask for Stream {}", id);

    // 2
    if (ir().useSyntheticData()) {
      tasks.add(setInitTensorValTask(tensor));
    } else {
      tasks.add(streamFromHostTask(tensor));
    }
  }

  // Init the random seed
  if (ir().requiresRandomSeed()) {
    auto seedTen = ir().getTensor(GetRandomSeedOp::getStreamedSeedTensorId());
    tasks.add(fromHostTask(seedTen, progs.setRandomSeedFromHostFragment()));
    tasks.add(initRandomSeed());
  }

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
  if (ir().useSyntheticData() == false) {
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

  // Two-step task linearisation:
  //
  // Purpose: Avoids circular dependencies when growing ops and initialising
  //          tensors with their tile mapping (layout) by deferring the
  //          growOpx call on Ops that call subgraphs as long as possible.
  //
  // Example:
  //
  // Graph A (nested subgraph):
  //        Data path x:    Data path y:
  //              InitOp    InitOp
  //                |         |
  //         weight_init    bias_init
  //                |         |
  //           CacheLoad    CacheLoad
  //                |         |
  //       weight_loaded    bias_loaded
  //      (graph output)    (graph output)
  //
  // Graph B (subgraph):
  //
  //         data    Call(A)
  //          |      |     |
  //          |   weight  bias
  //          |      |     |
  //         Convolution   |
  //                 |     |
  //            conv_out   |
  //                 |     |
  //                 AddBias
  //                    |
  //                   out
  //
  // Graph C (main graph):
  //              in0       in1
  //               |         |
  //             Call(B)     |
  //               |      Call(B)
  //               |         |
  //              out0      out1
  //
  // With one-step task linearisation, a circular dependency would arise
  // because the tile mapping of "bias" depends on "conv_out" existing, since
  // AddBias will clone the tile mapping of "conv_out" to "bias".
  // To create "conv_out", however, "weight" needs to have a tile mapping,
  // created by "Convolution". But "weight" is a copy of the "weight_loaded"
  // output of subgraph "A". That means "weight_init" is the tensor to which
  // the tensor layout of "Convolution" will be unwound to.
  // However, Call(A) is the producer of both "weight" and "bias", and would
  // therefore have to be grown before "Convolution", causing a circular
  // dependency:
  // Call(A) > Convolution > AddBias
  // (due to Opx output directed acyclic graph)
  // Convolution > AddBias > Call(A)
  // (due to AddBias creating the tensor mapping for "bias" based on "conv_out")
  //
  // Two-step linearisation separates growing Opx and creating Poplar
  // tensors, from adding the Poplar sequences, resulting from growing Opx,
  // together to form a linear order, in which the Opx are executed.
  //
  // With two-step task linearisation, there are multiple types of dependencies:
  // Call(A) depends on all Op tasks within graph A (SUBGRAPH dependency)
  // Call(B) depends on all Op tasks within graph B (SUBGRAPH dependency)
  // Convolution depends on "data" and "weight" (OUTPUT or TENSOR dependency,
  //       depending on the creator and populator of those tensors)
  // Convolution depends on Call(A) (SCHEDULE dependency)
  //
  // Step 1:
  // For growing Opx, we can ignore SCHEDULE dependencies, because data paths
  // flowing through subgraphs (from A to B in the example above):
  //
  // InitOp->weight_init->CacheLoad->weight_loaded->weight->Convolution->...
  //
  // can be grown independently from growing Call(A), thereby removing the
  // circular dependency.
  //
  // The linearized order in which Opx are grown for the example above becomes:
  // InitOp(x) > CacheLoad(x) > Convolution > InitOp(y) > CacheLoad(y) > AddBias
  // > Call(A) > Call(B) > Call(B)
  //
  // Step 2:
  // As soon as all data paths in a given graph (here A, B or C) are grown,
  // so the SUBGRAPH dependencies for CallOpx are fulfilled,
  // and we have one or more Poplar sequences for every opTask in the graph.
  // The Poplar sequences can be emplaced in graph functions (for subgraphs)
  // or the main sequence (main graph) according to SCHEDULER dependencies
  // (following the IR scheduler order).
  // All TENSOR dependencies can be ignored for emplacing sequences, since they
  // are only necessary to create the weight tensors (InitOp outputs) in step 1.
  //
  // The linearized order in which sequences are emplaced for the example above
  // becomes:
  // InitOp(x) > CacheLoad(x) > InitOp(y) > CacheLoad(y) > Call(A)
  // > Convolution > AddBias > Call(B) > Call(B)

  // Mappings for each task from final sequence to intermediate sequence
  std::map<TaskId, SequenceMap> seqs;
  std::vector<TaskId> taskOrder;

  logging::devicex::debug("Creating linear task schedule with OUTPUT, "
                          "SUBGRAPH and TENSOR dependencies.");
  auto createSchedule = tasks.getLinearised({DependencyType::OUTPUT,
                                             DependencyType::SUBGRAPH,
                                             DependencyType::TENSOR});

  logging::devicex::debug("Creating linear task schedule with OUTPUT, "
                          "SUBGRAPH and SCHEDULER dependencies.");
  auto emplaceSchedule = tasks.getLinearised({DependencyType::OUTPUT,
                                              DependencyType::SUBGRAPH,
                                              DependencyType::SCHEDULER});

  auto emplaceTaskSeqs = [&](std::set<TaskId> filter) {
    // 2.) Add intermediate sequences in final sequence
    // Linearised, ignoring TENSOR creation dependencies (weight init deps)
    // because all tensors exist at this point

    for (auto &emplaceTask : emplaceSchedule) {
      if ((filter.size() == 0 ||
           filter.find(emplaceTask.name) != filter.end()) &&
          seqs.find(emplaceTask.name) != seqs.end()) {
        logging::devicex::trace("Adding sequences for task {}",
                                emplaceTask.name);
        for (auto seq : seqs[emplaceTask.name]) {
          // Emplace intermediate sequence in final sequence
          seq.first->add(seq.second);
          logging::devicex::trace("  Target sequence: {}", seq.first);
        }
        // Erase sequences for task, so that each tasks's sequences
        // are only added once.
        seqs.erase(emplaceTask.name);
        taskOrder.push_back(emplaceTask.name);
      }
    }
  };

  // 1.) Create sequences and tensors
  // Linearised, ignoring SCHEDULER order dependencies, so that any circular
  // dependencies are avoided, when the scheduler order disagrees with the
  // tensor creation order

  for (auto &createTask : createSchedule) {
    logging::devicex::debug("Creating sequence for task {}", createTask.name);
    std::set<TaskId> subgraphTaskNames =
        createTask.getDependenciesOfTypes({DependencyType::SUBGRAPH});
    if (subgraphTaskNames.size() > 0) {
      // Make sure the subgraph sequences are emplaced before the scopeFragments
      // of the called graphs are constructed.
      logging::devicex::trace("  Task depends on {} subgraph tasks",
                              subgraphTaskNames.size());
      emplaceTaskSeqs(subgraphTaskNames);
    }
    seqs[createTask.name] = createTask.f();
  }
  // Emplace any main graph task sequences
  emplaceTaskSeqs({});

  logging::devicex::debug(getMainGraphOpString(taskOrder));

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
  setRandomSeedFromHost(); // Stream random seed value by default (prog empty if
                           // no randomness)

  trySaveTensorTileMap();

  prepareHasBeenCalled_ = true;
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
          logging::devicex::debug("Loading poplar Executable from '{}'",
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
                              poplar::program::Sequence &sq) const {

  auto f = [&sq, tensor, this]() {
    SequenceMap seqs;
    logging::devicex::debug("Adding poplar::program::Copy from host " +
                            tensor->id);

    seqs[&sq].add(poplar::program::Copy(fromHostStreams.at(tensor->id),
                                        tensors.get(tensor->id),
                                        doRearrangeOnHost(tensor)));
    return seqs;
  };

  return {-1e6, // writes to device: always as late as possible
          fromHostTaskId(tensor->id),
          {
              {streamFromHostTaskId(tensor->id),
               DependencyType::TENSOR}, // poplar::Stream created
              {initTensorTaskId(tensor->id),
               DependencyType::TENSOR} // poplar::Tensor created
          },
          f};
}

PriTask Devicex::toHostTask(Tensor *tensor,
                            poplar::program::Sequence &sq,
                            bool isAnchorStream) const {

  auto f = [&sq, tensor, this, isAnchorStream]() {
    SequenceMap seqs;
    logging::devicex::debug(
        "Adding poplar::program::Copy to host (isAnchorStream = {}) " +
            tensor->id,
        isAnchorStream);

    auto pToHostStreams = &toHostAnchorStreams;
    if (!isAnchorStream) {
      pToHostStreams = &toHostWeightStreams;
    }

    const auto &poplarTensor = tensors.get(tensor->id);
    const auto &poplarStream = pToHostStreams->at(tensor->id);

    // verify that number of elements of poplar Tensor and poplar Stream are the
    // same
    auto nElmsStream = poplarStream.numElements();
    auto nElmsTensor = tensors.get(tensor->id).numElements();
    if (nElmsStream != nElmsTensor) {
      throw internal_error("[Devicex::toHostTask] "
                           "The poplar::Tensor {} has {}, whereas the "
                           "poplar::Stream has {}. These should be the same.",
                           tensor->id,
                           nElmsTensor,
                           nElmsStream);
    }

    seqs[&sq].add(poplar::program::Copy(
        poplarTensor, poplarStream, doRearrangeOnHost(tensor)));
    return seqs;
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

  return {
      +1e6, // writes to host: always as early as possible
      taskId,
      {// the dependencies:
       // poplar::Stream creation task,
       {streamToHostTaskId(tensor->id, isAnchorStream), DependencyType::OUTPUT},
       // poplar::Tensor has its final values
       {finalPopulator, DependencyType::OUTPUT}},
      f};
}

PriTask Devicex::initBatchCounterTensorsTask() {

  auto f = [this]() {
    logging::devicex::debug("Adding batch counter tensors");

    // Add scalar tensors outside of the ir to track the batch
    // Id and decide when to execute the copy to the host
    for (ReturnPeriod N : ir().getDataFlow().rps()) {
      // Add to map so copy task can access
      batchCountingTensors[N] = getScalarVariable(graph(), poplar::INT, "");
      batchCountCheckingTensors[N] =
          getScalarVariable(graph(), poplar::BOOL, "");

      getConst(graph(), poplar::INT, {}, N, "batchCounter");

      poputil::mapTensorLinearly(graph(), batchCountingTensors[N]);
      poputil::mapTensorLinearly(graph(), batchCountCheckingTensors[N]);
    }

    // Make sure const 1 tensor exists
    getConst(graph(), poplar::INT, {}, 1, "one");
    return SequenceMap();
  };

  return {+1e6, // followed by writes to host: always as early as possible
          initBatchCounterTensorsTaskId(),
          {},
          f};
}

PriTask Devicex::updateBatchCountTask(poplar::program::Sequence &sq) {

  auto f = [&sq, this]() {
    SequenceMap seqs;
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
          seqs[&sq]);

      batchCountCheckingTensors[N] =
          popops::eq(graph(),
                     batchCountingTensors[N],
                     getConst(graph(), poplar::INT, {}, N, "batchCount/n"),
                     seqs[&sq]);

      // Reset batch count once it has reached N
      auto zero = getConst(graph(), poplar::INT, {}, 0, "batchCount/zero");
      seqs[&sq].add(poplar::program::If(
          batchCountCheckingTensors[N],
          poplar::program::Copy(zero, batchCountingTensors[N]),
          emptyseq));
    }
    return seqs;
  };

  return {+1e6, // followed by writes to host: always as early as possible
          updateBatchCountTaskId(),
          {
              {initBatchCounterTensorsTaskId(),
               DependencyType::TENSOR} // poplar::Tensor creation task
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
    SequenceMap seqs;
    logging::devicex::debug(
        "Adding conditional poplar::program::Copy to host " + tensor->id);

    poplar::Tensor isNthBatch = batchCountCheckingTensors.at(N);

    poplar::program::Sequence copyseq;
    copyseq.add(poplar::program::Copy(tensors.get(tensor->id),
                                      toHostAnchorStreams.at(tensor->id),
                                      doRearrangeOnHost(tensor)));

    // Placeholder 'do nothing' branch if not running copy program
    poplar::program::Sequence emptyseq;

    seqs[&sq].add(poplar::program::If(isNthBatch, copyseq, emptyseq));
    return seqs;
  };

  bool isAnchorStream = true;
  return {
      +1e6, // writes to host: always as early as possible
      toHostTaskId(tensor->id, isAnchorStream),
      {// the dependencies:
       // updating poplar::Tensor task,
       {updateBatchCountTaskId(), DependencyType::OUTPUT},
       // poplar::Stream creation task,
       {streamToHostTaskId(tensor->id, isAnchorStream), DependencyType::OUTPUT},
       // poplar::Tensor value setting task
       {taskWhichPopulates(tensor->id), DependencyType::OUTPUT}},
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

std::string Devicex::getSummaryReport(bool resetProfile) const {
  doProfileChecks();
  const auto &g_prof = pEngine->getGraphProfile();
  const auto &e_prof = pEngine->getExecutionProfile();

  std::stringstream ss;
  printProfileSummary(ss, g_prof, e_prof, reportOptions);

  if (resetProfile) {
    pEngine->resetExecutionProfile();
  }
  return ss.str();
}

std::string Devicex::getGraphReport(bool useCbor) const {
  doProfileChecks();
  std::stringstream ss;
  auto report = pEngine->getGraphProfile();
  if (useCbor) {
    serializeToCBOR(ss, report);
  } else {
    serializeToJSON(ss, report);
  }

  return ss.str();
}

std::string Devicex::getExecutionReport(bool useCbor, bool resetProfile) const {
  doProfileChecks();
  std::stringstream ss;
  auto report = pEngine->getExecutionProfile();

  if (useCbor) {
    serializeToCBOR(ss, report);
  } else {
    serializeToJSON(ss, report);
  }

  if (resetProfile) {
    pEngine->resetExecutionProfile();
  }
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

PopStreamId Devicex::gradientStoreStreamId(TensorId id) const {
  return gradientStoreStreamPrefix + id;
}

PopStreamId Devicex::gradientLoadStreamId(TensorId id) const {
  return gradientLoadStreamPrefix + id;
}

PopStreamId Devicex::weightLoadStreamId(TensorId id) const {
  return weightLoadStreamPrefix + id;
}

poplar::DataStream &Devicex::insertGradientStoreStream(TensorId tensorId,
                                                       TensorInfo tensorInfo,
                                                       poplar::Graph &graph) {
  auto streamMapEntry = toHostGradientStreams.find(tensorId);

  if (streamMapEntry == toHostGradientStreams.end()) {
    toHostGradientStreams.emplace(tensorId,
                                  poplar::DataStream(graph.addDeviceToHostFIFO(
                                      gradientStoreStreamId(tensorId),
                                      popType(tensorInfo),
                                      tensorInfo.nelms())));
    streamMapEntry = toHostGradientStreams.find(tensorId);
  } else {
    throw error("Tensor Id " + tensorId +
                " already exists in toHostGradientStreams");
  }

  return streamMapEntry->second;
}

poplar::DataStream &Devicex::insertGradientLoadStream(TensorId tensorId,
                                                      TensorInfo tensorInfo,
                                                      poplar::Graph &graph) {
  auto streamMapEntry = fromHostGradientStreams.find(tensorId);

  if (streamMapEntry == fromHostGradientStreams.end()) {
    fromHostGradientStreams.emplace(
        tensorId,
        poplar::DataStream(graph.addHostToDeviceFIFO(
            gradientLoadStreamId(tensorId),
            popType(tensorInfo),
            tensorInfo.nelms(),
            poplar::ReplicatedStreamMode::BROADCAST)));
    streamMapEntry = fromHostGradientStreams.find(tensorId);
  } else {
    throw error("Tensor Id " + tensorId +
                " already exists in fromHostGradientStreams");
  }

  return streamMapEntry->second;
}

poplar::DataStream &Devicex::insertWeightLoadStream(TensorId tensorId,
                                                    TensorInfo tensorInfo,
                                                    poplar::Graph &graph) {
  auto streamMapEntry = fromHostWeightLoadStreams.find(tensorId);

  if (streamMapEntry == fromHostWeightLoadStreams.end()) {
    fromHostWeightLoadStreams.emplace(
        tensorId,
        poplar::DataStream(graph.addHostToDeviceFIFO(
            weightLoadStreamId(tensorId),
            popType(tensorInfo),
            tensorInfo.nelms(),
            poplar::ReplicatedStreamMode::BROADCAST)));
    streamMapEntry = fromHostWeightLoadStreams.find(tensorId);
  } else {
    throw error("Tensor Id " + tensorId +
                " already exists in weightStoreStreams");
  }

  return streamMapEntry->second;
}

const std::vector<TensorId> &Devicex::getHostReduceStreamIds() const {
  return hostReduceStreamIds;
}

std::vector<TensorId> &Devicex::getHostReduceStreamIds() {
  return hostReduceStreamIds;
}

void Devicex::connectStreamToCallback(const std::string &streamHandle,
                                      std::function<void(void *)> callback,
                                      unsigned index) {
  pEngine->connectStreamToCallback(streamHandle, index, callback);
}

} // namespace popx
} // namespace popart
