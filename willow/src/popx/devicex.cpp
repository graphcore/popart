#include <algorithm>
#include <cctype>
#include <fstream>
#include <random>
#include <set>

#include <memory>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>
#include <popsys/CSRFunctions.hpp>
#include <popsys/codelets.hpp>
#include <poputil/exceptions.hpp>
#include <poponnx/devicemanager.hpp>
#include <poponnx/error.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/graph.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/logging.hpp>
#include <poponnx/op.hpp>
#include <poponnx/op/call.hpp>
#include <poponnx/op/if.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/devicexmanager.hpp>
#include <poponnx/popx/opx.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/popx/poplaroptionsx.hpp>
#include <poponnx/pritask.hpp>
#include <poponnx/recompute.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensordata.hpp>
#include <poponnx/tojson.hpp>

namespace poponnx {
namespace popx {

class devicex_memory_allocation_err : public poponnx::memory_allocation_err {

  const poplar::graph_memory_allocation_error exception;
  const poplar::OptionFlags reportOptions;

public:
  devicex_memory_allocation_err(const devicex_memory_allocation_err &rhs)
      : poponnx::memory_allocation_err(rhs.what()),
        exception(std::move(rhs.exception)), reportOptions(rhs.reportOptions) {}

  devicex_memory_allocation_err(const poplar::graph_memory_allocation_error &e,
                                const poplar::OptionFlags &_reportOptions)
      : poponnx::memory_allocation_err(e.what()), exception(std::move(e)),
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

  auto turningPointOp = ir().getTurningPointOp();
  std::stringstream ss;
  auto seriesNums = getMainGraphOpSeriesNums();
  std::set<Op *> seen;
  for (auto op : mainGraphOpRegistery) {
    auto found = seen.count(op);
    seen.insert(op);
    std::string type;
    if (op == turningPointOp) {
      type = "turn";
    } else if (found != 0) {
      type = "re.1";
    } else if (op->settings.recomputeType == RecomputeType::RECOMPUTE) {
      type = "re.0";
    } else if (op->getPhase() == Phase::BWD) {
      type = "....";
    } else {
      type = "    ";
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

const std::string randomSeedId = "randomSeed";

void Devicex::run(PopPrograms::ProgramIndex ind) {
  if (isEngineLoaded() == false) {
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
      MutableVoidData stepout = weights.weight(id);
      hostStreamToHost(stepout, id);
    }
  }
}

void Devicex::writeWeights(const IWeightsIO &weights) {
  // Better to do this the otherway round
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
        throw error("No TensorId " + id + " in final host destination map");
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
  auto irTensorStr   = ir.getTensor(id)->str();
  auto expectedShape = ir.getTensor(id)->info.shape_szt();

  if (pt.shape() != expectedShape) {
    std::stringstream ss;
    ss << "poplar::Tensor " << id << " of unexpected shape. "
       << "Poplar tensor shape: ";
    appendSequence(ss, pt.shape());
    ss << ". Expected (Ir) tensor shape: ";
    appendSequence(ss, expectedShape);
    ss << ". This for tensor " << irTensorStr;
    throw error(ss.str());
  }

  // confirm types agree
  auto expectedType = popType(ir.getTensor(id)->info);
  if (pt.elementType() != expectedType) {
    std::stringstream ss;
    ss << "poplar::Tensor " << id << " of unexpected Type. "
       << "Poplar tensor type : " << pt.elementType();
    ss << ". Expected (Ir) tensor type : " << expectedType;
    ss << ". This for tensor " << irTensorStr;
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

PopPrograms::PopPrograms(const int repeatCount_) : repeatCount(repeatCount_) {
  if (repeatCount_ <= 0) {
    throw error("Program repeat count must be greater than zero");
  }
}

poplar::program::Sequence &PopPrograms::streamWeightsFromHostFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::STREAMWEIGHTSFROMHOST)];
}

poplar::program::Sequence &PopPrograms::streamOptimizerFromHostFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::STREAMOPTIMIZERFROMHOST)];
}

poplar::program::Sequence &PopPrograms::initFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::INIT)];
}

poplar::program::Sequence &PopPrograms::mainProgramFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::MAINPROGRAM)];
}

poplar::program::Sequence &PopPrograms::setRandomSeedFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::SETRANDOMSEED)];
}

poplar::program::Sequence &PopPrograms::setRandomDropoutSeedFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::SETRANDOMDROPOUTSEED)];
}

poplar::program::Sequence &PopPrograms::toHostFinalCopyFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::TOHOSTFINALCOPY)];
}

poplar::program::Sequence &PopPrograms::weightsToHostFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::WEIGHTSTOHOST)];
}

poplar::program::Sequence PopPrograms::weightsFromHost() {
  poplar::program::Sequence prog;
  prog.add(streamWeightsFromHostFragment());
  return prog;
}

poplar::program::Sequence PopPrograms::optimizerFromHost() {
  poplar::program::Sequence prog;
  prog.add(streamOptimizerFromHostFragment());
  return prog;
}

poplar::program::Sequence PopPrograms::program() {
  poplar::program::Sequence prog;
  prog.add(mainProgramFragment());

  poplar::program::Sequence outer;

  // Only add the init fragment if settings have been added
  if (!initFragment().isEmpty()) {
    outer.add(initFragment());
  }
  outer.add(setRandomSeedFragment());
  outer.add(setRandomDropoutSeedFragment());
  outer.add(poplar::program::Repeat(repeatCount, prog));
  outer.add(toHostFinalCopyFragment());

  return outer;
}

poplar::program::Sequence PopPrograms::weightsToHost() {
  return weightsToHostFragment();
}

std::vector<poplar::program::Program> PopPrograms::progs() {
  std::vector<poplar::program::Program> ps(ProgramIndex::N);

  ps[ProgramIndex::WEIGHTSFROMHOST]   = weightsFromHost();
  ps[ProgramIndex::OPTIMIZERFROMHOST] = optimizerFromHost();
  ps[ProgramIndex::PROGRAM]           = program();
  ps[ProgramIndex::WEIGHTSTOHOST]     = weightsToHost();

  return ps;
}

poplar::program::Sequence &
PopPrograms::programFragment(PopPrograms::ProgramFragmentIndex index) {
  return seqs[static_cast<int>(index)];
}

poplar::program::Sequence &PopPrograms::programFragment(const Graph &graph) {
  if (graph.id.str().empty()) {
    return mainProgramFragment();
  } else {
    return scopeSeqs.at(graph.id.str());
  }
}

bool PopPrograms::containsFragment(const Graph &graph) const {
  if (graph.id.str().empty()) {
    return true;
  } else {
    return scopeSeqs.find(graph.id.str()) != scopeSeqs.end();
  }
}

void PopPrograms::createFragment(const Graph &graph) {
  scopeSeqs.insert({graph.id.str(), {}});
}

poplar::Graph &Devicex::graph() { return *pGraph; }
const poplar::Graph &Devicex::graph() const { return *pGraph; }

poplar::Graph &Devicex::getVirtualGraph(int64_t virtualGraphIndex) {
  if (virtualGraphIndex < 0 || virtualGraphIndex >= virtualGraphs.size()) {
    throw error("Invalid virtual graph index {} ({} available)",
                virtualGraphIndex,
                virtualGraphs.size());
  }
  return virtualGraphs.at(virtualGraphIndex);
}

Devicex::Devicex(const Ir &ir, std::shared_ptr<DeviceInfo> deviceInfo_)
    : poponnx::Device(ir),
      progs(PopPrograms(ir.getDataFlow().batchesPerStep())), tensors(ir),
      deviceInfo(deviceInfo_), prepareHasBeenCalled(false) {

  logging::devicex::info("Setting selected device: {}", *deviceInfo);

  if (!deviceInfo->attach()) {
    throw error("failed to attach to device");
  }

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

  // Not sure what these options should be
  if (ir.getExecutionMode() == Ir::ExecutionMode::TRAINING) {
    fwdMmOptions.options.insert({"fullyConnectedPass", "TRAINING_FWD"});
  } else {
    fwdMmOptions.options.insert({"fullyConnectedPass", "INFERENCE_FWD"});
  }

  bwdMmLhsOptions.options.insert({"fullyConnectedPass", "TRAINING_BWD"});
  bwdMmRhsOptions.options.insert({"fullyConnectedPass", "TRAINING_WU"});

  engineOptions.set("target.workerStackSizeInBytes", "0x200");
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
    ss << ",\nBatches per step : " << ir().getDataFlow().batchesPerStep()
       << '.';
    throw error(ss.str());
  }

  // Log the name and shape of the tensor
  logging::devicex::debug("       {} {}", id, srcInfo.shape());

  auto srcType = srcInfo.dataType();
  auto dstType = dstInfo.dataType();

  // check type compatibility
  if (srcType == dstType) {
    // copy the full step data from src to dst
    std::memcpy(dst, src, srcInfo.nbytes());
  }

  else if (srcType == DataType::INT64 && dstType == DataType::INT32) {
    logging::devicex::debug("Copying (host) tensor {} from INT64 to INT32", id);
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
    ss << "Type discrepency for tensor " << id
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

void Devicex::anchorsHostToHostStreams(const IStepIO &stepio) {

  if (useSyntheticData() == false) {
    std::string prefix = "     ";
    logging::devicex::debug(prefix + "Copying to h2d stream address(es) ");
    for (Tensor *tensor : ir().dataStreamTensors()) {
      ConstVoidData stepin = stepio.in(tensor->id);

      // where to write to on host,
      auto dst = static_cast<void *>(h2dBuffers.at(tensor->id).data());
      // where to read from on host,
      auto src = stepin.data;

      // we calculate the TensorInfo for dst. If batchesPerStep() = 1, then
      // it has the same dimensions as tensor->info. Otherwise it has has
      // an extra dimension of size batchesPerStep() to accommmodate all
      // step anchor tensors.
      auto stepDstShape = tensor->info.shape();
      if (ir().getDataFlow().batchesPerStep() > 1) {
        stepDstShape.insert(stepDstShape.begin(),
                            ir().getDataFlow().batchesPerStep());
      }
      // if the replicationFactor is greater than 1 then add an extra
      // dimension of size replicationFactor so we can report multiple
      // copies of the tensor
      // Q: Should replicated tensors be combined before returning?
      if (getReplicationFactor() > 1) {
        stepDstShape.insert(stepDstShape.begin(), getReplicationFactor());
      }
      TensorInfo dstInfo{tensor->info.dataType(), stepDstShape};

      // the info of the user provided src step tensor
      TensorInfo srcInfo = stepin.info;

      hostToHostStream(dst, src, dstInfo, srcInfo, tensor->id);
    }
  }
}

void Devicex::anchorsHostFromHostStreams(const IStepIO &stepio) {

  if (useSyntheticData() == false) {
    std::string prefix = "     ";
    logging::devicex::debug(prefix + "Copying from d2h stream address(es) ");
    for (TensorId anchorId : ir().getDataFlow().anchors()) {
      MutableVoidData stepout = stepio.out(anchorId);
      hostStreamToHost(stepout, anchorId);
    }
  }
}

void Devicex::run(const IStepIO &stepio) {
  if (!prepareHasBeenCalled) {
    throw error("Devicex::prepare() must be called before"
                " Devicex::run(const IStepIO &) is called.");
  }
  logging::devicex::debug("Performing one step: ");
  anchorsHostToHostStreams(stepio);

  pEngine->enableExecutionProfiling();
  run(PopPrograms::ProgramIndex::PROGRAM);

  anchorsHostFromHostStreams(stepio);
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

TaskId Devicex::taskWhichCreates(TensorId id) const {
  Tensor *tensor = ir().getTensor(id);
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

std::vector<InputCreatorCandidate>
Devicex::getCreatorEndpoints(Tensor *tensor,
                             std::vector<OpxInAndOutIndex> pathFromInput,
                             bool excludeEndpointsFromPath,
                             bool includeDeadends) const {

  std::vector<InputCreatorCandidate> endpoints;
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
        endpoints.push_back({inIndex, opx, updatedPath});
        break;
      }
      // Recursively search the DAG downstream of the op until we
      // have set of endpoints that can create the tensor
      case InputCreatorType::CANUNWIND: {
        for (auto &ind_ten : op->output->tensorMap()) {
          auto nextOutputTensor = ind_ten.second;
          auto outIndex         = ind_ten.first;
          updatedPath.push_back({opx, inIndex, outIndex});
          for (auto candidate :
               getCreatorEndpoints(nextOutputTensor, updatedPath)) {
            endpoints.push_back(candidate);
          }
        }
        break;
      }
      // Consuming op can't create tensor
      case InputCreatorType::DEADEND: {
        if (includeDeadends) {
          if (!excludeEndpointsFromPath) {
            updatedPath.push_back(
                {opx, inIndex, -1}); // note: no valid outIndex
          }
          endpoints.push_back({inIndex, opx, updatedPath});
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

optional<InputCreatorCandidate>
Devicex::getTensorCreator(Tensor *tensor) const {

  auto errorbase = [&tensor]() {
    std::stringstream ss;
    ss << "Failed to add tensor " << tensor->id << '.';
    tensor->consumers.append(ss);
    return ss.str();
  };

  // Search of the graph to get the candidate Opxs that
  // know how to create this tensor.
  // The pathFromInput argument is an empty vector, as
  // we are starting the search from the root (input)
  std::vector<InputCreatorCandidate> candidates =
      getCreatorEndpoints(tensor, {});

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
    } else {
      logging::devicex::warn("Input tensor '{}' has multiple creator "
                             "candidates, but they are not in agreement",
                             tensor->id);
    }
  }

  if (candidates.size() > 1) {
    throw error(errorbase() + "\nConflicting creator candidates.");
  } else if (candidates.size() == 1) {
    return candidates.front();
  } else {
    return boost::none;
  }
}

// Design decision : leave the option for a Tensor to be
// created based on complex global criteria open.
PriTask Devicex::initTensorTask(Tensor *tensor) {
  auto candidate = getTensorCreator(tensor);

  // 1. A unique candidate creator will create the tensor
  // 2. The tensor will be unwound (have its layout modified)
  //    by view-changing opxs on the path from the input to
  //    the candidate candidate
  if (candidate) {
    const Opx *creator = candidate->opx;
    int inIndex        = candidate->index;
    auto pathFromInput = candidate->getPathFromInput();

    auto f = [this, creator, inIndex, pathFromInput, tensor]() {
      logging::devicex::debug("Creating poplar::Tensor {}", tensor->id);
      poplar::Tensor input = creator->createInput(inIndex, tensor->str());
      logging::devicex::debug("poplar::Tensor {} created", tensor->id);

      // Reverse the path,
      // The first element is now the Opx producing a tensor consumed by
      // the candidate.
      // The last element is now the Opx consuming the input we are mapping.
      auto pathToInput = pathFromInput;
      std::reverse(pathToInput.begin(), pathToInput.end());

      for (auto opxOnPath : pathToInput) {
        input = opxOnPath.opx->unwindTensorLayout(
            input, opxOnPath.inIndex, opxOnPath.outIndex);
      }
      tensors.insert(tensor->id, input);
      efficientlyCreatedInputTensors.insert(tensor->id);
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
  } else {

    auto f = [this, tensor]() {
      logging::devicex::warn("Creating input tensor '{}' linearly. No "
                             "operator specific allocator found",
                             tensor->id);

      // Get paths to both creator candidates and deadends, and print for debug
      std::vector<InputCreatorCandidate> endpoints =
          getCreatorEndpoints(tensor, {}, false, true);
      int endpointId = 1;
      logging::devicex::debug("Printing paths to {} endpoint(s) found when "
                              "searching for a creator candidate for {}",
                              endpoints.size(),
                              tensor->id);
      for (auto endpoint : endpoints) {
        auto path = endpoint.getPathFromInput();
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
      std::vector<int64_t> ipus;
      for (auto *op : tensor->consumers.getOps()) {

        int64_t index = -1;
        if (op->getVirtualGraphId())
          index = *(op->getVirtualGraphId());

        // The copyToIpu op assume that the tensor will already
        // have been copied to the ipu from another op
        if (op->opid != Onnx::CustomOperators::IpuCopy) {

          auto &graph = getOpx(op->id)->graph();

          if (ipus.end() == std::find(ipus.begin(), ipus.end(), index)) {

            auto newTensor = graph.addVariable(
                popType(tensor->info), tensor->info.shape_szt(), tensor->str());
            linearMapper.mapTensor(graph, newTensor);

            tensors.insert(tensor->id, newTensor);
            linearlyCreatedInputTensors.insert(tensor->id);
            ipus.push_back(index);
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

    auto seedTensor =
        graph().addVariable(poplar::UNSIGNED_INT, {2}, randomSeedId);
    graph().setTileMapping(seedTensor, 0);

    auto &sq = progs.setRandomSeedFragment();

    if (!useSyntheticData()) {
      // Right now just use the same random seed on each replica if the user set
      // it T9638 - to corrupt the seed for each replicant
      auto dataStream =
          graph().addHostToDeviceFIFO(h2dId(randomSeedId),
                                      seedTensor.elementType(),
                                      seedTensor.numElements(),
                                      poplar::ReplicatedStreamMode::REPLICATE);

      sq.add(poplar::program::Copy(dataStream, seedTensor));
    }

    poprand::setSeed(
        graph(), seedTensor, 0, sq, fmt::format("{}/set", randomSeedId));
  };

  return {
      +1e6,              // high priority
      "initRandomSeed",  // name of this task
      {},                // depends on
      initRandomSeedTask // what to run when the task is executed
  };
}

PriTask Devicex::initDropoutRandomSeed() {
  auto initDropoutRandomSeedTask = [this]() {
    logging::devicex::debug("Initializing dropout random seed tensor.");

    dropoutRandomSeed = graph().addVariable(
        poplar::UNSIGNED_INT, {2}, dropoutRandomSeedTensorId());
    graph().setTileMapping(dropoutRandomSeed, 0);
  };

  return {
      +1e6,                      // high priority
      initDropoutRandomSeedId(), // name of this task
      {},                        // depends on
      initDropoutRandomSeedTask  // what to run when the task is executed
  };
}

PriTask Devicex::incrementDropoutRandomSeedTask() {
  auto incrementDropoutRandomSeedTask = [this]() {
    popops::addInPlace(graph(),
                       *getDropoutRandomSeed(),
                       getConst(graph(), poplar::UNSIGNED_INT, {}, 1, "one"),
                       mainProgramFragment());
  };

  return {
      +1e6,                          // high priority
      "incrementDropoutRandomSeed",  // name of this task
      {initDropoutRandomSeedId()},   // depends on
      incrementDropoutRandomSeedTask // what to run when the task is executed
  };
}

void Devicex::connectRandomSeedStream() {
  std::default_random_engine randomGenerator;

  // Generate a seperate random seed for each replicant.

  for (uint16_t replicaId = 0; replicaId < getReplicationFactor();
       ++replicaId) {

    auto callback = [randomGenerator, replicaId](void *ptr) mutable {
      std::uniform_int_distribution<uint64_t> distribution;
      uint64_t *data = reinterpret_cast<uint64_t *>(ptr);

      logging::devicex::debug("     Updating random seed for replica:{}",
                              replicaId);
      data[0] = distribution(randomGenerator);
    };

    pEngine->connectStreamToCallback(h2dId(randomSeedId), replicaId, callback);
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

    case DataType::UNDEFINED:
    case DataType::UINT8:
    case DataType::INT8:
    case DataType::INT64:
    case DataType::BOOL:
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
    std::vector<int64_t> ipus;
    for (auto *op : tensor->consumers.getOps()) {

      // Assume another op will copy the tensor for an ipucopy
      if (op->opid != Onnx::CustomOperators::IpuCopy) {
        auto &graph = getOpx(op->id)->graph();

        int64_t index = -1;
        if (op->getVirtualGraphId())
          index = *(op->getVirtualGraphId());

        // Only stream the tensor once for all op's that consume it on an ipu
        if (std::find(ipus.begin(), ipus.end(), index) == ipus.end()) {

          logging::devicex::debug(
              "Creating host-to-device FIFO {} copied to ipu:{} type:{}",
              tensor->id,
              index,
              tensor->tensorType());

          poplar::ReplicatedStreamMode mode;

          if (tensor->tensorType() == TensorType::Variable) {
            // If it is a variable we 'broadcast' the same tensor
            // to all replicants
            mode = poplar::ReplicatedStreamMode::BROADCAST;

          } else if (tensor->tensorType() == TensorType::Stream) {

            // If it is a stream then we 'duplicate' the stream to
            // each replicant and poplar takes care of the mapping.

            auto optimizerTensors = ir().optimizerTensors();

            if (std::find_if(optimizerTensors.begin(),
                             optimizerTensors.end(),
                             [tensor](const Tensor *value) -> bool {
                               return tensor->id == value->id;
                             }) != optimizerTensors.end()) {

              // Special case of the optimizer tensors which are streams, but
              // should be broadcast. i.e. 1 value sent to all devices
              mode = poplar::ReplicatedStreamMode::BROADCAST;
            } else {
              mode = poplar::ReplicatedStreamMode::REPLICATE;
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

          ipus.push_back(index);
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

PriTask Devicex::streamToHostTask(Tensor *tensor) {
  auto f = [this, tensor]() {
    logging::devicex::debug("Creating device-to-host FIFO {}", tensor->id);

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

poplar::program::Sequence &Devicex::mainProgramFragment() {
  return progs.programFragment(PopPrograms::ProgramFragmentIndex::MAINPROGRAM);
}

bool Devicex::containsFragment(const Graph &graph) const {
  return progs.containsFragment(graph);
}

void Devicex::createFragment(const Graph &graph) {
  return progs.createFragment(graph);
}

poplar::program::Sequence &Devicex::programFragment(const Graph &graph) {
  return progs.programFragment(graph);
}

void Devicex::addOpTasks(PriTasks &tasks) {

  // Ensure there is a program fragment for every graph
  for (auto graph : ir().getGraphSchedule()) {
    if (!containsFragment(*graph)) {
      createFragment(*graph);
    }
  }

  auto mainGraphSchedule = ir().getMainGraph().getOpSchedule({});
  auto turningPointOp    = ir().getTurningPointOp();

  // repeating logic in Ir::getOpSchedule (can be simpfified there?)
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
      // If the op calls another graph
      // the ops in that graph should be scheduled first
      for (auto calledGraph : op->getCalledGraphs()) {
        addGraph(calledGraph);
      }
      if (op->settings.recomputeType != RecomputeType::CHECKPOINT) {
        throw error("non-main Graph Op which is not a CHECKPOINT");
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

  bool isPostTurningPoint = false;
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

    auto task = opTask(op, priority, prevOpTaskId, isPostTurningPoint);

    tasks.add(task);
    prevOpTaskId = task.name;
    priority -= 1.;
    isPostTurningPoint = isPostTurningPoint || op == turningPointOp;
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

PriTask Devicex::opTask(Op *op,
                        double priority,
                        TaskId prevOpTaskId,
                        bool isPostTurningPoint) {

  Opx *opx = getOpx(op->id);

  // although priority should guarantee that this
  // task is only run after inputs are all created,
  // we add a dependency to the input tensors, just
  // in case someone plays with the priorities.
  // Moreover, we must state the copy-from-host deps
  std::vector<TaskId> deps;
  for (auto t_inds : op->input->indicesMap()) {
    Tensor *tensor = t_inds.first;

    auto creatorTask = taskWhichCreates(tensor->id);
    // Make sure we only add the creatorTask once in the dependency list
    if (std::find(deps.begin(), deps.end(), creatorTask) == deps.end())
      deps.push_back(creatorTask);

    // if the tensor is streamed on, we must wait
    // 'til the Copy has happened
    if (tensor->tensorType() == TensorType::Stream) {
      if (useSyntheticData() == false)
        deps.push_back(fromHostTaskId(tensor->id));
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

  // The following code can be useful to debug floating point exceptions by
  // printing the names of the Ops as they are executed.
  // poplar::Tensor d1 = masterGraph().addVariable(poplar::FLOAT, {1},
  // opx->op_p->str()); masterGraph().setTileMapping(d1, 0);
  // programFragment(opx->op_p->getGraph()).add(poplar::program::PrintTensor(opx->op_p->str(),
  // d1));

  auto f = [op, opx, this, isPostTurningPoint]() {
    const auto &containingGraph = opx->op_p->getGraph();
    // if this Op is not in the main scope
    if (!containingGraph.id.str().empty()) {
      logging::devicex::debug("Creating output tensors for non-main " +
                              opx->op_p->debugName());
      opx->grow(programFragment(containingGraph));
    }

    // else if this Op is in the main scope
    else {

      // pre-loss : create vertices for all recompute types
      if (!isPostTurningPoint) {
        if (op->settings.recomputeType == RecomputeType::CHECKPOINT) {
          logging::devicex::debug("Adding checkpoint Op {}", op->debugName());
          opx->grow(mainProgramFragment());
        } else if (op->settings.recomputeType == RecomputeType::RECOMPUTE) {
          logging::devicex::debug("Adding (first) recompute Op {}",
                                  op->debugName());
          opx->grow(progs.recomputeFragment(op->id));
          mainProgramFragment().add(progs.recomputeFragment(op->id));
        } else {
          throw error("Unrecognised recompute type");
        }
        mainGraphOpRegistery.push_back(op);
      }

      // post-loss
      else {
        if (op->settings.recomputeType != RecomputeType::CHECKPOINT) {
          throw error("ILE: Non-checkpoint post turning point");
        }

        // decide what needs to be re-run
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

        std::vector<Op *> toRerunVector;
        for (auto x : toRerun) {
          toRerunVector.push_back(x);
          progs.recordRecomputed(x->id);
        }
        std::sort(toRerunVector.begin(),
                  toRerunVector.end(),
                  [](const Op *a, const Op *b) { return a->id < b->id; });
        for (auto opToRerun : toRerunVector) {
          logging::devicex::debug("Adding (second) recompute Op {}",
                                  opToRerun->debugName());
          mainProgramFragment().add(progs.recomputeFragment(opToRerun->id));
          mainGraphOpRegistery.push_back(opToRerun);
        }

        logging::devicex::debug("Adding post-turning check-point Op {}",
                                op->debugName());

        opx->grow(mainProgramFragment());
        mainGraphOpRegistery.push_back(op);
      }
    }
  };
  return {priority, opTaskId(op), deps, f};
}

InputCreatorCandidate::InputCreatorCandidate(
    int conIndex_,
    const Opx *opx_,
    std::vector<OpxInAndOutIndex> pathFromInput_)
    : index(conIndex_), opx(opx_), pathFromInput(pathFromInput_) {}

std::vector<OpxInAndOutIndex> InputCreatorCandidate::getPathFromInput() {
  return pathFromInput;
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
      pEngine->connectStream(h2dId(id), tensor->tensorData()->data());
    }

    // Random seed
    connectRandomSeedStream();

    logging::devicex::debug("Connecting optimizer streams");
    for (Tensor *tensor : ir().optimizerTensors()) {
      logging::devicex::debug(" {}", tensor->id);
      pEngine->connectStream(h2dId(tensor->id), tensor->tensorData()->data());
    }

    auto engineToStream = [&pEngine = pEngine](char *data0,
                                               int64_t n_bytes,
                                               PopStreamId streamId) {
      // Poplar has no const void * version
      auto addr0 = static_cast<void *>(data0);
      auto addr1 = static_cast<void *>(data0 + n_bytes);
      // connect the stream (circular buffer)
      pEngine->connectStream(streamId, addr0, addr1);
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

    logging::devicex::debug(
        "Creating host buffers for h2d streams, and connecting");
    for (Tensor *tensor : ir().dataStreamTensors()) {
      logging::devicex::debug(" {}", tensor->id);
      PopStreamId streamId = h2dId(tensor->id);
      // allocate host memory, where the poplar::Stream will read data from
      int64_t n_bytes = ir().getDataFlow().batchesPerStep() *
                        tensor->info.nbytes() * getReplicationFactor();
      h2dBuffers[tensor->id] = std::vector<char>(n_bytes);
      char *data0            = h2dBuffers[tensor->id].data();
      engineToStream(data0, n_bytes, streamId);
    }

    logging::devicex::debug(
        "Creating host buffers for anchor d2h streams, connecting");
    for (TensorId anchorId : ir().getDataFlow().anchors()) {
      logging::devicex::debug(" {}", anchorId);

      PopStreamId streamId = d2hId(anchorId);
      Tensor *tensor       = ir().getTensor(anchorId);
      int64_t batch_bytes  = tensor->info.nbytes();
      int64_t n_bytes;
      switch (ir().getDataFlow().art(anchorId).id()) {
      case (AnchorReturnTypeId::FINAL): {
        n_bytes = batch_bytes * getReplicationFactor();
        break;
      }
      case (AnchorReturnTypeId::EVERYN): {
        n_bytes = batch_bytes *
                  (ir().getDataFlow().batchesPerStep() /
                   ir().getDataFlow().art(anchorId).rp()) *
                  getReplicationFactor();
        break;
      }
      case (AnchorReturnTypeId::ALL): {
        n_bytes = batch_bytes * ir().getDataFlow().batchesPerStep() *
                  getReplicationFactor();
        break;
      }
      }

      // The host data need to be multiplied
      d2hBuffers[anchorId] = std::vector<char>(n_bytes);
      char *data0          = d2hBuffers[tensor->id].data();
      engineToStream(data0, n_bytes, streamId);
    }

    logging::devicex::debug(
        "Creating host buffers for weight d2h streams, connecting");

    for (auto initId : ir().getTensorIds(TensorType::Variable)) {
      logging::devicex::debug(" {}", initId);
      PopStreamId streamId = d2hId(initId);
      Tensor *tensor       = ir().getTensor(initId);
      int64_t n_bytes      = tensor->info.nbytes();
      d2hBuffers[initId]   = std::vector<char>(n_bytes);
      char *data0          = d2hBuffers[initId].data();
      engineToStreamVariables(data0, n_bytes, streamId);
    }
  }
}

// Floating point settings are not suported on CPU
void Devicex::setFloatingPointBehaviour(poplar::Graph &graph) {

  if (ir().getSessionOptions().enableFloatingPointChecks) {
    if (deviceInfo->getType() == DeviceType::Ipu) {
      logging::devicex::info("Enabling all floating point checks");
      // Not enabling stochasitc rounding, that is done in a seperate call
      popsys::FloatingPointBehaviour behaviour(true, true, true, false, true);
      popsys::setFloatingPointBehaviour(
          graph, progs.initFragment(), behaviour, "/init");
    } else {
      logging::devicex::warn(
          "Floating point checks can not be enabled for non IPU devices");
    }
  }
}

// Stocastic rounding is only supported on the IPU
void Devicex::setStochasticRoundingBehaviour(poplar::Graph &graph) {

  if (ir().getSessionOptions().enableStochasticRounding) {
    if (deviceInfo->getType() == DeviceType::Ipu) {
      logging::devicex::info("Enabling stochastic rounding");
      bool behaviour = true;
      popsys::setStochasticRounding(
          graph, progs.initFragment(), behaviour, "/init");
    } else {
      logging::devicex::warn(
          "Stochastic rounding can not be enabled for non IPU devices");
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
  popsys::addCodelets(graph());

  setFloatingPointBehaviour(graph());
  setStochasticRoundingBehaviour(graph());

  if (ir().getSessionOptions().enableVirtualGraphs) {
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
      if (op->getVirtualGraphId()) {
        int64_t index = *(op->getVirtualGraphId());
        if (index < 0 || index >= numIPUs) {
          throw error("{} has been assigned to an invalid virtual graph {}. "
                      "numIPUs = {}.",
                      op->debugName(),
                      index,
                      numIPUs);
        }
      }
    }
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
      tasks.add(streamToHostTask(tensor));
      // 5
      tasks.add(toHostTask(tensor, progs.weightsToHostFragment()));
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

    if (useSyntheticData() == false) {
      // 2
      tasks.add(streamFromHostTask(tensor));
    }
  }

  // Init the random seed(s)
  tasks.add(initRandomSeed());
  if (isDropoutRandomSeedRequired()) {
    // Dropout has a separate random seed
    tasks.add(initDropoutRandomSeed());
    tasks.add(incrementDropoutRandomSeedTask());
  }

  // Depending on anchor return types specified by the user, some
  // tensors may need to be added to the graph to keep track of
  // batch count.
  if (ir().getDataFlow().isBatchCountingRequired()) {
    tasks.add(initBatchCounterTensorsTask());
    tasks.add(updateBatchCountTask(progs.mainProgramFragment()));
  }

  // stream-to-host tensors : 1) make streams 2) make copy programs
  // note that the order in which tasks are added does not matter,
  // they will be topologically sorted before running
  if (useSyntheticData() == false) {
    for (auto anchorId : ir().getDataFlow().anchors()) {
      Tensor *tensor = ir().getTensor(anchorId);

      // 1
      tasks.add(streamToHostTask(tensor));
      // 2
      switch (ir().getDataFlow().art(anchorId).id()) {
      // Copy program runs after every batch
      case (AnchorReturnTypeId::ALL): {
        tasks.add(toHostTask(tensor, mainProgramFragment()));
        break;
      }
      // Copy program runs at the end of the step
      case (AnchorReturnTypeId::FINAL): {
        tasks.add(toHostTask(tensor, progs.toHostFinalCopyFragment()));
        break;
      }
      // Copy program runs at the end of every N batches
      case (AnchorReturnTypeId::EVERYN): {
        tasks.add(toHostEveryNBatchesTask(tensor,
                                          ir().getDataFlow().art(anchorId).rp(),
                                          mainProgramFragment()));
        break;
      }
      }
    }

    // create Program to write optimizer tensors to device
    for (Tensor *tensor : ir().optimizerTensors()) {
      tasks.add(fromHostTask(tensor, progs.streamOptimizerFromHostFragment()));
    }

    for (Tensor *tensor : ir().dataStreamTensors()) {
      tasks.add(fromHostTask(tensor, mainProgramFragment()));
    }
  }

  addOpTasks(tasks);

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
    auto poponnxTensorTileMap = std::getenv("POPONNX_TENSOR_TILE_MAP");
    if (poponnxTensorTileMap && strcmp(poponnxTensorTileMap, "") != 0) {
      saveTensorTileMap(poponnxTensorTileMap);
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

std::string Devicex::getPoponnxCachePath() {
  return ir().getSessionOptions().cachePath + ".poponnx";
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

    // save the poponnx ir hash
    auto poponnxCachePath = getPoponnxCachePath();
    std::ofstream poponnxFs(poponnxCachePath, std::ofstream::binary);
    logging::devicex::debug("Saving poponnx ir hash to '{}'", poponnxCachePath);
    SavedInfo savedInfo(*this);
    savedInfo.serialize(poponnxFs);
  };
}

void Devicex::tryLoadExecutable() {
  auto warn = [&](const std::string &msg) {
    logging::devicex::warn("Unable to load cached poplar::Executable, {}", msg);
  };

  auto cachePath    = ir().getSessionOptions().cachePath;
  auto cacheEnabled = ir().getSessionOptions().enableEngineCaching;

  if (cacheEnabled && !cachePath.empty() &&
      deviceInfo->getType() == DeviceType::Ipu) {
    // load the poponnx ir hash
    auto poponnxCachePath = getPoponnxCachePath();
    std::ifstream poponnxFs(poponnxCachePath, std::ifstream::binary);
    if (poponnxFs.is_open()) {
      if (SavedInfo(*this) == SavedInfo::deserialize(poponnxFs)) {
        auto poplarCachePath = getPoplarCachePath();
        std::ifstream poplarFs(poplarCachePath, std::ifstream::binary);
        if (poplarFs.is_open()) {
          logging::devicex::trace("Loading poplar Executable from '{}'",
                                  cachePath);
          cachedExecutable.emplace(poplar::Executable::deserialize(poplarFs));
          usingCachedExecutable = true;
        } else {
          warn(fmt::format("could not open file `{}'", poplarCachePath));
        }
      } else {
        warn("ir hashes differ");
      }
    } else {
      warn(fmt::format("could not open file `{}'", poponnxCachePath));
    }
  }
}

TaskId Devicex::streamFromHostTaskId(TensorId id) const {
  return "streamFromHostTask_" + id;
}

TaskId Devicex::setInitTensorValTaskId(TensorId id) const {
  return "setInitTensorValTask_" + id;
}

TaskId Devicex::streamToHostTaskId(TensorId id) const {
  return "streamToHostTask_" + id;
}

TaskId Devicex::fromHostTaskId(TensorId id) const {
  return "fromHostTask_" + id;
}

TaskId Devicex::toHostTaskId(TensorId id) const { return "toHostTask_" + id; }

TaskId Devicex::initBatchCounterTensorsTaskId() const {
  return "initBatchCounterTensorsTask";
}

TaskId Devicex::updateBatchCountTaskId() const {
  return "updateBatchCountTask";
}

TaskId Devicex::initDropoutRandomSeedId() const {
  return "initDropoutRandomSeed";
}

TaskId Devicex::initTensorTaskId(TensorId id) const {
  return "initTensorTaskId_" + id;
}

TaskId Devicex::opTaskId(Op *op) const {

  std::stringstream ss;
  ss << "fromOpTask_" << op->id << '_' << op->opid;
  return ss.str();
}

PopStreamId Devicex::h2dId(TensorId id) const { return "h2d_" + id; }

PopStreamId Devicex::d2hId(TensorId id) const { return "d2h_" + id; }

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
                            poplar::program::Sequence &sq) const {

  auto f = [&sq, tensor, this]() {
    logging::devicex::debug("Adding poplar::program::Copy to host " +
                            tensor->id);

    sq.add(poplar::program::Copy(tensors.get(tensor->id),
                                 toHostStreams.at(tensor->id),
                                 doRearrangeOnHost(tensor)));
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

PriTask Devicex::toHostEveryNBatchesTask(Tensor *tensor,
                                         int N,
                                         poplar::program::Sequence &sq) {

  auto f = [&sq, tensor, N, this]() {
    logging::devicex::debug(
        "Adding conditional poplar::program::Copy to host " + tensor->id);

    poplar::Tensor isNthBatch = batchCountCheckingTensors.at(N);

    poplar::program::Sequence copyseq;
    copyseq.add(poplar::program::Copy(tensors.get(tensor->id),
                                      toHostStreams.at(tensor->id),
                                      doRearrangeOnHost(tensor)));

    // Placeholder 'do nothing' branch if not running copy program
    poplar::program::Sequence emptyseq;

    sq.add(poplar::program::If(isNthBatch, copyseq, emptyseq));
  };

  return {+1e6, // writes to host: always as early as possible
          toHostTaskId(tensor->id),
          {
              // the dependencies:
              updateBatchCountTaskId(),       // updating poplar::Tensor task,
              streamToHostTaskId(tensor->id), // poplar::Stream creation task,
              taskWhichCreates(tensor->id)    // poplar::Tensor creation task.
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

  case DataType::UNDEFINED:
  case DataType::UINT8:
  case DataType::INT8:
  case DataType::UINT16:
  case DataType::INT16:
  case DataType::INT64:
  case DataType::STRING:
  case DataType::BFLOAT16:
  case DataType::DOUBLE:
  case DataType::UINT32:
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

bool Devicex::useSyntheticData() {
  return (ir().getSessionOptions().ignoreData);
}

bool Devicex::isDropoutRandomSeedRequired() const {
  return requiresDropoutRandomSeed;
}

void Devicex::setDropoutRandomSeedIsRequired(bool isRequired) {
  requiresDropoutRandomSeed = isRequired;
}

bool PopPrograms::hasBeenRecomputed(OpId id) const {
  auto itHas = (beenRecomputed.find(id) != beenRecomputed.end());
  return itHas;
}

void PopPrograms::recordRecomputed(OpId id) { beenRecomputed.insert(id); }

std::string Devicex::dropoutRandomSeedTensorId() const {
  return "dropoutRandomSeed";
}

const poplar::Tensor *Devicex::getDropoutRandomSeed() const {
  return &dropoutRandomSeed;
}

poplar::program::Sequence &PopPrograms::recomputeFragment(OpId id) {
  auto found = recomputeSeqs.find(id);
  if (found != recomputeSeqs.end()) {
    return found->second;
  }
  recomputeSeqs.insert({id, poplar::program::Sequence{}});
  return recomputeSeqs[id];
}

} // namespace popx
} // namespace poponnx
