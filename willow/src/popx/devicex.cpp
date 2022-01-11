// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <random>
#include <set>
#include <thread>
#include <tuple>
#include <utility>
#include <popart/popx/creatorx.hpp>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/range/algorithm/find.hpp>
#include <boost/range/algorithm_ext.hpp>

#include <filereader.hpp>
#include <gcl/TileAllocation.hpp>
#include <pva/pva.hpp>
#include <poplar/CSRFunctions.hpp>
#include <poplar/CycleCount.hpp>
#include <poplar/RandomSeed.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/Zero.hpp>
#include <popops/codelets.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>
#include <poprithms/logging/timepartitionlogger.hpp>
#include <poputil/exceptions.hpp>
#include <popx/rng/rngstatelowering.hpp>
#include <popart/devicemanager.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/liveness.hpp>
#include <popart/logging.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/devicexmanager.hpp>
#include <popart/popx/executablex.hpp>
#include <popart/popx/executablexserialization.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/poplaroptionsx.hpp>
#include <popart/popx/popopx.hpp>
#include <popart/popx/pritask.hpp>
#include <popart/recompute.hpp>
#include <popart/stepio.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/randomsetup.hpp>
#include <popart/variablesettings.hpp>
#include <poparttracepoint.hpp>

#include <stepiosplitter.hpp>

namespace popart {
namespace popx {

Devicex::Datastream::Datastream(Tensor *t, PopStreamId s)
    : tensor(t), streamId(s), io(nullptr) {}

TensorId Devicex::Datastream::getTensorId() { return tensor->id; }

Devicex::InputDatastream::InputDatastream(Tensor *t, PopStreamId s)
    : Datastream(t, s) {}

Devicex::PrefetchCallback::PrefetchCallback(
    std::shared_ptr<InputDatastream> ds_)
    : ds(ds_) {}

poplar::StreamCallback::Result Devicex::PrefetchCallback::prefetch(void *dest) {
  POPART_TRACEPOINT();
  if (ds->readPrefetch(dest)) {
    return poplar::StreamCallback::Result::Success;
  } else {
    return poplar::StreamCallback::Result::NotAvailable;
  }
}

void Devicex::PrefetchCallback::fetch(void *dest) {
  POPART_TRACEPOINT();
  ds->read(dest);
}

void Devicex::PrefetchCallback::complete() {
  POPART_TRACEPOINT();
  ds->readComplete();
}

void Devicex::InputDatastream::read(void *ptr) {
  POPART_TRACEPOINT();
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
      throw runtime_error(ss.str());
    }

  } else {
    logging::devicex::warn(
        "No stepio set for tensor {} stream {}", getTensorId(), streamId);
  }
}

bool Devicex::InputDatastream::readPrefetch(void *ptr) {
  POPART_TRACEPOINT();
  if (io) {

    ConstVoidData data = io->in(getTensorId(), tensor->info.nelms(), true);

    if (data.data == nullptr) {
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
        throw runtime_error(ss.str());
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
  POPART_TRACEPOINT();
  if (io) {
    io->inComplete(getTensorId(), tensor->info.nelms());
  }
}

Devicex::OutputDatastream::OutputDatastream(Tensor *t, PopStreamId s)
    : Datastream(t, s) {}

void Devicex::OutputDatastream::write(void *ptr) {
  POPART_TRACEPOINT();
  if (io) {
    MutableVoidData data = io->out(getTensorId(), tensor->info.nelms());
    memcpy(data.data, ptr, tensor->info.nbytes());
    io->outComplete(getTensorId());
  } else {
    logging::devicex::warn(
        "No stepio set for tensor {} stream {}", getTensorId(), streamId);
  }
}

void Devicex::run(PopPrograms::ProgramIndex ind, const std::string debugName) {
  POPART_TRACEPOINT();
  if (isEngineLoaded() == false) {
    logging::devicex::debug("Reloading engine & connecting streams");
    loadEngineAndConnectStreams();
  }
  pEngine->run(ind, debugName);
}

void Devicex::weightsToHost() {
  POPART_TRACEPOINT();
  if (ir().useSyntheticData() == false) {
    logging::devicex::debug("Writing weights to host");
    pEngine->disableExecutionProfiling();
    // Weights on the IPU
    run(PopPrograms::ProgramIndex::WeightstoHost, "WeightsToHost");
    // Weights in the remote buffers
    remoteBufferWeightsToHost();
    logging::devicex::debug("Writing weights to host complete.");
  }
}

void Devicex::remoteBufferWeightsToHost() {
  POPART_TRACEPOINT();
  for (auto *tensor : executable_.getWeightTensors()) {
    const auto &initId = tensor->id;
    if (tensor->tensorLocationInfo.isRemote()) {
      logging::devicex::debug("remoteBufferWeightsToHost: {}, [type={}]",
                              initId,
                              tensor->tensorType());
      auto remoteBufferInfo = tensor->tensorLocationInfo.getRemoteBufferInfo();
      char *data0           = d2hWeightBuffers[initId].data();

      CommGroup commGroup =
          tensor->getVariableSettings().getSharedVariableDomain();

      unsigned replicas = getReplicationFactor();
      unsigned returned =
          tensor->getVariableSettings().numReplicasReturningVariable(replicas);

      if (!replicas) {
        logging::devicex::err("Replicas == {}", replicas);
      }
      if (!returned) {
        logging::devicex::err("Returned == {}", returned);
      }

      unsigned groups = commGroup.replicaGroupSize
                            ? (replicas / commGroup.replicaGroupSize)
                            : returned;
      unsigned replication_refactor = getReplicationFactor() / groups;

      unsigned nelms            = tensor->info.nelms();
      unsigned returnedPerGroup = returned / groups;

      // Lamba expression that does the reading op automatically
      auto copyFromRemoteBuffer = [&](char *from, unsigned replica_id) {
        pEngine->copyFromRemoteBuffer(
            lowering().getRemoteBufferName(remoteBufferInfo.first),
            from,
            static_cast<int>(remoteBufferInfo.second),
            replica_id);
      };

      for (unsigned group = 0; group < groups; group++) {
        // every group always returns at least one

        // defines the group by the first member and the increment between
        // groups
        unsigned group_main =
            tensor->getVariableSettings().getGroupRepresentative(group);
        unsigned group_increment = (commGroup.type == CommGroupType::Orthogonal)
                                       ? commGroup.replicaGroupSize
                                       : 1;

        if (tensor->tensorLocationInfo.isSharded()) {
          // Replicated weight sharding, each replica holds 1/repfactor
          // parts of the weight
          const auto &cbr =
              executable_.getCollectiveBalancedHostRearrangement(tensor->id);

          auto elemSize =
              static_cast<int64_t>(tensor->info.getDataTypeInfo()->nbytes());
          auto nelms = cbr.getNumRearrangedTensorElems();

          // Temporary buffer that can hold the padded weight shards
          // from all replicas
          std::vector<char> tmp(nelms * elemSize);

          for (unsigned group_member = 0; group_member < replication_refactor;
               group_member++) {
            unsigned replica_id = (group_main + group_member) * group_increment;
            copyFromRemoteBuffer(
                &tmp[replica_id * nelms / replication_refactor * elemSize],
                replica_id);
          }

          unsigned address;
          if (returned == groups) {
            address = group * tensor->info.nelms() * elemSize;
          } else if (returned == getReplicationFactor()) {
            address = group_main * tensor->info.nelms() * elemSize;
          } else {
            throw internal_error(
                "Attempting to return an unsuported number of "
                "weight replicas: Returned (r) = {}, Groups (G)"
                " = {}, Replication Factor (R) = {}. r != G && "
                "r != R",
                returned,
                groups,
                getReplicationFactor());
          }

          cbr.undoRearrangeForCollective(&tmp[0], &data0[address], elemSize);

          for (unsigned group_member = 1; group_member < returnedPerGroup;
               group_member++) {
            unsigned replica_id = (group_main + group_member) * group_increment;
            unsigned address_ =
                replica_id * tensor->info.nelms() / returned * elemSize;
            memcpy(
                &data0[address_],
                &data0[address],
                static_cast<int64_t>(tensor->info.getDataTypeInfo()->nbytes()));
          }
        } else {
          for (unsigned group_member = 0; group_member < returnedPerGroup;
               group_member++) {
            unsigned replica_id = (group_main + group_member) * group_increment;
            unsigned long index =
                replica_id * nelms *
                static_cast<int64_t>(tensor->info.getDataTypeInfo()->nbytes()) /
                returned;
            unsigned id = group * returnedPerGroup + group_member;
            copyFromRemoteBuffer(&data0[index], id);
          }
        }
      }
    }
  }
}

void Devicex::readWeights(const IWeightsIO &weights) {
  POPART_TRACEPOINT();
  // Better to do this the otherway round
  for (auto *tensor : executable_.getWeightTensors()) {
    const auto &id = tensor->id;
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
  POPART_TRACEPOINT();
  // Better to do this the other way round
  // Also : should check that all weights have valid names
  for (auto *tensor : executable_.getWeightTensors()) {
    const auto &id = tensor->id;
    if (weights.contains(id)) {
      MutableVoidData stepout = weights.weight(id);
      tensor->tensorData()->resetData(stepout.info, stepout.data);
    }
  }
}

void Devicex::weightsToHost(
    const std::map<TensorId, MutableVoidData> &onnxModelData) {
  POPART_TRACEPOINT();

  if (!prepareHasBeenCalled()) {
    throw runtime_error(
        "Devicex::prepare() must be called before Devicex::weightsToHost(const "
        "std::map<TensorId, MutableVoidData> &) is called.");
  }

  if (ir().useSyntheticData() == false) {
    logging::devicex::debug("Writing weights to host");
    // write weights from IPU to host stream memory points

    pEngine->disableExecutionProfiling();
    // Weights on the IPU
    run(PopPrograms::ProgramIndex::WeightstoHost, "WeightsToHost");
    // Weights in the remote buffers
    remoteBufferWeightsToHost();

    logging::devicex::debug("Writing weights to ONNX ModelProto");
    // copy from the host stream memory points to the
    // addresses on onnxModelData
    for (auto *tensor : executable_.getWeightTensors()) {
      const auto &id = tensor->id;
      if (!ir().storingIsDisabledForTensor(tensor)) {
        auto found = onnxModelData.find(id);
        if (found == onnxModelData.end()) {
          std::ostringstream oss;
          oss << "No TensorId " << id
              << " in final host destination map. The TensorIds are [ ";
          for (auto x : onnxModelData) {
            oss << x.first << ' ';
          }
          oss << ']';
          throw runtime_error(oss.str());
        }
        MutableVoidData mv_data = found->second;
        hostStreamToHost(mv_data, id);
      }
    }
  }
}

std::map<std::string, std::vector<uint64_t>> Devicex::cycleCountTensorToHost() {
  if (ir().getSessionOptions().instrumentWithHardwareCycleCounter) {
    // Calls the copy from device to host
    logging::devicex::debug("Writing cycle count to host");
    pEngine->disableExecutionProfiling();
    run(PopPrograms::ProgramIndex::CycleCountTensortoHost,
        "CycleCountTensorToHost");
    logging::devicex::debug("Writing cycle count to host complete.");

    return cycleCount;
  } else {
    throw runtime_error("SessionOption 'instrumentWithHardwareCycleCounter' "
                        "must be set to true in order to measure cycle count");
  }
}

Devicex::~Devicex() = default;

Devicex::Devicex(Executablex &exe, std::shared_ptr<DeviceInfo> deviceInfo_)
    : executable_(exe), deviceInfo(deviceInfo_), prepareHasBeenCalled_(false) {
  POPART_TRACEPOINT();

  logging::devicex::info("Setting selected device: {}", *deviceInfo);

  if (ir().getSessionOptions().enablePrefetchDatastreams) {
    logging::devicex::info("Setting engine options for prefetch data streams "
                           "(exchange.streamBufferOverlap = hostRearrangeOnly, "
                           "exchange.enablePrefetch = true");
    lowering().engineOptions.set("exchange.streamBufferOverlap",
                                 "hostRearrangeOnly");
    lowering().engineOptions.set("exchange.enablePrefetch", "true");
  }

  if (ir().getSessionOptions().enableDistributedReplicatedGraphs) {
    logging::devicex::info("Setting firstRuntimeReplica {}",
                           ir().getSessionOptions().globalReplicaOffset);

    logging::devicex::info("Setting numberRuntimeReplica {}",
                           ir().getSessionOptions().replicatedGraphCount);

    std::string firstRuntimeReplica =
        std::to_string(ir().getSessionOptions().globalReplicaOffset);
    std::string numberRuntimeReplica =
        std::to_string(ir().getSessionOptions().replicatedGraphCount);

    lowering().engineOptions.set("target.syncReplicasIndependently", "true");
    lowering().engineOptions.set("target.firstRuntimeReplica",
                                 firstRuntimeReplica);
    lowering().engineOptions.set("target.numberRuntimeReplica",
                                 numberRuntimeReplica);
  }

  // The engine option `target.deterministicWorkers=true` ensures that random
  // behaviour is deterministic on all hardware but comes at the cost of
  // some performance. Note that we expect actual random Ops to be explicitly
  // seeded (so they are not affected) so the only time we actually need this
  // option is when the user enables stochastic rounding. We set this to "false"
  // when stochastic rounding is not enabled for a small performance boost. Note
  // that we avoid setting the option all together if the user sets it
  // explicitly.
  if (ir().getSessionOptions().engineOptions.find(
          "target.deterministicWorkers") ==
      ir().getSessionOptions().engineOptions.end()) {
    auto detWorkerValue =
        (ir().getSessionOptions().enableStochasticRounding) ? "true" : "false";
    lowering().engineOptions.set("target.deterministicWorkers", detWorkerValue);
  }

  for (auto it : ir().getSessionOptions().engineOptions) {
    logging::devicex::info(
        "Setting engine option {} = {}", it.first, it.second);
    lowering().engineOptions.set(it.first, it.second);
  }

  for (auto it : ir().getSessionOptions().reportOptions) {
    logging::devicex::info(
        "Setting report option {} = {}", it.first, it.second);
    lowering().reportOptions.set(it.first, it.second);
  }

  executable_.lowering().setDevicex(this);
}

const Ir &Devicex::ir() const { return lowering().ir(); }
IrLowering &Devicex::lowering() { return executable_.lowering(); }
const IrLowering &Devicex::lowering() const { return executable_.lowering(); }

void Devicex::weightsFromHost() {
  POPART_TRACEPOINT();
  if (ir().useSyntheticData() == false) {
    logging::devicex::debug("Writing weights from host, ");
    pEngine->disableExecutionProfiling();
    // Weights in the remote buffers
    remoteBufferWeightsFromHost();
    // Weights on the IPU

    initializeH2dWeightBuffers();
    run(PopPrograms::ProgramIndex::WeightsFromHost, "WeightsFromHost");
    deinitializeH2dWeightBuffers();

    logging::devicex::debug("done.");
  }
}

void Devicex::remoteBufferWeightsFromHost() {
  POPART_TRACEPOINT();
  if (isEngineLoaded() == false) {
    loadEngineAndConnectStreams();
  }
  for (auto *tensor : executable_.getWeightTensors()) {
    const auto &initId = tensor->id;
    if (tensor->tensorLocationInfo.isRemote()) {
      logging::devicex::debug("remoteBufferWeightsFromHost: {}", initId);
      auto remoteBufferInfo = tensor->tensorLocationInfo.getRemoteBufferInfo();
      char *data0           = static_cast<char *>(tensor->tensorData()->data());

      if (tensor->tensorLocationInfo.isSharded()) {
        // Replicated weight sharding, each replica holds 1/repfactor
        // parts of the weight
        const auto &cbr =
            executable_.getCollectiveBalancedHostRearrangement(initId);

        auto elemSize =
            static_cast<int64_t>(tensor->info.getDataTypeInfo()->nbytes());
        auto nelms = cbr.getNumRearrangedTensorElems();

        // Temporary buffer that can hold the padded weight shards
        // for all replicas
        std::vector<char> tmp(nelms * elemSize);

        // Rearrange weights into tmp buffer
        cbr.rearrangeForCollective(data0, &tmp[0], elemSize);

        for (unsigned replica_id = 0; replica_id < getReplicationFactor();
             ++replica_id) {
          // 1/repfactor weight shard to each replica
          pEngine->copyToRemoteBuffer(
              &tmp[replica_id * nelms / getReplicationFactor() * elemSize],
              lowering().getRemoteBufferName(remoteBufferInfo.first),
              static_cast<int>(remoteBufferInfo.second),
              replica_id);
        }
      } else {
        for (unsigned replica_id = 0; replica_id < getReplicationFactor();
             ++replica_id) {
          // Identical weights to each replica
          pEngine->copyToRemoteBuffer(
              data0,
              lowering().getRemoteBufferName(remoteBufferInfo.first),
              static_cast<int>(remoteBufferInfo.second),
              replica_id);
        }
      }
    }
  }
}

void Devicex::optimizerFromHost() {
  POPART_TRACEPOINT();
  if (ir().useSyntheticData() == false) {
    logging::devicex::debug("Writing optimizer from host, ");
    pEngine->disableExecutionProfiling();
    run(PopPrograms::ProgramIndex::OptimizerFromHost, "OptimizerFromHost");
    logging::devicex::debug("done.");
  }
}

// Copy from the host end of a d2h stream, to some final host memory.
// This is the step which follows a copy from device to host.
// poplar::Streams cannot write to an arbitrary dynamic address,
// they are connected to a fixed host address. This function copies
// from that fixed address to a dynamic address (mv_data).
void Devicex::hostStreamToHost(const MutableVoidData &mv_data, TensorId id) {
  POPART_TRACEPOINT();

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
  logging::devicex::debug(
      "       {} {}", id, executable_.getTensor(id)->info.shape());

  // We confirm that the sizes of src and dst are the same
  if (nbytes_src != nbytes_dst) {
    std::stringstream errms;
    errms << "sizes (in bytes) of src (" << nbytes_src << ") and dst ("
          << nbytes_dst << ") differ in hostStreamToHost for " << id;
    throw runtime_error(errms.str());
  }

  std::memcpy(dst, src, nbytes_src);
}

void Devicex::anchorsHostToHostStreams(IStepIO &stepio) {
  POPART_TRACEPOINT();

  if (ir().useSyntheticData() == false) {
    if (isEngineLoaded() == false) {
      loadEngineAndConnectStreams();
    }
    std::string prefix = "     ";
    logging::devicex::debug(prefix + "Copying to h2d stream address(es) ");
    if (stepIoSplitter) {
      stepIoSplitter->setUpstreamIo(&stepio);
    } else {
      throw runtime_error("StepIO splitter has not been initialised");
    }
  }
}

void Devicex::anchorsHostFromHostStreams(IStepIO &stepio) {
  POPART_TRACEPOINT();

  if (ir().useSyntheticData() == false) {
    if (isEngineLoaded() == false) {
      loadEngineAndConnectStreams();
    }
    std::string prefix = "     ";
    logging::devicex::debug(prefix + "Copying from d2h stream address(es) ");
    if (stepIoSplitter) {
      stepIoSplitter->setUpstreamIo(&stepio);
    } else {
      throw runtime_error("StepIO splitter has not been initialised");
    }
  }
}

void Devicex::run(IStepIO &stepio, std::string debugName) {
  POPART_TRACEPOINT();

  if (!prepareHasBeenCalled()) {
    throw runtime_error("Devicex::prepare() must be called before"
                        " Devicex::run(const IStepIO &) is called.");
  }

  // Check that the input and output buffers have the correct number of
  // elements. As run(.) is called multiple times during a user's session, the
  // check is only performed in the first call to run, under the assumption that
  // the user is unlikely to change the size of buffers between runs.
  if (nCallsToRun == 0 && stepio.runtimeAssertsEnabled()) {
    stepio.assertNumElements(executable_);
  }

  logging::devicex::debug("Performing one step: ");

  // Reconnect input streams.
  reconnectInputStreams();

  // Configure the inputstreams
  anchorsHostToHostStreams(stepio);

  // Configure the outputstreams
  anchorsHostFromHostStreams(stepio);

  pEngine->enableExecutionProfiling();
  run(PopPrograms::ProgramIndex::Program, debugName);

  ++nCallsToRun;
}

void Devicex::connectRandomSeedStream() {
  POPART_TRACEPOINT();

  // Generate a separate random seed for each replicant.
  for (uint16_t replicaId = 0; replicaId < getReplicationFactor();
       ++replicaId) {

    auto callback = [this, replicaId](void *ptr) {
      const Tensor *seedTensor = executable_.getSeedTensor();
      const uint64_t *seedVal =
          reinterpret_cast<const uint64_t *>(seedTensor->tensorData()->data());
      const unsigned globalReplicaId =
          replicaId + (getReplicationFactor() * getGlobalReplicaOffset());
      logging::devicex::debug("Updating random seed for globalReplica:{} to {}",
                              globalReplicaId,
                              *seedVal);
      uint64_t *data = reinterpret_cast<uint64_t *>(ptr);
      data[0]        = *seedVal;
    };

    pEngine->connectStreamToCallback(
        lowering().h2dId(RandomSetup::getStreamedSeedTensorId()),
        replicaId,
        callback);
  }
}

void Devicex::connectRngStateStream() {
  int totalNumTiles = deviceInfo->getNumIpus() * deviceInfo->getTilesPerIPU();
  int rngSize       = totalNumTiles * deviceInfo->getNumWorkerContexts() *
                RngStateLowering::rngStateSizePerWorker *
                RngStateLowering::numRngStateTensors;
  for (uint16_t replicaId = 0; replicaId < getReplicationFactor();
       ++replicaId) {
    rngBuffer[replicaId] = std::vector<uint32_t>(rngSize);

    auto h2d_callback = [this, replicaId, rngSize](void *ptr) {
      uint32_t *data   = reinterpret_cast<uint32_t *>(ptr);
      uint32_t *buffer = rngBuffer[replicaId].data();
      std::copy(buffer, buffer + rngSize, data);
      logging::devicex::debug("Updating RNG state for replica:{}", replicaId);
    };

    auto d2h_callback = [this, replicaId, rngSize](void *ptr) {
      uint32_t *data   = reinterpret_cast<uint32_t *>(ptr);
      uint32_t *buffer = rngBuffer[replicaId].data();
      std::copy(data, data + rngSize, buffer);
      logging::devicex::debug("Retrieving RNG for replica:{}", replicaId);
    };

    pEngine->connectStreamToCallback(
        "h2d_rngStateTensor", replicaId, h2d_callback);
    pEngine->connectStreamToCallback(
        "d2h_rngStateTensor", replicaId, d2h_callback);
  }
}

void Devicex::setRandomSeedFromHost() {
  POPART_TRACEPOINT();
  if (ir().useSyntheticData() == false) {
    pEngine->disableExecutionProfiling();
    run(PopPrograms::ProgramIndex::SetRandomSeedFromHost, "SetRandomSeed");
  }
}

void Devicex::setRngStateFromHost() {
  if (1) { // Add Session option
    pEngine->disableExecutionProfiling();
    run(PopPrograms::ProgramIndex::RngStateFromHost, "SetRngState");
  }
}

std::vector<uint32_t> Devicex::getRngStateToHost() {
  // Reset the buffer
  logging::devicex::debug("Cleaning the rng buffer before receiving data");
  int totalNumTiles = deviceInfo->getNumIpus() * deviceInfo->getTilesPerIPU();
  int rngSize       = totalNumTiles * deviceInfo->getNumWorkerContexts() *
                RngStateLowering::rngStateSizePerWorker *
                RngStateLowering::numRngStateTensors;
  for (auto &buffer : rngBuffer) {
    buffer.second = std::vector<uint32_t>(rngSize);
  }
  pEngine->disableExecutionProfiling();
  run(PopPrograms::ProgramIndex::RngStateToHost, "GetRngState");
  logging::devicex::debug("Copying data to host");
  std::vector<uint32_t> rngState;
  for (uint16_t replicaId = 0; replicaId < getReplicationFactor();
       ++replicaId) {
    rngState.insert(rngState.end(),
                    rngBuffer[replicaId].begin(),
                    rngBuffer[replicaId].end());
  }
  return rngState;
}

void Devicex::setRngStateValue(const std::vector<uint32_t> rngState) {
  int totalNumTiles = deviceInfo->getNumIpus() * deviceInfo->getTilesPerIPU();
  int rngSize       = totalNumTiles * deviceInfo->getNumWorkerContexts() *
                RngStateLowering::rngStateSizePerWorker *
                RngStateLowering::numRngStateTensors;
  if (rngState.size() != rngSize) {
    throw runtime_error("Devicex::setRngStateValue received rngState of size "
                        "{}; was expecting size {}",
                        rngState.size(),
                        rngSize);
  }
  const uint32_t *seed_ptr = rngState.data();
  for (uint16_t replicaId = 0; replicaId < getReplicationFactor();
       ++replicaId) {
    rngBuffer[replicaId].assign(seed_ptr, seed_ptr + rngSize);
    seed_ptr += rngSize;
  }
}

// TODO consider moving the test in this function into the Ir (T12636)
unsigned Devicex::getAccumulationFactor() const {
  return lowering().getAccumulationFactor();
}

unsigned Devicex::getGlobalReplicationFactor() const {
  return lowering().getGlobalReplicationFactor();
}

unsigned Devicex::getGlobalReplicaOffset() const {
  return lowering().getGlobalReplicaOffset();
}

unsigned Devicex::getReplicationFactor() const {
  return lowering().getReplicationFactor();
}

bool Devicex::isReplicatedGraph() const {
  return lowering().isReplicatedGraph();
}

bool Devicex::isEngineLoaded() const { return engineIsLoaded; }

void Devicex::setEngineIsLoaded(bool isLoaded) { engineIsLoaded = isLoaded; }

void Devicex::loadEngineAndConnectStreams() {
  POPART_TRACEPOINT();
  if (deviceInfo->getConnectionType() == DeviceConnectionType::Never) {
    throw runtime_error("Trying to load an engine on an offline device");
  }
  if (!pEngine) {
    throw runtime_error("Trying to load an engine but no compiled engine. Did "
                        "you forget to call prepareDevice()?");
  }
  DevicexInfo &di = dynamic_cast<DevicexInfo &>(*deviceInfo);

  // Let the device info know that this devicex's engine
  // has most recently loaded its engine onto the poplar
  // device
  for (auto d : di.previouslyLoadedDevicexs) {
    d->setEngineIsLoaded(false);
  }
  di.previouslyLoadedDevicexs.insert(this);
  setEngineIsLoaded(true);

  if (di.getConnectionType() == DeviceConnectionType::OnDemand) {
    logging::devicex::debug("Attaching to device on demand");
    if (!di.attach()) {
      if (di.getOnDemandAttachTimeout() > 0) {
        di.tryAttachUntilTimeout();
      }
      // If still not attached, error
      if (!di.isAttached()) {
        throw runtime_error("Failed to attach to device");
      }
    } else {
      di.writeToDeviceAccessLog("attach");
    }
  }

  pEngine->load(di.getDevice());
  logging::devicex::info("Engine loaded");

  if (ir().useSyntheticData() == false) {
    logging::devicex::debug("Connecting initializer streams");

    for (auto *tensor : executable_.getWeightTensors()) {
      if (!tensor->hasProducer()) {
        const auto &id = tensor->id;
        if (!ir().streamingIsDisabledForTensor(tensor)) {
          logging::devicex::debug("   {}", tensor->str());

          auto replicationFactor =
              ir().getSessionOptions().replicatedGraphCount;
          auto groupCount =
              tensor->getVariableSettings().groupCount(replicationFactor);

          if (groupCount == 1 || groupCount == replicationFactor) {
            //  The underlying data of the tensor has the correct number and
            //  order of data elements for the stream to be able to directly
            //  copy from it.
            pEngine->connectStream(lowering().h2dId(id),
                                   tensor->tensorData()->data());
          } // else see initializeH2dWeightBuffers
        }
      }
    }

    // Random seed
    if (ir().getRequiresRandomSeed()) {
      connectRandomSeedStream();
    }

    // Rng
    if (ir().getSessionOptions().enableLoadAndOffloadRNGState) {
      connectRngStateStream();
    }

    logging::devicex::debug("Connecting optimizer streams");

    for (auto *tensor : executable_.getOptimizerTensors()) {
      logging::devicex::debug("   {}", tensor->str());
      pEngine->connectStream(lowering().h2dId(tensor->id),
                             tensor->tensorData()->data());
    }

    // The splitter needs to know how many input buffers and output buffers
    // to expect per replica. For input buffers this is a constant figure
    // for all tensors. For output buffers, it depends on the AnchorReturnType
    // for that tensor.
    const int bps = static_cast<unsigned>(ir().getDataFlow().batchesPerStep());
    const int accumFactor = ir().getSessionOptions().enableGradientAccumulation
                                ? ir().getSessionOptions().accumulationFactor
                                : 1;
    const int maxInputsPerRepl = bps * accumFactor;

    stepIoSplitter = std::make_unique<StepIOSplitter>(
        getReplicationFactor(),
        [=](const TensorId &id) { return maxInputsPerRepl; },
        [&](const TensorId &id) {
          return ir().getDataFlow().numOutFetchesPerRepl(
              ir().getSessionOptions(), id);
        });
    stepIoSplitter->reset();

    auto engineToInputStreamWithCallback = [&pEngine = pEngine,
                                            this](Tensor *tensor,
                                                  TensorId streamTensorId,
                                                  PopStreamId streamId) {
      auto replicationFactor = getReplicationFactor();
      for (auto replicationIndex = 0; replicationIndex < replicationFactor;
           ++replicationIndex) {

        logging::devicex::debug(
            "Connecting input stream {}@{}", tensor->id, replicationIndex);

        IStepIO *downstreamIo = stepIoSplitter->getDownstreamStepIO(
            streamTensorId, tensor->info, replicationIndex);

        std::shared_ptr<InputDatastream> ds =
            std::make_shared<InputDatastream>(tensor, streamId);
        ds->setStepIO(downstreamIo);

        this->inputStreams[std::make_tuple(tensor->id, replicationIndex)] = ds;

        auto callback = std::make_unique<PrefetchCallback>(ds);

        if (tensor->getReplicatedStreamMode() ==
                ReplicatedStreamMode::Broadcast &&
            replicationIndex != 0)
          break;

        pEngine->connectStreamToCallback(
            streamId, replicationIndex, std::move(callback));
      }
    };

    auto engineToOutputStreamWithCallback = [&pEngine = pEngine,
                                             this](Tensor *tensor,
                                                   PopStreamId streamId) {
      auto replicationFactor = getReplicationFactor();
      for (auto replicationIndex = 0; replicationIndex < replicationFactor;
           ++replicationIndex) {

        logging::devicex::debug(
            "Connecting output stream {}@{}", tensor->id, replicationIndex);

        IStepIO *downstreamIo = stepIoSplitter->getDownstreamStepIO(
            tensor->id, tensor->info, replicationIndex);

        std::shared_ptr<OutputDatastream> ds =
            std::make_shared<OutputDatastream>(tensor, streamId);
        ds->setStepIO(downstreamIo);

        this->outputStreams[std::make_tuple(tensor->id, replicationIndex)] = ds;

        auto callback = [ds](void *ptr) mutable { ds->write(ptr); };

        pEngine->connectStreamToCallback(streamId, replicationIndex, callback);
      }
    };

    // Variables can return different number of data based on their
    // VariableSettings this makes certain that the replicas that are required
    // to return their values do so
    auto engineToStreamVariables = [&pEngine          = pEngine,
                                    replicationFactor = getReplicationFactor()](
                                       char *data0,
                                       int64_t n_bytes,
                                       PopStreamId streamId,
                                       Tensor *tensor) {
      auto id = tensor->id;

      auto groups = tensor->getVariableSettings().groups(replicationFactor);
      auto returned =
          tensor->getVariableSettings().numReplicasReturningVariable(
              replicationFactor);

      for (auto g = 0; g < groups.size(); g++) {
        auto group     = groups[g];
        bool returnAll = replicationFactor == returned;

        for (auto i = 0; i < group.size(); i++) {
          // return if first in group
          // or returning all
          bool returning = (i == 0) || returnAll;
          auto replicaId = group[i];

          auto segment = (returnAll ? replicaId : g);

          char *data = &data0[(n_bytes * segment)];

          logging::devicex::debug("Connecting stream variable {}@{} -{}-> {}",
                                  id,
                                  replicaId,
                                  (returning ? "-" : "/"),
                                  segment);

          auto callback = [returning, data, n_bytes, id](void *ptr) mutable {
            if (returning) {
              char *re_data = reinterpret_cast<char *>(ptr);
              memcpy(data, re_data, n_bytes);
            }
          };
          pEngine->connectStreamToCallback(streamId, replicaId, callback);
        }
      }
    };

    logging::devicex::debug("Connected h2d input data streams");
    for (Tensor *tensor : executable_.getDataStreamTensors()) {
      logging::devicex::debug(" {}", tensor->id);
      engineToInputStreamWithCallback(
          tensor, tensor->id, lowering().h2dId(tensor->id));
    }
    // If using overlapped IO, there are no stream tensors, only host load
    // tensors, so we loop through those.
    logging::devicex::debug("Connected h2d host load data streams");
    for (auto idAndTensors : ir().getHostLoadTensors()) {
      logging::devicex::debug(" {}", idAndTensors.first);
      engineToInputStreamWithCallback(idAndTensors.second.front(),
                                      idAndTensors.first,
                                      lowering().h2dId(idAndTensors.first));
    }

    logging::devicex::debug("Connected d2h anchor data streams");
    for (Tensor *tensor : executable_.getAnchorTensors()) {
      const auto &anchorId = tensor->id;
      bool isAnchorStream  = true;
      PopStreamId streamId = lowering().d2hId(anchorId, isAnchorStream);
      logging::devicex::debug(" {}", tensor->id);
      engineToOutputStreamWithCallback(tensor, streamId);
    }

    logging::devicex::debug("Connected d2h weight data streams");
    for (auto *tensor : executable_.getWeightTensors()) {
      if (!tensor->hasProducer()) {
        const auto &initId = tensor->id;
        int64_t n_bytes    = tensor->info.nbytes();
        logging::devicex::debug(
            "Connecting {} [type={}]", initId, tensor->tensorType());
        if (tensor->tensorType() == TensorType::Variable) {
          auto ret_factor =
              tensor->getVariableSettings().numReplicasReturningVariable(
                  getReplicationFactor());
          d2hWeightBuffers[initId] = std::vector<char>(ret_factor * n_bytes);
        } else {
          d2hWeightBuffers[initId] = std::vector<char>(n_bytes);
        }

        char *data0 = d2hWeightBuffers[initId].data();
        if (!ir().streamingIsDisabledForTensor(tensor)) {
          // Only connect non-cached tensor streams,
          // RemoteBuffer handled separately
          bool isAnchorStream  = false;
          PopStreamId streamId = lowering().d2hId(initId, isAnchorStream);
          logging::devicex::debug(" {}", initId);
          engineToStreamVariables(data0, n_bytes, streamId, tensor);
          logging::devicex::debug(
              "Created buffer (size {} B) and stream for {}",
              n_bytes,
              tensor->id);
        }
      }
    }
  }

  // Hardware cycle counter - connect stream even if synthetic data mode is
  // not off
  if (ir().getSessionOptions().instrumentWithHardwareCycleCounter) {
    for (auto &kv : cycleCount) {
      pEngine->connectStream(lowering().cycleCountStreamId(kv.first),
                             static_cast<void *>(kv.second.data()));
    }
  }

  setRandomSeedFromHost(); // Stream random seed value by default (prog empty if
                           // no randomness)
  if (ir().canTrain()) {
    executable_.updateOptimizerTensors();
    optimizerFromHost();
  }
}

void Devicex::reconnectInputStreams() {
  POPART_TRACEPOINT();
  logging::devicex::debug(
      "Reconnecting input streams, invalidating prefetches.");

  auto engineToInputStreamWithCallback = [&pEngine = pEngine,
                                          this](Tensor *tensor, TensorId id) {
    auto replicationFactor = getReplicationFactor();
    for (auto replicationIndex = 0; replicationIndex < replicationFactor;
         ++replicationIndex) {
      logging::devicex::debug(
          "Reconnecting input stream {}@{}", tensor->id, replicationIndex);

      auto callback = std::make_unique<PrefetchCallback>(
          this->inputStreams[std::make_tuple(tensor->id, replicationIndex)]);

      if (tensor->getReplicatedStreamMode() ==
              ReplicatedStreamMode::Broadcast &&
          replicationFactor != 0) {
        logging::devicex::debug("Tensor is broadcasted ({}) and should not be "
                                "streamed to non-zero replicas.",
                                tensor->getReplicatedStreamMode());
        break;
      }

      pEngine->connectStreamToCallback(
          lowering().h2dId(id), replicationIndex, std::move(callback));
    }
  };

  for (Tensor *tensor : executable_.getDataStreamTensors()) {
    // The data stream for a tensor won't exist if using synthetic data, so
    // don't try and recreate them.
    if (!ir().useSyntheticData() && !tensor->tensorLocationInfo.isRemote()) {
      engineToInputStreamWithCallback(tensor, tensor->id);
    }
  }
}

// go all the way to creating the engine and connecting streams
void Devicex::prepare() {

  const auto lifetimeTimer =
      ir().timePartitionLogger().scopedStopwatch("Preparing devicex");

  POPART_TRACEPOINT();
  if (!lowering().prepareGraphHasBeenCalled()) {
    lowering().prepareGraph();
  }

  logging::devicex::info(std::string("\nNext step is poplar Engine creation. "
                                     "Breakdown of compile time so far:\n") +
                         ir().timePartitionLoggerStr());

  if (ir().getSessionOptions().compileEngine) {

    const auto engineCreationTimer =
        ir().timePartitionLogger().scopedStopwatch("Engine creation");

    try {

      auto executable = lowering().getExecutable();
      pEngine.reset(
          new poplar::Engine(std::move(executable), lowering().engineOptions));

      if (!executable_.isDeserialized() && executable_.shouldSerialize()) {
        const std::string cachePath = ir().getSessionOptions().cachePath;
        serializeExecutable(executable_.getCachePath(cachePath));
      }

      logging::devicex::info(
          std::string("\npoplar Engine construction complete. Breakdown of "
                      "compile time:\n") +
          ir().timePartitionLoggerStr());

    } catch (const poplar::graph_memory_allocation_error &e) {
      // If the creation of the engine throw an exception due to memory
      // allocation i.e. the program does not fit show graph profile and
      // re-throw the exception In certain cases poplar will throw the error
      // without a graph profile. The following engine option needs to be set to
      // enable the graph profile in this case "debug.allowOutOfMemory":"true"
      logging::devicex::err("Memory allocation error : {}", e.what());
      throw devicex_memory_allocation_err(e, lowering().reportOptions);
    }
  } else {
    logging::devicex::info("Not compiling engine by request");
    return;
  }

  if (getDeviceInfo()->getConnectionType() == DeviceConnectionType::Never) {
    prepareHasBeenCalled_ = true;
    return;
  }

  if (ir().getSessionOptions().instrumentWithHardwareCycleCounter) {
    for (const auto &id : lowering().getCycleCountIds()) {
      // Allocate host side buffer for cycle count
      std::vector<uint64_t> zeros(getReplicationFactor(), 0);
      cycleCount[id] = zeros;
    }
  }

  prepareHasBeenCalled_ = true;
  setEngineIsLoaded(false);
}

void Devicex::doProfileChecks() const {
  if (pEngine == nullptr) {
    throw runtime_error("Session must have been prepared before a report can "
                        "be fetched");
  }
  if (executable_.isDeserialized()) {
    throw runtime_error(
        "Unable to get reports when using a cached executable.\n"
        "Either remove the cache file ({}), or \ndisable engine "
        "caching (userOptions.enableEngineCaching = false)",
        ir().getSessionOptions().cachePath);
  }
}

std::string Devicex::getSummaryReport(bool resetProfile) const {
  POPART_TRACEPOINT();
  doProfileChecks();

  std::stringstream ss;
  pEngine->printProfileSummary(ss, lowering().reportOptions);

  if (resetProfile) {
    pEngine->resetExecutionProfile();
  }
  return ss.str();
}

pva::Report Devicex::getReport() const {
  POPART_TRACEPOINT();
  doProfileChecks();
  return pEngine->getReport();
}

std::string Devicex::getSerializedGraph() const {
  POPART_TRACEPOINT();
  doProfileChecks();
  return lowering().getSerializedGraph();
}

std::set<TensorId> Devicex::getLinearlyCreatedInputTensors() const {
  return lowering().getLinearlyCreatedInputTensors();
}
std::set<TensorId> Devicex::getEfficientlyCreatedInputTensors() const {
  return lowering().getEfficientlyCreatedInputTensors();
}

void Devicex::serializeExecutable(const std::string &path) {
  POPART_TRACEPOINT();
  // If target directory does not exist, create it
  auto target = boost::filesystem::path(path);
  if (target.has_parent_path()) {
    auto targetDir = target.parent_path();
    if (!boost::filesystem::exists(targetDir)) {
      logging::devicex::warn("Specified directory not found. "
                             "Creating {} directory ",
                             targetDir);
      if (!boost::filesystem::create_directories(targetDir))
        throw error("Cannot create cache directory. Aborting.");
    }
  }
  std::string filename = path;
  if (boost::filesystem::is_directory(target)) {
    filename = logging::format("{}/executable.popef", filename);
    logging::devicex::warn(
        "{} is a directory, saving serialized Executablex to {}",
        target.string(),
        filename);
  } else {
    logging::devicex::info("Saving serialized Executablex to {}", filename);
  }
  std::ofstream out(filename, std::ofstream::binary);
  if (!out.is_open()) {
    throw error("Unable to open file '{}'", filename);
  }
  serializeExecutable(out);
}

void Devicex::serializeExecutable(std::ostream &out) {
  POPART_TRACEPOINT();
  popx::serialization::serializeEngineExecutable(
      out, pEngine.get(), &executable_, executable_.ir().getHash());
}

void Devicex::connectStream(const std::string &streamHandle,
                            void *host_buffer) {
  POPART_TRACEPOINT();
  pEngine->connectStream(streamHandle, host_buffer);
}

void Devicex::connectStreamToCallback(const std::string &streamHandle,
                                      std::function<void(void *)> callback,
                                      unsigned index) {
  POPART_TRACEPOINT();
  pEngine->connectStreamToCallback(streamHandle, index, callback);
}

void Devicex::connectHostFunction(
    const std::string &functionHandle,
    std::function<void(const void *const *, size_t, void *const *, size_t)>
        callback,
    unsigned index) {
  POPART_TRACEPOINT();

  using Callback =
      std::function<void(const void *const *, size_t, void *const *, size_t)>;
  struct Adaptor final : public poplar::HostCallback {
    Callback cb;

    Adaptor(Callback cb) : cb(std::move(cb)) {}

    void operator()(poplar::ArrayRef<const void *> inputs,
                    poplar::ArrayRef<void *> outputs) override {
      cb(inputs.data(), inputs.size(), outputs.data(), outputs.size());
    }
  };
  poplar::HostCallbackHandle cbHandle{
      std::unique_ptr<poplar::HostCallback>(new Adaptor(std::move(callback)))};
  pEngine->connectHostFunction(functionHandle, index, std::move(cbHandle));
}

void Devicex::copyFromRemoteBuffer(const PopStreamId buffer,
                                   void *w,
                                   int repeat_index,
                                   unsigned replication_index) {
  POPART_TRACEPOINT();
  pEngine->copyFromRemoteBuffer(buffer, w, repeat_index, replication_index);
}

void Devicex::copyToRemoteBuffer(void *w,
                                 const PopStreamId buffer,
                                 int repeat_index,
                                 unsigned replication_index) {
  POPART_TRACEPOINT();
  pEngine->copyToRemoteBuffer(w, buffer, repeat_index, replication_index);
}

void Devicex::initializeH2dWeightBuffers() {
  auto replicationFactor = getReplicationFactor();

  if (ir().useSyntheticData() == false) {
    logging::devicex::debug("Connecting temporary initializer streams");

    for (auto *tensor : executable_.getWeightTensors()) {
      auto id = tensor->id;
      if (!tensor->hasProducer()) {
        int64_t nbytes   = tensor->info.nbytes();
        int64_t fullsize = nbytes * replicationFactor;

        auto groups = tensor->getVariableSettings().groups(replicationFactor);

        if (groups.size() != 1 && groups.size() != replicationFactor) {
          h2dWeightBuffers[id] = std::vector<char>(fullsize);

          char *source = static_cast<char *>(tensor->tensorData()->data());
          char *destin = static_cast<char *>(h2dWeightBuffers[id].data());

          for (auto g = 0; g < groups.size(); g++) {
            auto group = groups[g];
            char *addr = source + (g * nbytes);
            for (auto r = 0; r < group.size(); r++) {
              auto replicaId = group[r];
              char *dest     = destin + (replicaId * nbytes);
              memcpy(dest, addr, nbytes);
            }
          }
          pEngine->connectStream(lowering().h2dId(id),
                                 h2dWeightBuffers[id].data());
        }
      }
    }
  }
}

void Devicex::deinitializeH2dWeightBuffers() {
  for (auto &pair : h2dWeightBuffers) {
    auto id = pair.first;
    logging::devicex::debug(" * deinitialize_h2dWeightBuffers: {}", id);
    h2dWeightBuffers[id].clear();
  }
}

} // namespace popx
} // namespace popart
