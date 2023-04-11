// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include "popart/util/expressionchecking.hpp"
#include <algorithm>
#include <boost/filesystem.hpp>
#include <cstdint>
#include <cstring>
#include <engineoptionscreator.hpp>
#include <fstream>
#include <functional>
#include <gcl/CollectiveBalancedReorder.hpp>
#include <iterator>
#include <map>
#include <memory>
#include <profilecacher.hpp>
#include <pva/pva.hpp>
#include <set>
#include <stepiosplitter.hpp>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <popdist/backend.hpp>
#include <popdist/collectives.hpp>
#include <poplar/ArrayRef.hpp>
#include <poplar/Engine.hpp>
#include <poplar/HostFunctionCallback.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/StreamCallback.hpp>
#include <poplar/exceptions.hpp>
#include <poprithms/logging/timepartitionlogger.hpp>
#include <popx/rng/rngstatelowering.hpp>
#include <popart/devicemanager.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/devicexmanager.hpp>
#include <popart/popx/executablex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/popefserializer.hpp>
#include <popart/stepio.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/transforms/randomsetup.hpp>
#include <popart/variablesettings.hpp>
#include <poparttracepoint.hpp>

#include "popart/dataflow.hpp"
#include "popart/datatype.hpp"
#include "popart/istepio.hpp"
#include "popart/popx/exchangebundle.hpp"
#include "popart/popx/popprograms.hpp"
#include "popart/replicagrouping.hpp"
#include "popart/replicatedstreammode.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/util.hpp"
#include "popart/util/expressionchecking.hpp"
#include "popart/voiddata.hpp"

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
    const bool isBroadcast =
        tensor->getReplicatedStreamMode() == ReplicatedStreamMode::Broadcast;
    ConstVoidData data =
        io->in(getTensorId(), tensor->info.nelms(), false, isBroadcast);

    const void *srcAddr = data.data;
    void *dstAddr       = ptr;

    auto srcInfo = data.info;
    auto dstInfo = tensor->info;

    // check the shape

    // Not sure how best to match the shape as the shape of the input does not
    // match the shape of the data.info. In fact that is a bit wrong now.

    // check the type

    // Because FP8 does not exist on host, check if we are reading UINT8
    // that the user wants to be interpreted as popart FLOAT8_*
    bool readingFP8 = dstInfo.dataType() == DataType::FLOAT8_143 ||
                      dstInfo.dataType() == DataType::FLOAT8_152;

    if (srcInfo.dataType() == dstInfo.dataType() ||
        (readingFP8 && srcInfo.dataType() == DataType::UINT8)) {
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
      ss << "Type discrepancy for tensor " << getTensorId()
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

    const bool isBroadcast =
        tensor->getReplicatedStreamMode() == ReplicatedStreamMode::Broadcast;
    ConstVoidData data =
        io->in(getTensorId(), tensor->info.nelms(), true, isBroadcast);

    if (data.data == nullptr) {
      return false;
    } else {

      const void *srcAddr = data.data;
      void *dstAddr       = ptr;

      auto srcInfo = data.info;
      auto dstInfo = tensor->info;

      // check the shape

      // Not sure how best to match the shape as the shape of the input does not
      // match the shape of the data.info. In fact that is a bit wrong now.

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
        ss << "Type discrepancy for tensor " << getTensorId()
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
    const bool isBroadcast =
        tensor->getReplicatedStreamMode() == ReplicatedStreamMode::Broadcast;
    io->inComplete(getTensorId(), tensor->info.nelms(), isBroadcast);
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

void Devicex::run(unsigned ind, const std::string debugName) {
  POPART_TRACEPOINT();
  if (isEngineLoaded() == false) {
    logging::devicex::debug("Reloading engine & connecting streams");
    loadEngineAndConnectStreams();
  }

  // Use a synchronized engine run in distributed environments. It defaults
  // to non-synchronized `engine.run(...)` in non-distributed environments.
  popdist::run(*pEngine, ind, debugName);
}

void Devicex::weightsToHost() {
  POPART_TRACEPOINT();
  if (ir().useSyntheticData() == false) {
    logging::devicex::debug("Writing weights to host");
    pEngine->disableExecutionProfiling();
    // Weights on the IPU
    run(PopPrograms::ProgramIndex::WeightsToHost, "WeightsToHost");
    // Weights in the remote buffers
    remoteBufferWeightsToHost();
    logging::devicex::debug("Writing weights to host complete.");
  }
}

void Devicex::popxlWeightsToTensorData() {
  POPART_TRACEPOINT();

  // Recall: attached <=> in the popxl.Session context.
  // If this is not the case, we do not run the WeightsToHost program, but still
  // transfer from the d2hWeightBuffers to the TensorData of the weights, so
  // that the results of the last popxl.Session.weights_to_host are reflected.
  // If the weights are already in sync, we do not do the transfer either.

  // Fetch weights from device into d2hWeightBuffers.
  weightsToHost();
  popxlMarkHostWeightsInSync();

  auto tensors = executable_.getWeightTensors();

  // Copy from d2hWeightBuffers into the tensors of the weights.
  d2hWeightBuffersToTensors(tensors);
}

bool Devicex::popxlAreHostWeightsInSync() {
  const auto tensors = executable_.getWeightTensors();

  bool all_in_sync = std::all_of(tensors.cbegin(), tensors.cend(), [](auto &t) {
    return t->tensorData()->getIsSyncedWithIPU();
  });

  return all_in_sync;
}

void Devicex::popxlMarkHostWeightsOutOfSync() {
  for (auto t : executable_.getWeightTensors()) {
    t->tensorData()->setIsSyncedWithIPU(false);
  }
}
void Devicex::popxlMarkHostWeightsInSync() {
  for (auto t : executable_.getWeightTensors()) {
    t->tensorData()->setIsSyncedWithIPU(true);
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
      // Collect information
      const auto remoteBufferInfo =
          tensor->tensorLocationInfo.getRemoteBufferInfo();
      char *data0          = getD2hWeightData(tensor);
      const auto data0Size = getD2hWeightBufferSize(tensor);
      const auto elemSize =
          static_cast<int64_t>(tensor->info.getDataTypeInfo()->nbytes());
      const unsigned nelms = tensor->info.nelms();

      const unsigned instanceReplicas    = getReplicationFactor();
      const unsigned globalReplicas      = getGlobalReplicationFactor();
      const unsigned globalReplicaOffset = getGlobalReplicaOffset();

      const auto grouping =
          tensor->getVariableSettings().getReplicaGrouping(globalReplicas);

      const unsigned numGroups     = grouping.getNumGroups();
      const unsigned realGroupSize = grouping.getGroupSize();

      // Get the number of replicas that return their copy of this variable
      const unsigned returned =
          tensor->getVariableSettings().numReplicasReturningVariable(
              globalReplicas);
      // How many instances each group returns.
      POPART_ASSERT_EQ(returned % numGroups, 0);
      const unsigned returnedPerGroup = returned / numGroups;

      const auto retrievalMode =
          tensor->getVariableSettings().getRetrievalMode();

      // Lambda expression that does the reading op automatically
      auto copyFromRemoteBuffer = [&](char *to, unsigned replicaId) {
        pEngine->copyFromRemoteBuffer(
            lowering().getExchangeBundle().getRemoteBufferName(
                remoteBufferInfo.first),
            to,
            static_cast<int>(remoteBufferInfo.second),
            replicaId);
      };

      if (tensor->tensorLocationInfo.isSharded()) {
        /*
        Iterate over each group, create a buffer to copy the cbr-padded shards
        from the replicas into, then `cbr.undoRearrangeForCollective` that
        buffer into data0.

        If multi-instance, when we hit a group member not on this instance,
        we simply skip the copyFromRemoteBuffer, which leaves the data for that
        shard at 0. Later, we perform a Sum-AllReduce across instances,
        resulting in the full data for the group residing on each instance.

        Note, like in the code for the unsharded case, we could reduce the
        iteration space by iterating over only the local replicas, instead of
        all the replicas and skipping if that replica is not on this instance.
        This is also arguably cleaner.
        However, rather than making a temporary cbr-padded buffer for only one
        group at a time, we would have to hold a single large buffer for all
        groups, copy from each local replica into the relevant index, then do
        the AllReduce. Though this results in less iterating, it requires
        `groups` times more memory to be live at once. We choose to go with the
        more memory-efficient route, as memory is quite likely to be a problem
        for large models, and iterating over all the replicas is likely barely
        a noticeable overhead.
       */
        for (unsigned group = 0; group < numGroups; group++) {
          // Replicated weight sharding, each replica holds parts of the
          // weight
          const auto &cbr =
              executable_.getCollectiveBalancedHostRearrangement(tensor->id);

          auto cbrNelms = cbr.getNumRearrangedTensorElems();
          auto cbrSize  = cbr.getReplicationFactor();
          // Throw internal_error because this should have been checked higher
          // up the stack.
          POPART_ASSERT_EQ(realGroupSize % cbrSize, 0);

          // Temporary buffer that can hold the padded weight shards
          // from all replicas in this group.
          std::vector<char> tmp(cbrNelms * elemSize);

          // Iterate over group members, collect the Tensor's shards.
          // When cbr_size < realGroupSize, we only collect shards from the
          // first cbr_size members.
          for (unsigned groupMember = 0; groupMember < cbrSize; groupMember++) {
            const unsigned globalReplicaId =
                grouping.getReplicaAt(group, groupMember);
            const int localReplicaId = globalReplicaId - globalReplicaOffset;
            if (globalReplicaOffset <= globalReplicaId &&
                globalReplicaId < globalReplicaOffset + instanceReplicas) {
              const unsigned shardNelms = cbrNelms / cbrSize;
              const unsigned addr       = groupMember * shardNelms * elemSize;
              copyFromRemoteBuffer(&tmp[addr], localReplicaId);
            }
          }

          // Calculate the address in the output buffer we want to write to.
          // If OnePerGroup, then data0 (the d2hWeightBuffers entry for this
          // weight) will have one entry per group.
          // If AllReplicas, then data0 will have one entry per replica. We
          // copy the data into the firstInGroup's index (then later copy this
          // to the indices of the other replicas in the group).
          unsigned address;
          if (returned == numGroups) {
            address = group * nelms * elemSize;
          } else if (returned == globalReplicas) {
            const unsigned firstInGroup = grouping.getReplicaAt(group);
            address                     = firstInGroup * nelms * elemSize;
          } else {
            throw internal_error(
                "Attempting to return an unsupported number of weight replicas:"
                " Returned (r) = {}, Groups (G) = {}, Global replication"
                " Factor (R) = {}. r != G && r != R",
                returned,
                numGroups,
                globalReplicas);
          }

          cbr.undoRearrangeForCollective(&tmp[0],
                                         tmp.size() * sizeof(tmp[0]),
                                         &data0[address],
                                         data0Size - address,
                                         elemSize);

          // Copy the contents of the collection to the space of the other
          // replicas. This means their collection is synthesized and will
          // always be the same.
          // Note, if returning only OnePerGroup, this loop will never run.
          for (unsigned groupMember = 1; groupMember < returnedPerGroup;
               groupMember++) {
            const unsigned replicaId =
                grouping.getReplicaAt(group, groupMember);
            const unsigned replicaAddr = replicaId * nelms * elemSize;
            const unsigned elements    = elemSize * nelms;
            memcpy(&data0[replicaAddr], &data0[address], elements);
          }

          // If multi-instance, we later do an AllReduce on data0 to reconstruct
          // the full tensor. Recall, within each entry in data0, there will be
          // missing data (set to 0) from the shards not on this instance.
        }
      } else {
        switch (retrievalMode) {
        case VariableRetrievalMode::AllReplicas:
          for (unsigned localReplicaId = 0; localReplicaId < instanceReplicas;
               localReplicaId++) {
            const auto globalReplicaId = localReplicaId + globalReplicaOffset;
            const unsigned long index  = globalReplicaId * nelms * elemSize;
            copyFromRemoteBuffer(&data0[index], localReplicaId);
          }

          // We have filled in the indices of data0 covered by the replicas of
          // this instance. Later, we will AllGather data0 with the other
          // instances to get the remaining data.
          break;
        case VariableRetrievalMode::OnePerGroup:
          for (unsigned localReplicaId = 0; localReplicaId < instanceReplicas;
               localReplicaId++) {
            const auto globalReplicaId = localReplicaId + globalReplicaOffset;
            const auto group           = grouping.getGroupAt(globalReplicaId);

            // Copy if this replica is the first in its group.
            if (grouping.getIndexInGroupAt(globalReplicaId) == 0) {
              const unsigned long index = group * nelms * elemSize;
              copyFromRemoteBuffer(&data0[index], localReplicaId);
            }
            // Else, memory should already be zero-d.
          }

          // If multi-instance:
          // For each group, the instances on which the first replica in the
          // group does not reside will all have 0 in the data0 entry for that
          // group. The instance on which the first replica in the group does
          // reside will have the actual value for that group. Thus, we can
          // collect the value of each group _on all instances_ by doing a sum
          // AllReduce on data0 across instances. This is more memory-efficient
          // than doing an AllGather + slice, with the same comms overhead. This
          // is done later in this function.
          break;
        default:
          throw internal_error("[Devicex::remoteBufferWeightsToHost] "
                               "Unsupported VariableRetrievalMode with int "
                               "value {}",
                               static_cast<int>(retrievalMode));
        }
      }

      // The cases are (OnePerGroup, AllReplicas) x (sharded, unsharded).
      // The above logic throughout the function explains what has happened to
      // data0 so far in each case, and what is further required here to handle
      // multiple instances.
      if (distributedReplicatedGraphsEnabled()) {
        if (retrievalMode == VariableRetrievalMode::OnePerGroup) {
          // OnePerGroup, sharded or unsharded
          // Note in OnePerGroup, data0 has numGroups entries.
          popdist::collectives::allReduceSum(
              data0, numGroups * nelms, popType(tensor->info));
        } else if (tensor->tensorLocationInfo.isSharded()) {
          // AllReplicas, sharded
          // Note in AllReplicas, data0 has globalReplicas entries.
          popdist::collectives::allReduceSum(
              data0, globalReplicas * nelms, popType(tensor->info));

          // TODO(T69345): AllGather then, per group: Reduce_Local then memcpy
        } else {
          // AllReplicas, unsharded
          // Note in AllReplicas, data0 has globalReplicas entries, and we are
          // gathering instanceReplicas entries from each instance.
          popdist::collectives::allGather(nullptr,
                                          data0,
                                          instanceReplicas * nelms,
                                          popType(tensor->info),
                                          popdist::defaultCommunicatorId(),
                                          true // inplace
          );
        }
      }
    }
  }
}

void Devicex::readWeights(const IWeightsIO &weights) {
  POPART_TRACEPOINT();
  // Better to do this the other way round
  for (auto *tensor : executable_.getWeightTensors()) {
    const auto &id = tensor->id;
    if (weights.contains(id)) {
      logging::devicex::debug("Reading weights (host stream -> host) for {}",
                              id);
      MutableVoidData stepout = weights.weight(id);
      hostStreamToHost(stepout, id, DownsampleStream::No);
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
      tensor->verifyMutableVoidInfo(stepout.info, getReplicationFactor());
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
    run(PopPrograms::ProgramIndex::WeightsToHost, "WeightsToHost");
    // Weights in the remote buffers
    remoteBufferWeightsToHost();

    d2hWeightBuffersToTensorData(onnxModelData);
  }
}

void Devicex::d2hWeightBuffersToTensors(const std::vector<Tensor *> &tensors) {

  std::map<TensorId, MutableVoidData> Wdata;

  // Prepare for d2hWeightBuffersToTensorData copy.
  std::transform(tensors.begin(),
                 tensors.end(),
                 std::inserter(Wdata, Wdata.end()),
                 [replicas = getGlobalReplicationFactor()](Tensor *t) {
                   MutableVoidData tData;
                   tData.data     = t->tensorData()->data();
                   auto hostShape = t->getVariableSettings().shapeOnHost(
                       t->info.shape(), replicas);

                   tData.info = t->info;
                   tData.info.set(tData.info.dataType(), hostShape);
                   return std::make_pair(t->id, tData);
                 });

  // Copy from d2hWeightBuffers into the TensorData of the weights.
  d2hWeightBuffersToTensorData(Wdata);
}

void Devicex::d2hWeightBuffersToTensorData(
    const std::map<TensorId, MutableVoidData> &onnxModelData) {
  if (!prepareHasBeenCalled()) {
    throw internal_error("Devicex::prepare() must be called before "
                         "Devicex::d2hWeightBuffersToTensorData(const "
                         "std::map<TensorId, MutableVoidData> &) is called.");
  }

  if (ir().useSyntheticData() == false) {
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
        hostStreamToHost(mv_data, id, DownsampleStream::GroupPrimary);
      }
    }
  }
}

std::map<std::string, std::vector<uint64_t>> Devicex::cycleCountTensorToHost() {
  if (ir().getSessionOptions().instrumentWithHardwareCycleCounter) {
    // Calls the copy from device to host
    logging::devicex::debug("Writing cycle count to host");
    pEngine->disableExecutionProfiling();
    run(PopPrograms::ProgramIndex::CycleCountTensorToHost,
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

  EngineOptionsCreator engineOptionsCreator{ir().getSessionOptions(),
                                            deviceInfo_->getTarget()};
  lowering().engineOptions = engineOptionsCreator.getOptionFlags();

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

    run(PopPrograms::ProgramIndex::WeightsFromHost, "WeightsFromHost");

    logging::devicex::debug("done.");
  }
}

void Devicex::buffersFromHost() {
  POPART_TRACEPOINT();
  logging::devicex::trace("Devicex::buffersFromHost");

  if (ir().useSyntheticData() == false) {
    logging::devicex::debug("Writing named buffers from host, ");
    pEngine->disableExecutionProfiling();
    // Weights in the remote buffers
    remoteBufferWeightsFromHost(true);
    // Weights on the IPU
    auto &indexMap = lowering().getProgramHandleIndexMap();
    auto it        = indexMap.find("copyNamedBuffers");
    if (it == indexMap.end()) {
      throw error(
          "[Devicex::buffersFromHost] copyNamedBuffers program not found");
    }

    run(it->second, "buffersFromHost");

    logging::devicex::debug("done.");
  }
}

void Devicex::remoteBufferWeightsFromHost(const bool isUpdate) {
  POPART_TRACEPOINT();
  if (isEngineLoaded() == false) {
    loadEngineAndConnectStreams();
  }
  for (auto *tensor : executable_.getWeightTensors()) {
    const auto &initId = tensor->id;
    if (tensor->tensorLocationInfo.isRemote()) {
      const auto &buffers = ir().getSessionOptions().updatableNamedBuffers;
      if (isUpdate &&
          std::find(buffers.begin(), buffers.end(), initId) == buffers.end())
        continue;
      logging::devicex::debug("remoteBufferWeightsFromHost: {}", initId);
      const auto remoteBufferInfo =
          tensor->tensorLocationInfo.getRemoteBufferInfo();
      char *data0          = static_cast<char *>(tensor->tensorData()->data());
      const auto data0Size = tensor->tensorData()->size();

      // Various values used throughout the function
      const auto elemSize  = tensor->info.getDataTypeInfo()->nbytes();
      const unsigned nelms = tensor->info.nelms();

      const unsigned instanceReplicas    = getReplicationFactor();
      const unsigned globalReplicas      = getGlobalReplicationFactor();
      const unsigned globalReplicaOffset = getGlobalReplicaOffset();

      const auto grouping =
          tensor->getVariableSettings().getReplicaGrouping(globalReplicas);

      const unsigned numGroups     = grouping.getNumGroups();
      const unsigned realGroupSize = grouping.getGroupSize();

      // Lambda expression that does the writing op automatically
      const auto copyToRemoteBuffer =
          [this, remoteBufferInfo](char *from, const unsigned replicaId) {
            pEngine->copyToRemoteBuffer(
                from,
                lowering().getExchangeBundle().getRemoteBufferName(
                    remoteBufferInfo.first),
                static_cast<int>(remoteBufferInfo.second),
                replicaId);
          };

      if (tensor->tensorLocationInfo.isSharded()) {
        // Replicated weight sharding, each replica holds parts of the weight
        const auto &cbr =
            executable_.getCollectiveBalancedHostRearrangement(initId);

        const auto cbrNelms        = cbr.getNumRearrangedTensorElems();
        const auto shardDomainSize = cbr.getReplicationFactor();

        // If sharded, iterate over each group; create a temporary buffer for
        // the cbr-padded value of the group; cbr-rearrange the group data into
        // into this buffer, then copy the relevant shard into the remote buffer
        // of each group member. If running with multiple instances, whenever we
        // hit a replica not on this instance, we skip the copy.
        //
        // We could alternatively iterate only over the local replicas, but we
        // would have to cbr-rearrange the data for that replica's group into
        // the temporary buffer every single time, or hold buffer(s) containing
        // the cbr-rearranged data of all groups at once. These are both
        // prohibitive due to the sizes of these buffers, therefore we choose to
        // simply iterate over a larger space and skip whenever a replica is not
        // on this instance.
        for (unsigned group = 0; group < numGroups; group++) {

          std::vector<char> tmp(cbrNelms * elemSize);

          // Address in input buffer from which this group fetches their
          // weights.
          const unsigned address = group * nelms * elemSize;

          // Rearrange weights into tmp buffer
          cbr.rearrangeForCollective(&data0[address],
                                     data0Size - address,
                                     &tmp[0],
                                     tmp.size() * sizeof(tmp[0]),
                                     elemSize);

          // For each group member, copy the relevant shard into the remote
          // buffer.
          for (unsigned groupMember = 0; groupMember < realGroupSize;
               groupMember++) {
            const unsigned globalReplicaId =
                grouping.getReplicaAt(group, groupMember);
            // Skip this group member if not on this instance.
            if (globalReplicaId < globalReplicaOffset ||
                globalReplicaId >= globalReplicaOffset + instanceReplicas) {
              continue;
            }

            const unsigned localReplicaId =
                globalReplicaId - globalReplicaOffset;

            // View the replicas in the group as their local group indices
            // 0..realGroupSize. On this space, the sharding replica grouping is
            // defined. Therefore, we convert from replica_id to group_member,
            // then using the sharding replica grouping, find the assignment of
            // group_member to sharding group.
            //
            // NOTE: tensor->tensorLocationInfo only holds whether the group is
            // sharded or not. However, the CBHR that has been calculated for it
            // does contain the sharding group size at least, but not the
            // stride. Therefore, we do not currently support any grouping other
            // than (size=cbr_size, stride=1).

            // We could more generally express the subsets of the group that we
            // shard over using a grouping, but we only support stride=1.
            // Therefore, the cbr_member is always this simple modulo:
            const unsigned shardDomainMember = groupMember % shardDomainSize;
            const unsigned addr =
                shardDomainMember * cbrNelms * elemSize / shardDomainSize;

            copyToRemoteBuffer(&tmp[addr], localReplicaId);
          }
        }
      } else {
        for (unsigned localReplicaId = 0; localReplicaId < instanceReplicas;
             localReplicaId++) {
          const auto globalReplicaId = localReplicaId + globalReplicaOffset;
          const auto group           = grouping.getGroupAt(globalReplicaId);

          const unsigned long index = group * nelms * elemSize;
          copyToRemoteBuffer(&data0[index], localReplicaId);
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

void Devicex::hostStreamToHost(const MutableVoidData &mv_data,
                               TensorId id,
                               DownsampleStream downsample) {
  POPART_TRACEPOINT();

  // The host end of the poplar::Stream,
  // we will try to copy from here
  const void *src;

  // size of the host end of the poplar stream.
  // It is a char vector, so this is in bytes.
  int64_t nbytes_src;

  Tensor *tensor        = executable_.getTensor(id);
  auto variableSettings = tensor->getVariableSettings();

  std::vector<char> tmp;

  if (variableSettings.getRetrievalMode() ==
          VariableRetrievalMode::AllReplicas &&
      downsample == DownsampleStream::GroupPrimary) {
    // Down-sample
    auto groupCount =
        variableSettings.getGroupCount(getGlobalReplicationFactor());
    auto downSampledSize = groupCount * tensor->info.nbytes();

    // Create temporary buffer
    tmp = std::vector<char>(downSampledSize);

    // Note, if ::AllReplicas, you may still not need a d2hWeightBuffer if
    // enablesVariableCaching is off and the replica group size is 1 (and thus
    // the num replicas returning a value is the same as the number of groups).
    char *addr_src_base = getD2hWeightData(tensor);

    // Copy the samples into the buffer
    for (auto group = 0; group < groupCount; group++) {

      auto replica  = variableSettings.getGroupRepresentative(group);
      auto addr_dst = tmp.data() + (group * tensor->info.nbytes());
      auto addr_src = addr_src_base + (replica * tensor->info.nbytes());
      memcpy(addr_dst, addr_src, tensor->info.nbytes());
    }
    src        = static_cast<const void *>(tmp.data());
    nbytes_src = downSampledSize;
  } else {
    // Do nothing special
    src        = static_cast<const void *>(getD2hWeightData(tensor));
    nbytes_src = getD2hWeightBufferSize(tensor);
  }

  auto dst = mv_data.data;

  if (src == dst) {
    // Should only happen when this function is called as part of weightsToHost
    // to copy from the d2hWeightBuffer to the TensorData, but these buffers are
    // the same as the tensor did not need an intermediary d2hWeightBuffer.
    POPART_ASSERT(!needsIntermediaryD2hWeightBuffer(tensor));
    return;
  }

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
  // check is only performed in the first call to run, under the assumption
  // that the user is unlikely to change the size of buffers between runs.
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

  if (ir().canTrain()) {
    popxlMarkHostWeightsOutOfSync();
  }

  ++nCallsToRun;
}

void Devicex::run(std::string programHandle,
                  IStepIO &stepio,
                  std::string debugName) {
  POPART_TRACEPOINT();

  if (!prepareHasBeenCalled()) {
    throw runtime_error("Devicex::prepare() must be called before"
                        " Devicex::run(const IStepIO &) is called.");
  }

  // Check that the input and output buffers have the correct number of
  // elements. As run(.) is called multiple times during a user's session, the
  // check is only performed in the first call to run, under the assumption
  // that the user is unlikely to change the size of buffers between runs.
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

  auto &indexMap = lowering().getProgramHandleIndexMap();
  auto it        = indexMap.find(programHandle);

  if (it == indexMap.end()) {
    throw error("[Devicex::run] Program {} not found.", programHandle);
  }

  run(it->second, debugName);

  ++nCallsToRun;
}

void Devicex::connectRandomSeedStream() {
  POPART_TRACEPOINT();

  // Host to device stream.
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

  // Device to host stream.
  for (uint16_t replicaId = 0; replicaId < getReplicationFactor();
       ++replicaId) {

    auto callback = [this, replicaId](void *ptr) {
      if (replicaId == 0) {
        getRandomSeedBuffer = *reinterpret_cast<uint64_t *>(ptr);
      }
    };

    pEngine->connectStreamToCallback("d2h_randomSeed", replicaId, callback);
  }
}

void Devicex::connectRngStateStream() {
  // popart::DeviceInfo object is used to calculate rng state tensor size
  // (instead of poplar::Graph) because poplar::Graph might not exist when
  // we are using deserialized executable. Note that poplar::Target in
  // DeviceInfo contains info about all replicas and poplar::Target in
  // poplar::Graph about one replica.
  const unsigned repFactor = getReplicationFactor();
  const size_t rngSize     = RngStateLowering::getCombinedRngStateTensorSize(
      *lowering().getDeviceInfo(), repFactor);

  for (uint16_t replicaId = 0; replicaId < repFactor; ++replicaId) {
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
    run(PopPrograms::ProgramIndex::RandomSeedFromHost, "SetRandomSeed");
  }
}

uint64_t Devicex::getRandomSeedToHost() {
  POPART_TRACEPOINT();
  if (ir().useSyntheticData() == false) {
    pEngine->disableExecutionProfiling();
    run(PopPrograms::ProgramIndex::RandomSeedToHost, "GetRandomSeed");
  }

  return getRandomSeedBuffer;
}

void Devicex::setRngStateFromHost() {
  if (ir().useSyntheticData() == false) {
    pEngine->disableExecutionProfiling();
    run(PopPrograms::ProgramIndex::RngStateFromHost, "SetRngState");
  }
}

std::vector<uint32_t> Devicex::getRngStateToHost() {
  std::vector<uint32_t> rngState;

  if (ir().useSyntheticData() == false) {
    // Reset the buffer
    logging::devicex::debug("Cleaning the rng buffer before receiving data");

    // popart::DeviceInfo object is used to calculate rng state tensor size
    // (instead of poplar::Graph) because poplar::Graph might not exist when
    // we are using deserialized executable. Note that poplar::Target in
    // DeviceInfo contains info about all replicas and poplar::Target in
    // poplar::Graph about one replica.
    const unsigned repFactor = getReplicationFactor();
    const size_t rngSize     = RngStateLowering::getCombinedRngStateTensorSize(
        *lowering().getDeviceInfo(), repFactor);

    for (auto &buffer : rngBuffer) {
      buffer.second = std::vector<uint32_t>(rngSize);
    }
    pEngine->disableExecutionProfiling();
    run(PopPrograms::ProgramIndex::RngStateToHost, "GetRngState");
    logging::devicex::debug("Copying data to host");
    for (uint16_t replicaId = 0; replicaId < repFactor; ++replicaId) {
      rngState.insert(rngState.end(),
                      rngBuffer[replicaId].begin(),
                      rngBuffer[replicaId].end());
    }
  }

  return rngState;
}

void Devicex::setRngStateValue(const std::vector<uint32_t> rngState) {
  // popart::DeviceInfo object is used to calculate rng state tensor size
  // (instead of poplar::Graph) because poplar::Graph might not exist when
  // we are using deserialized executable. Note that poplar::Target in
  // DeviceInfo contains info about all replicas and poplar::Target in
  // poplar::Graph about one replica.
  const unsigned repFactor = getReplicationFactor();
  const size_t rngSize     = RngStateLowering::getCombinedRngStateTensorSize(
      *lowering().getDeviceInfo(), repFactor);

  if (rngState.size() != rngSize * repFactor) {
    throw runtime_error("Devicex::setRngStateValue received rngState of size "
                        "{}; was expecting size {}",
                        rngState.size(),
                        rngSize * getReplicationFactor());
  }
  const uint32_t *seed_ptr = rngState.data();
  for (uint16_t replicaId = 0; replicaId < repFactor; ++replicaId) {
    rngBuffer[replicaId].assign(seed_ptr, seed_ptr + rngSize);
    seed_ptr += rngSize;
  }
}

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

namespace {

/**
 * \brief Callback connected to the HostToDevice host streams of on-chip
 * variables, to implement `weightsFromHost`.
 *
 * When you do `pEngine->connectStream(stream, data)`, either the stream is
 * BROADCAST, and you pass one datum that is then broadcast to each replica; or
 * the stream is REPLICATE and you must provide one datum for every replica.
 *
 * When a variable is replica-grouped, the TensorData has one datum per group.
 * We therefore either must duplicate those values into a new buffer that has
 * one datum per replica; or use
 * `pEngine->connectStreamToCallback(stream, replica, tdata[group(replica)])`
 * for every replica.
 *
 * The former requires an additional rf * datum_size memory, whereas the latter
 * is in-place. In order to support large models where these buffers are so big
 * that the host memory is exhausted, we therefore choose the latter approach.
 *
 * This callback implements this approach.
 */
class H2dWeightStreamCallback final : public poplar::StreamCallback {
public:
  using Result = poplar::StreamCallback::Result;

  H2dWeightStreamCallback(void *tensorDataGroup_, std::size_t groupNbytes_);

  // We know that of all the poplar::Program's we pass to the engine to compile,
  // only WeightsFromHost copies from this stream, and it will only do so once.
  // Therefore, in prefetch/fetch, we unconditionally copy our one datum (the
  // tensor data) every time; and complete() and prefetchInvalidate() do not
  // need to be overriden.

  Result prefetch(void *__restrict p) noexcept override;

  void fetch(void *__restrict p) noexcept override;

private:
  void *tensorDataGroup;
  std::size_t groupNbytes;

  void doCopy(void *__restrict p) const noexcept;
};

H2dWeightStreamCallback::H2dWeightStreamCallback(void *tensorDataGroup_,
                                                 std::size_t groupNbytes_)
    : tensorDataGroup(tensorDataGroup_), groupNbytes(groupNbytes_) {}

H2dWeightStreamCallback::Result
H2dWeightStreamCallback::prefetch(void *__restrict p) noexcept {
  doCopy(p);
  return Result::Success;
}

void H2dWeightStreamCallback::fetch(void *__restrict p) noexcept { doCopy(p); }

void H2dWeightStreamCallback::doCopy(void *__restrict p) const noexcept {
  std::memcpy(p, tensorDataGroup, groupNbytes);
}

} // namespace

bool Devicex::isEngineLoaded() const {
  return getDevicexInfoUnsafe()->isMostRecentlyLoaded(this);
}

void Devicex::setEngineIsLoaded(bool isLoaded) {
  if (isLoaded) {
    getDevicexInfoUnsafe()->setMostRecentlyLoaded(this);
  } else {
    getDevicexInfoUnsafe()->setMostRecentlyLoaded(nullptr);
  }
}

void Devicex::loadEngineAndConnectStreams() {
  POPART_TRACEPOINT();
  if (deviceInfo->getConnectionType() == DeviceConnectionType::Never) {
    throw runtime_error("Trying to load an engine on an offline device");
  }
  if (!pEngine) {
    throw runtime_error("Trying to load an engine but no compiled engine. Did "
                        "you forget to call prepareDevice()?");
  }

  DevicexInfo &di = *getDevicexInfoUnsafe();

  // Let the device info know that this devicex's engine
  // has most recently loaded its engine onto the poplar
  // device
  di.setMostRecentlyLoaded(this);

  if (di.getConnectionType() == DeviceConnectionType::OnDemand) {
    logging::devicex::debug("Attaching to device on demand");
    if (!di.tryAttachUntilTimeout()) {
      throw runtime_error("Failed to attach to device");
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
              tensor->getVariableSettings().getGroupCount(replicationFactor);

          if (groupCount == 1 || groupCount == replicationFactor) {
            //  The underlying data of the tensor has the correct number and
            //  order of data elements for the stream to be able to directly
            //  copy from it.
            pEngine->connectStream(lowering().h2dId(id),
                                   tensor->tensorData()->data());
          } else {
            // If we need to send different data to the replicas, then the
            // Poplar stream should have been set to REPLICATE so that we can do
            // this.
            POPART_ASSERT_EQ(tensor->getReplicatedStreamMode(),
                             ReplicatedStreamMode::Replicate);

            const auto globalReplicaOffset = getGlobalReplicaOffset();
            const auto rg = tensor->getVariableSettings().getReplicaGrouping(
                getGlobalReplicationFactor());

            for (unsigned r = 0; r < replicationFactor; r++) {
              const auto globalReplica = r + globalReplicaOffset;
              const auto group         = rg.getGroupAt(globalReplica);

              const auto groupDataAddr = tensor->info.nbytes() * group;
              auto groupData           = static_cast<void *>(
                  static_cast<char *>(tensor->tensorData()->data()) +
                  groupDataAddr);

              pEngine->connectStreamToCallback(
                  lowering().h2dId(id),
                  r,
                  std::make_unique<H2dWeightStreamCallback>(
                      groupData, tensor->info.nbytes()));
            }
          }
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
        if (tensor->getReplicatedStreamMode() ==
                ReplicatedStreamMode::Broadcast &&
            replicationIndex != 0)
          break;

        logging::devicex::debug(
            "Connecting input stream {}@{}", tensor->id, replicationIndex);

        IStepIO *downstreamIo = stepIoSplitter->getDownstreamStepIO(
            streamTensorId, tensor->info, replicationIndex);

        std::shared_ptr<InputDatastream> ds =
            std::make_shared<InputDatastream>(tensor, streamId);
        ds->setStepIO(downstreamIo);

        this->inputStreams[std::make_tuple(tensor->id, replicationIndex)] = ds;

        auto callback = std::make_unique<PrefetchCallback>(ds);

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
      auto returnAll =
          tensor->getVariableSettings().numReplicasReturningVariable(
              replicationFactor) == replicationFactor;

      for (auto g = 0; g < groups.size(); g++) {
        auto group = groups[g];

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

        initD2hWeightBuffer(tensor);
        auto *data0 = getD2hWeightData(tensor);

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

  setRandomSeedFromHost(); // Stream random seed value by default (prog empty
                           // if no randomness)
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
  const auto &sessionOptions = ir().getSessionOptions();

  const auto lifetimeTimer =
      ir().timePartitionLogger().scopedStopwatch("Preparing devicex");

  POPART_TRACEPOINT();
  if (!lowering().prepareGraphHasBeenCalled()) {
    lowering().prepareGraph();
  }

  logging::devicex::info(std::string("\nNext step is poplar Engine creation. "
                                     "Breakdown of compile time so far:\n") +
                         ir().timePartitionLoggerStr());

  if (sessionOptions.compileEngine) {
    const auto engineCreationTimer =
        ir().timePartitionLogger().scopedStopwatch("Engine creation");

    try {
      // Construct ProfileCacher in case cached executables is to be
      // profiled
      std::string cachedExecutablePathStr =
          executable_.getCachePath(sessionOptions.cachePath);
      auto profileCacher = ProfileCacher(sessionOptions,
                                         cachedExecutablePathStr,
                                         lowering().getPoplarGraphDebugName());

      // Obtain the executable
      auto executable = lowering().getExecutable(profileCacher);
      // Restore profiles from cache to the autoReport dir given a cache hit
      if (lowering().usingCachedExecutable()) {
        profileCacher.restoreProfilesFromCache();
      }
      pEngine.reset(
          new poplar::Engine(std::move(executable), lowering().engineOptions));

      if (!executable_.isDeserialized() && executable_.shouldSerialize()) {
        static constexpr bool serializePopartMetadata = true;
        const std::string cachePath = sessionOptions.cachePath;
        serializeExecutable(executable_.getCachePath(cachePath),
                            serializePopartMetadata,
                            sessionOptions.enableVariablesCaching);
      }

      logging::devicex::info(
          std::string("\npoplar Engine construction complete. Breakdown of "
                      "compile time:\n") +
          ir().timePartitionLoggerStr());

    } catch (const poplar::graph_memory_allocation_error &e) {
      // If the creation of the engine throw an exception due to memory
      // allocation i.e. the program does not fit show graph profile and
      // re-throw the exception In certain cases poplar will throw the error
      // without a graph profile. The following engine option needs to be set
      // to enable the graph profile in this case
      // "debug.allowOutOfMemory":"true"
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

std::string
Devicex::prepareFileToSerialization(const std::string &path,
                                    const std::string &defaultFilename) {
  auto target = boost::filesystem::path(path);
  if (target.has_parent_path()) {
    auto targetDir = target.parent_path();
    if (!boost::filesystem::exists(targetDir)) {
      logging::devicex::warn("Specified directory not found. "
                             "Creating {} directory ",
                             targetDir);
      if (!boost::filesystem::create_directories(targetDir)) {
        throw error("Cannot create cache directory. Aborting.");
      }
    }
  }

  // Check that the filename is not a directory
  std::string filePath = path;
  if (boost::filesystem::is_directory(target)) {
    filePath = logging::format("{}/{}", path, defaultFilename);
    logging::devicex::warn(
        "{} is a directory, saving serialized Executablex to {}",
        target.string(),
        filePath);
  } else {
    logging::devicex::info("Saving serialized Executablex to {}", filePath);
  }

  // Check that one can open the file
  std::ofstream out(filePath, std::ofstream::binary);
  if (!out.is_open()) {
    throw error("Unable to open file '{}'", filePath);
  }

  return filePath;
}

void Devicex::serializeExecutable(std::ostream &out,
                                  bool serializePopartMetadata,
                                  bool serializeTensorData) {
  POPART_TRACEPOINT();
  serialization::Writer writer(out, *this);
  writer.serializePoplarExecutable();
  if (serializePopartMetadata) {
    writer.serializePopartMetadata();
  }
  if (serializeTensorData) {
    if (!popxlAreHostWeightsInSync()) {
      popxlWeightsToTensorData();
    }
    writer.serializeTensorData();
  }
}

void Devicex::serializeExecutable(const std::string &path,
                                  bool serializePopartMetadata,
                                  bool serializeTensorData) {
  std::string filePath = prepareFileToSerialization(path, "executable.popef");
  std::ofstream out(filePath, std::ofstream::binary);
  serializeExecutable(out, serializePopartMetadata, serializeTensorData);
  out.flush();
  out.close();
}

void Devicex::serializeTensorData(const std::string &path) {
  std::string filePath = prepareFileToSerialization(path, "variables.popef");
  std::ofstream out(filePath, std::ofstream::binary);
  {
    serialization::Writer writer(out, *this);
    if (!popxlAreHostWeightsInSync()) {
      popxlWeightsToTensorData();
    }
    writer.serializeTensorData();
  }
  out.flush();
  out.close();
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

popx::DevicexInfo *Devicex::getDevicexInfoUnsafe() const {
  if (!deviceInfo) {
    throw internal_error("Devicex::deviceInfo unexpectedly not set.");
  }

  auto castedDeviceInfo = dynamic_cast<DevicexInfo *>(deviceInfo.get());
  if (castedDeviceInfo == nullptr) {
    throw internal_error(
        "Devicex::deviceInfo could not be cast to DevicexInfo.");
  }

  return castedDeviceInfo;
}

bool Devicex::distributedReplicatedGraphsEnabled() const {
  return ir().getSessionOptions().enableDistributedReplicatedGraphs;
}

bool Devicex::needsIntermediaryD2hWeightBuffer(const Tensor *t) const {
  const auto globalReplicas = getGlobalReplicationFactor();
  const auto retFactor =
      t->getVariableSettings().numReplicasReturningVariable(globalReplicas);

  const auto numGroups = t->getVariableSettings()
                             .getReplicaGrouping(globalReplicas)
                             .getNumGroups();

  // `devicex->prepare()` must result in a cache entry whose TensorData matches
  // the data in the Executablex. Furthermore, `weightsToHost` must not make the
  // tensor data out-of-sync with the data in the cache.
  //
  // As the Executablex is essentially a view into the Ir, and so the tensor
  // data comes from the TensorData in the Ir, we cannot update the TensorData
  // in weightsToHost, as it would no longer match the cache entry created
  // earlier. Therefore, we must use a separate d2hWeightBuffer for
  // weightsToHost.
  //
  // However, in the case where enableVariablesCaching is off, there is no
  // tensor data in the cache, so the above requirement is void. Therefore, we
  // can write directly to the Ir's TensorData in weightsToHost.
  const auto enableVariablesCaching =
      ir().getSessionOptions().enableVariablesCaching;

  // What is the case where t in executable.getWeightTensors() but not
  // TensorType::Variable? This condition was added to maintain the semantics of
  // the previous code.
  return enableVariablesCaching || t->tensorType() != TensorType::Variable ||
         retFactor != numGroups;
}

void Devicex::initD2hWeightBuffer(const Tensor *t) {
  const auto retFactor = t->getVariableSettings().numReplicasReturningVariable(
      getGlobalReplicationFactor());

  if (needsIntermediaryD2hWeightBuffer(t)) {
    d2hWeightBuffers[t->id] = std::vector<char>(t->info.nbytes() * retFactor);
  } else {
    logging::devicex::trace(
        "Reusing TensorData for d2hWeightBuffer of tensor {}", t->id);
  }
}

std::size_t Devicex::getD2hWeightBufferSize(const Tensor *t) const {
  // Whether using the TensorData, or an intermediary d2hWeightBuffer, this
  // gives the size of the buffer.
  return t->getVariableSettings().numReplicasReturningVariable(
             getGlobalReplicationFactor()) *
         t->info.nbytes();
}

char *Devicex::getD2hWeightData(Tensor *t) {
  return needsIntermediaryD2hWeightBuffer(t)
             ? d2hWeightBuffers[t->id].data()
             : static_cast<char *>(t->tensorData()->data());
}

} // namespace popx
} // namespace popart
