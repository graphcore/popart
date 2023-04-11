// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_DEVICEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_DEVICEX_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <poplar/Engine.hpp> // IWYU pragma: keep
#include <poplar/StreamCallback.hpp>
#include <poplar/Type.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/MatMul.hpp>
#include <popart/popx/popefserializer.hpp> // IWYU pragma: keep

#include "popart/datatype.hpp"
#include "popart/tensordebuginfo.hpp"

namespace pva {
class Report;
} // namespace pva

namespace popart {
class StepIOSplitter;
class DeviceInfo;
class IStepIO;
class IWeightsIO;
class Ir;
class MutableVoidData;
class Tensor;
class TensorInfo;

namespace popx {
namespace serialization {
class WriterImpl;
} // namespace serialization

using PopStreamId = std::string;

class IrLowering;
class Executablex;
class DevicexInfo;

poplar::Type popType(const TensorInfo &);
poplar::Type popType(DataType);

/**
 * DownsampleStream is a flag used to tell certain functions
 * that they should only work on the Primary Sample of a Group.
 */
enum class DownsampleStream {
  // Do no downsampling
  No = 0,

  // Sample only the primary member of a group
  GroupPrimary
};

class Devicex {
private:
  Executablex &executable_;
  int nCallsToRun{0};
  std::shared_ptr<DeviceInfo> deviceInfo;
  bool prepareHasBeenCalled_;

public:
  const Ir &ir() const;
  const IrLowering &lowering() const;
  IrLowering &lowering();

  Devicex(Executablex &exe, std::shared_ptr<DeviceInfo> deviceInfo);
  ~Devicex();

  // Compiles the graph and then prepares the streams for running on the device
  void prepare();

  void weightsFromHost();
  void buffersFromHost();
  void remoteBufferWeightsFromHost(const bool isUpdate = false);
  void optimizerFromHost();
  // Streams the random seed value from host, and sets the rng registers on
  // the device
  void setRandomSeedFromHost();
  // Streams the random seed value to host.
  uint64_t getRandomSeedToHost();

  // Stream RNG state host -> device
  void setRngStateFromHost();
  // Stream RNG state device -> host
  std::vector<uint32_t> getRngStateToHost();
  // write RNG State to host buffer (host -> host)
  void setRngStateValue(const std::vector<uint32_t>);

  std::map<std::string, std::vector<uint64_t>> cycleCountTensorToHost();
  void run(IStepIO &, std::string debugName = "");
  void run(std::string programHandle, IStepIO &, std::string debugName = "");

  // device -> host stream
  void weightsToHost();
  void remoteBufferWeightsToHost();
  // device -> host stream -> specified host addresses
  // (weightsToHost() + d2hWeightBuffersToTensorData)
  void weightsToHost(const std::map<TensorId, MutableVoidData> &);

private:
  // host stream -> specified host addresses
  void d2hWeightBuffersToTensorData(
      const std::map<TensorId, MutableVoidData> &onnxModelData);

  // host stream -> tensors
  void d2hWeightBuffersToTensors(const std::vector<Tensor *> &tensors);

  // Create file needed by serialization process. If the path points
  // existing directory, the function will create file "<path>/defaultFilename".
  std::string prepareFileToSerialization(const std::string &path,
                                         const std::string &defaultFilename);

public:
  /// Copy data from the device, to the host buffers, to the
  /// `tensor.tensorData()` buffers. Will not run a WeightsToHost program if
  /// weights already in sync with ipu. After WeightsToHost, marks the
  /// weights as in sync with the ipu.
  void popxlWeightsToTensorData();
  /// Mark the d2hWeightBuffers as out of sync with the ipu.
  void popxlMarkHostWeightsOutOfSync();
  /// Mark the d2hWeightBuffers as in sync with the ipu.
  void popxlMarkHostWeightsInSync();
  /// Are all the weights in sync with the ipu?
  bool popxlAreHostWeightsInSync();

  // Write weights from (CPU end of) stream, to dst (host -> host)
  void readWeights(const IWeightsIO &dst);

  // Write weights from src to Ir Tensor memory (host -> host)
  void writeWeights(const IWeightsIO &src);

  std::string getSummaryReport(bool resetProfile = true) const;
  std::string getSerializedGraph() const;

  pva::Report getReport() const;

  bool isEngineLoaded() const;
  void setEngineIsLoaded(bool isLoaded);

  void connectRandomSeedStream();

  void connectRngStateStream();

  // Connect a callback to the given poplar stream handle.
  void connectStreamToCallback(const std::string &streamHandle,
                               std::function<void(void *)> callback,
                               unsigned index);

  // Connect the poplar stream handle to the default poplar callback with the
  // given pointer.
  void connectStream(const std::string &streamHandle, void *host_buffer);

  // Connect a callback to the given poplar host function handle.
  void connectHostFunction(
      const std::string &functionHandle,
      std::function<void(const void *const *, size_t, void *const *, size_t)>
          callback,
      unsigned index);

  void copyFromRemoteBuffer(const PopStreamId buffer,
                            void *w,
                            int repeat_index,
                            unsigned replication_index = 0);

  void copyToRemoteBuffer(void *w,
                          const PopStreamId buffer,
                          int repeat_index,
                          unsigned replication_index = 0);

  // Although these belong in IrLowering we keep these in devicex since
  // they may be used in custom ops
  poplin::PlanningCache convCache;
  poplin::matmul::PlanningCache matmulCache;
  // These are always expected to be true. They have only been exposed for
  // testing.
  bool prePlanConvolutions = true;
  bool prePlanMatMuls      = true;

  // Helper method to get the replication factor based on the user options
  unsigned getReplicationFactor() const;
  unsigned getAccumulationFactor() const;

  // If globalReplicatedGraphs are enabled then this will return an
  // offset into the global instances, otherwise 0.
  unsigned getGlobalReplicaOffset() const;

  unsigned getGlobalReplicationFactor() const;

  bool isReplicatedGraph() const;

  const DeviceInfo *getDeviceInfo() const { return deviceInfo.get(); }
  DeviceInfo *getDeviceInfo() { return deviceInfo.get(); }
  void setDeviceInfo(std::shared_ptr<DeviceInfo> deviceInfo_) {
    deviceInfo = std::move(deviceInfo_);
  }

  // Return stored input tensors based on how they are allocated
  std::set<TensorId> getLinearlyCreatedInputTensors() const;
  std::set<TensorId> getEfficientlyCreatedInputTensors() const;

  bool prepareHasBeenCalled() const { return prepareHasBeenCalled_; }

  // Wrapper for calls to poplar Engine API calls: loading
  // engine onto the poplar device and connecting streams.
  // Must be called before running a poplar program with a
  // call to this Devicex's engine.
  void loadEngineAndConnectStreams();

  // Serialize the Poplar executable stored inside the device's engine.
  void serializeExecutable(std::ostream &out,
                           bool serializePopartMetadata,
                           bool serializeTensorData);
  void serializeExecutable(const std::string &path,
                           bool serializePopartMetadata,
                           bool serializeTensorData);
  // Serialize the tensor data stored inside the executablex's tensors.
  void serializeTensorData(const std::string &path);

private:
  friend class serialization::WriterImpl;

  std::unique_ptr<poplar::Engine> pEngine{nullptr};

  // We have datastreams which are created during the prepare phase and
  // associated with the stream call back
  // Then when run is called the data streams are associated with the
  // step oi class

  class Datastream {

  protected:
    Tensor *tensor;
    PopStreamId streamId;

    // This is per data stream to allow for different stepio
    // configurations per data stream.
    // Q : Is there a better type than a pointer?
    IStepIO *io;

  public:
    Datastream(Tensor *ten, PopStreamId s);

    void setStepIO(IStepIO *v) { io = v; }

    TensorId getTensorId();
  };

  // host to device data stream
  class InputDatastream : public Datastream {
  public:
    InputDatastream(Tensor *t, PopStreamId s);

    // Called to read data from an input stream
    void read(void *ptr);

    // Called to prefetch data from an input stream
    // return true is there is data prefetch else false
    bool readPrefetch(void *ptr);

    // Called to indicate the data has been consumed
    // by poplar
    void readComplete();
  };

  class PrefetchCallback : public poplar::StreamCallback {
  public:
    PrefetchCallback(std::shared_ptr<InputDatastream> ds_);

    poplar::StreamCallback::Result prefetch(void *dest) override;
    void fetch(void *dest) override;
    void complete() override;

    // NOTE: We do not need to override invalidatePrefetched because
    // our current StepIOSplitter implementation will never allow a
    // successful prefetch to happen across a step (i.e. a call to
    // Session::run call) and therefore there is never any
    // prefetches to invalidate.

  private:
    std::shared_ptr<InputDatastream> ds;
  };

  // device to host data stream
  class OutputDatastream : public Datastream {
  public:
    OutputDatastream(Tensor *t, PopStreamId s);
    void write(void *ptr);
  };

  // Splits one IStepIO into one for each replica.
  std::unique_ptr<StepIOSplitter> stepIoSplitter;

  // Map from TensorId,replicationIndex to the data streams
  using StreamId = std::tuple<TensorId, unsigned>;
  std::map<StreamId, std::shared_ptr<InputDatastream>> inputStreams;
  std::map<StreamId, std::shared_ptr<OutputDatastream>> outputStreams;

  // Q: Consider replacing the d2h weight buffer with a data stream as
  // done for inputs
  std::map<TensorId, std::vector<char>> d2hWeightBuffers;

  // Does this weight need an intermediary d2hWeightBuffer, or can the
  // TensorData be re-used?
  // Precondition: Tensor is in `executable_.getWeightTensors()`
  bool needsIntermediaryD2hWeightBuffer(const Tensor *) const;

  // Initialise the d2hWeightBuffer for the tensor. This will either re-use the
  // TensorData of the tensor (so do nothing) or create an intermediary
  // d2hWeightBuffer and add it to the map `d2hWeightBuffers`.
  // Precondition: Tensor is in `executable_.getWeightTensors()`
  void initD2hWeightBuffer(const Tensor *);

  // Get the size in bytes of the d2hWeightBuffer for this tensor.
  // Precondition: Tensor is in `executable_.getWeightTensors()`
  std::size_t getD2hWeightBufferSize(const Tensor *) const;

  // Get a pointer to the raw data of the d2hWeightBuffer for this tensor.
  // Precondition: Tensor is in `executable_.getWeightTensors()`
  char *getD2hWeightData(Tensor *);

  // Buffers for storing the hardware cycle count
  std::map<std::string, std::vector<uint64_t>> cycleCount;

  // Stream buffer for storing RNG states for replicas (HwSeeds)
  std::map<uint16_t, std::vector<uint32_t>> rngBuffer;

  // Stream buffer for storing result of getRandomSeed for one replica
  uint64_t getRandomSeedBuffer;

  // We may have prefetched data ready to be fed into the model, but we have
  // provided a new buffer which we want to be fetched. We invalidate the
  // prefetch by reconnecting the datastreams before each program run.
  void reconnectInputStreams();

  // Wrapper function that checks the calling devicex was the
  // last to have loaded its engine to deviceInfo's device
  void run(unsigned ind, std::string debugName);

  /** Copy from the host end of a d2h stream, to some final host memory.
   * This is the step which follows a copy from device to host.
   * poplar::Streams cannot write to an arbitrary dynamic address,
   * they are connected to a fixed host address. This function copies
   * from that fixed address to a dynamic address (mv_data).
   *
   * \param mv_data    Destination mutable data
   * \param id         Tensor to move into mutable data
   * \param downsample a flag (No/GroupPrimary), sending the entire stream if
   *                   set to no, and sampling only the group primaries (based
   *                   on the VariableSettings) of the tensor in question.
   *
   */
  void hostStreamToHost(const MutableVoidData &mv_data,
                        TensorId id,
                        DownsampleStream downsample);

  // Call hostToHostStream on all the Tensors in pir->dataStreamTensors()
  void anchorsHostToHostStreams(IStepIO &stepio);

  // Call hostStreamToHost in all the Tensors in pir->dataFlow.anchors()
  void anchorsHostFromHostStreams(IStepIO &stepio);

  void doProfileChecks() const;

  // Helper function that casts deviceInfo to a DevicexInfo* and throws nice
  // errors if this is not possible. This WILL throw an error if deviceInfo is
  // not set or if it's not pointing to a DevicexInfo object.
  popx::DevicexInfo *getDevicexInfoUnsafe() const;

  bool distributedReplicatedGraphsEnabled() const;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_DEVICEX_HPP_
