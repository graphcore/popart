// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPDEVICE_HPP
#define GUARD_NEURALNET_POPDEVICE_HPP

#include <popart/vendored/optional.hpp>

#include <poplar/DataStream.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/MatMul.hpp>
#include <poputil/TileMapping.hpp>

#include <snap/Graph.hpp>

#include <popart/aliaszerocopy.hpp>
#include <popart/devicemanager.hpp>
#include <popart/popx/creatorx.hpp>
#include <popart/popx/enigma.hpp>
#include <popart/popx/poplaroptionsx.hpp>
#include <popart/popx/popprograms.hpp>
#include <popart/popx/poptensors.hpp>
#include <popart/popx/pritask.hpp>
#include <popart/popx/virtualgraph.hpp>

#include <set>
#include <tuple>
#include <popart/names.hpp>
// MutableVoidData is defined in here:
#include <popart/stepio.hpp>

#include <popart/tensordata.hpp>

namespace popart {
class StepIOSplitter;
namespace popx {

using PopStreamId = std::string;

class IrLowering;
class Executablex;

poplar::Type popType(const TensorInfo &);
poplar::Type popType(DataType);

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
  void remoteBufferWeightsFromHost();
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

  // device -> host stream
  void weightsToHost();
  void remoteBufferWeightsToHost();
  // device ->host stream -> specified host addresses
  void weightsToHost(const std::map<TensorId, MutableVoidData> &);

  // TODO T8229 : change these names to disambiguate
  // the source and destination

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
  void serializeExecutable(std::ostream &out);
  void serializeExecutable(const std::string &path);

private:
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

    // Called to indicate the data has been comsumed
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
    // successful prefetch to happen accross a step (i.e. a call to
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
  std::map<TensorId, std::vector<char>> chBuffers;

  // map of buffers for streaming to IPU.
  std::map<TensorId, std::vector<char>> h2dWeightBuffers;

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

  // Is this Devicex's engine the last to have been loaded onto
  // deviceInfo's device?
  // Becomes true once loadEngineAndConnectStreams() is called.
  // Becomes 'false' if another engine has been loaded after
  // loadEngineAndConnectStreams() has been called. This is
  // different to 'prepareHasBeenCalled_', which, once true,
  // is always true
  bool engineIsLoaded = false;

  // Wrapper function that checks the calling devicex was the
  // last to have loaded its engine to deviceInfo's device
  void run(PopPrograms::ProgramIndex ind, std::string debugName);

  void hostStreamToHost(const MutableVoidData &mv_data, TensorId id);

  // Call hostToHostStream on all the Tensors in pir->dataStreamTensors()
  void anchorsHostToHostStreams(IStepIO &stepio);

  // Call hostStreamToHost in all the Tensors in pir->dataFlow.anchors()
  void anchorsHostFromHostStreams(IStepIO &stepio);

  void doProfileChecks() const;

  /**
   * This function iterates over the h2dWeightBuffers map,
   * for each TensorId in the map it transfers the data of the corresponding
   * tensors tensor->tensorData()->data() to a fresh h2dWeightBuffer so that
   * data can be transferred to device
   * using the replicate mode.
   *
   * For a tensor to be added to the h2dWeightBuffers map it has to have
   * VariableSettings such that the number of replicated groups is not
   * broadcastable or replicatable as per ReplicatedStreamMode.
   */
  void initializeH2dWeightBuffers();

  /**
   * Clears h2dWeightBuffers, effectively erasing all the
   * redundant data storage on host.
   */
  void deinitializeH2dWeightBuffers();
};

} // namespace popx
} // namespace popart

#endif
