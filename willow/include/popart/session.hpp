// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_NET_HPP
#define GUARD_NEURALNET_NET_HPP

#include <memory>
#include <vector>

#include <poplar/DataStream.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/stepio.hpp>

namespace popart {

class DeviceInfo;

namespace popx {
class Devicex;
class IrLowering;
class Executablex;
} // namespace popx

/**
 * Session is a runtime instance that provides an interface for executing ONNX
 * graphs on IPU hardware.
 */
class Session {
private:
  void ctorCommonLogic();

protected:
  Session();
  Session(Ir ir, std::shared_ptr<DeviceInfo> deviceInfo);

  void configureFromOnnx(const std::string &modelProtoOrFilename,
                         const DataFlow &df,
                         const TensorId &lossIn,
                         const Optimizer *optimizerIn,
                         const InputShapeInfo &perk,
                         std::shared_ptr<DeviceInfo> deviceInfo,
                         const SessionOptions &userOptions,
                         const Patterns &patterns);

public:
  virtual ~Session() = 0;

  std::vector<uint32_t> getRNGState();
  void setRNGState(const std::vector<uint32_t>);

  /**
   * Sets the random number generator seed on all tiles of the device. This
   * ensures deterministic behaviour of random operations in the graph.
   *
   * \param The seed value.
   */
  void setRandomSeed(uint64_t seedValue);

  /**
   * Compiles the graph and exports it to the specified path.
   *
   * This will create a \c poplar::Graph and compile the \c poplar::Executable
   * before exporting the executable and metadata.

   * \param filename Name of the file where the compiled executable and
   *                 associated metadata will be saved.
   */
  void compileAndExport(std::string filename);

  /**
   * Compiles the graph and exports it to the specified stream.
   *
   * This will create a \c poplar::Graph and compile the \c poplar::Executable
   * before exporting the executable and metadata.

   * \param out Stream where the compiled executable and
   *            associated metadata will be written to.
   */
  void compileAndExport(std::ostream &out);

  /**
   * Load the \c poplar::Executable and the PopART metadata from the given
   * file. The file must have been created with compileAndExport()
   *
   * \param filename Name of the file to load the executable from.
   */
  void loadExecutableFromFile(std::string filename);

  /**
   * Load the \c poplar::Executable and the PopART metadata from the given
   * stream. The stream must have been created with compileAndExport()
   *
   * \param in Stream to load the executable from.
   */
  void loadExecutableFromStream(std::istream &in);

  /**
   * Prepare the network for execution.
   *
   * This will create the \c poplar::Graph and \c poplar::Engine.
   *
   * \param loadEngine Load the engine and connect the streams once
   *                   the device is ready.
   */
  void prepareDevice(bool loadEngine = true);

  /**
   * Load the engine on the device and connect the streams
   *
   * This will set up the \c poplar::Streams.
   *
   * Note: This call is optional. The engine will implicitly be loaded
   * on the device when required.
   */
  void loadEngineAndConnectStreams();

  /**
   * Write weights from host to the device.
   */
  void weightsFromHost();

  /**
   * Copy the weights to host from the device.
   */
  void weightsToHost();

  /**
   * Copy the cycle count tensor to host from the device.
   */
  uint64_t getCycleCount(std::string id = "");

  /**
   * Perform one step.
   *
   * Read input data from address in \c stepIO.in.
   * Write the output data to addresses in \c stepIO.out.
   *
   * \param stepIO Input and output data.
   * \param debugName Debug string to identify this run in logs.
   */
  void run(IStepIO &stepIO, std::string debugName = "");

  /**
   * Update the tensor locations of the tensors in the Session's ONNX model.
   * The new file will be created at this point, and written to when the ONNX
   * model is saved with a subsequent call to modelToHost.
   * \param fromLocation All externally saved tensors with location fromLocation
   *                     will have their location updated to toLocation.
   * \param toLocation The updated location. Must not already exist.
   */
  void updateExternallySavedTensorLocations(const std::string &fromLocation,
                                            const std::string &toLocation);

  /**
   * Write current model to ONNX file.
   *
   * \param fn Path to file. Can be absolute or relative. If you plan to run
   *           your program in multiple processes simultaneously, you should
   *           avoid possible race conditions by writing to different files, for
   *           example by using temporary files.
   */
  void modelToHost(const std::string &fn);

  /**
   * Get the TensorInfo on a Tensor.
   */
  TensorInfo getInfo(TensorId) const;

  /**
   * Returns whether or not a the tensor for the specified ID has info.
   *
   * If the return value is false, you will be unable to obtain an instance of
   * TensorInfo using getInfo.
   */
  bool hasInfo(TensorId) const;

  /**
   * Retrieve the summary from from the \c poplar::Engine.
   *
   * The options which were given to the constructor will influence the
   * information in the report.
   *
   * This may only be called after the prepareDevice() call has been made.
   * \param resetProfile Resets the execution profile.
   * \return A string containing the report.
   */
  std::string getSummaryReport(bool resetProfile = true) const;

  /**
   * Retrieve the graph report from the \c poplar::Engine.
   *
   * The options which were given to the constructor will influence the
   * information in the report.  By default a JSON format report is produced.
   *
   * This may only be called after the prepareDevice() call has been made.
   *
   * \param useCbor Produce a CBOR formatted report.
   * \return A string containing the graph (compilation) report.
   */
  std::string getGraphReport(bool useCbor = false) const;

  /**
   * Retrieve the execution report from the \c poplar::Engine.
   *
   * The options which were given to the constructor will influence the
   * information in the report.  By default a JSON format report is produced.
   *
   * This may only be called after the prepareDevice() call has been made.
   *
   * \param useCbor Produce a CBOR formatted report.
   * \param resetProfile Resets the execution profile.
   * \return A string containing the execution report.
   */
  std::string getExecutionReport(bool useCbor      = false,
                                 bool resetProfile = true) const;

  /**
   * Retrieve the serialized graph from the \c poplar::Engine.
   *
   * A JSON format report is produced.
   *
   * This may only be called after the prepareDevice() call has been made.
   *
   * \return A string containing the serialized graph.
   */
  std::string getSerializedGraph() const;

  /**
   * Reset the weights with the weights in an ONNX model that differs from the
   * current model only in weights. This only updates the weights on the host;
   * the user still needs to call weightsFromHost() after this to update the
   * weights on the device.
   *
   * \param model Either an ONNX model protobuf, or the name of a file
   *              containing an ONNX model protobuf.
   * \param ignoreWeightsInModelWithoutCorrespondingHostWeight If true, do
   *        not error if there are initializers in the ONNX model with no
   *        corresponding initializer tensor in the session's IR.
   */
  void resetHostWeights(
      const std::string &model,
      const bool ignoreWeightsInModelWithoutCorrespondingHostWeight = false);

  /**
   * Read the weights. Must have called weightsToHost() first.
   *
   * The weight data is written to the addresses in \c weightsIo.out.
   */
  void readWeights(const IWeightsIO &weightsIo);

  /**
   * Write the weights. Must call weightsFromHost() after this.
   *
   * The weight data is written to the addresses in \c weightsIo.out.
   */
  void writeWeights(const IWeightsIO &weightsIo);

  /**
   * Serizalise the IR graph to a string.
   *
   * \param format The format to use for serializing.
   */
  std::string serializeIr(IrSerializationFormat format);

  const Ir &getIr() const { return ir; }
  const popx::Devicex &getDevice() const { return *device_; }
  popx::Devicex &getDevice() { return *device_; }
  const popx::IrLowering &getIrLowering() const { return *lowering_; }
  const popx::Executablex &getExecutable() const { return *executable_; }

protected:
  /**
   * Select a device type.
   *
   * \param deviceInfo Defines the type of device to work on.
   */
  void setDevice(std::shared_ptr<DeviceInfo> deviceInfo);

  /**
   * Attempts to load a serialized executable. If successful then IR
   * preparation and \c poplar::Graph compilation are skipped.
   */
  bool tryLoadExecutable();

  /**
   * Throws an error if there is no executable.
   */
  void assertExecutableLoaded() const;

  /**
   * Initializes the progress logger to zero
   */
  void initProgressLogger(const SessionOptions &userOptions);

  /**
   * Abstraction of the computation. The Ir is where
   * all the compute graph optimisations, backwards pass construction,
   * re-computation growing etc. happens.
   */
  Ir ir;

  /**
   * Implementation of the computation. For the IPU back-end this is
   * where calls to Poplar are made.
   */
  std::unique_ptr<popx::Devicex> device_;

  /**
   * Implementation of the lowering of the PopART Ir to the
   * Poplar Graph.
   */
  std::unique_ptr<popx::IrLowering> lowering_;

  /**
   * The final executable which contains all the data, metadata
   * and configuration parameters necessary to start running
   * the program on the device.
   */
  std::unique_ptr<popx::Executablex> executable_;

  /**
   * Information about the device which this session uses.
   */
  std::shared_ptr<DeviceInfo> deviceInfo_;

  /**
   * Flag to indicate if weightsFromHost() has been called
   */
  bool weightsFromHostCalled = false;

  /**
   * Flag to indicate if run() has been called.
   */
  bool runCalled = false;

  /**
   * Map of hashes / filenames of cached executables.
   */
  HashesMap cacheEntries;
};

class InferenceSession : public Session {

  using Session::Session;

public:
  ~InferenceSession() override;

  static std::unique_ptr<InferenceSession>
  createFromIr(Ir ir, std::shared_ptr<DeviceInfo> deviceInfo);

  /** Create a runtime class for executing an ONNX graph on a set of IPU
   *  hardware for inference.
   *
   * \param model Either an ONNX model protobuf, or the name of a file
   *              containing an ONNX model protobuf.
   * \param inputShapeInfo Information about the shapes of input and output
   *                       tensors.
   * \param dataFlow Configuration for the data feeds and fetches.
   * \param userOptions String to configure session options.
   * \param patterns Optimization patterns to apply.
   */

  static std::unique_ptr<InferenceSession>
  createFromOnnxModel(const std::string &model,
                      const DataFlow &dataFlow,
                      std::shared_ptr<DeviceInfo> deviceInfo,
                      const InputShapeInfo &inputShapeInfo = InputShapeInfo(),
                      const SessionOptions &userOptions    = SessionOptions(),
                      const Patterns &patterns             = Patterns());
};

class TrainingSession : public Session {

  using Session::Session;

public:
  ~TrainingSession() override;

  static std::unique_ptr<TrainingSession>
  createFromIr(Ir ir, std::shared_ptr<DeviceInfo> deviceInfo);

  /** Create a runtime class for executing an ONNX graph on a set of IPU
   *  hardware for training.
   *
   * \param model Either an ONNX model protobuf, or the name of a file
   *              containing an ONNX model protobuf.
   * \param inputShapeInfo Information about the shapes of input and output
   *                       tensors.
   * \param dataFlow Configuration for the data feeds and fetches.
   * \param loss The TensorId of the final scalar loss tensor for training.
   * \param optimizer The name of an optimizer to use when training.
   * \param userOptions String to configure session options.
   * \param patterns Optimization patterns to apply.
   */

  static std::unique_ptr<TrainingSession>
  createFromOnnxModel(const std::string &model,
                      const DataFlow &dataFlow,
                      const TensorId &loss,
                      const Optimizer &optimizer,
                      std::shared_ptr<DeviceInfo> deviceInfo,
                      const InputShapeInfo &inputShapeInfo = InputShapeInfo(),
                      const SessionOptions &userOptions    = SessionOptions(),
                      const Patterns &patterns             = Patterns());

  /**
   * Update the optimizer and the associated hyperparameters but not the
   * optimizer state tensors.
   *
   * **NOTE**: The optimizer parameter has to be compatible with the optimizer
   * passed to the constructor. For example, you cannot call this function
   * with an SDG1 optimizer if you created the session with an SDG0 optimizer.
   * The reason for this is that it is not possible to change the IR after
   * it has been constructed.
   *
   * \param optimizer A pointer to a popart::Optimizer.
   */
  void updateOptimizerFromHost(const Optimizer *optimizer);

  /**
   * Access the stream IDs for variables that are involved in host side
   * reductions on the host. Only populated if \c hostAllReduce is enabled in
   * the SessionOptions
   */
  const std::vector<std::string> &getHostReduceStreamIds() const;

  /**
   * Access the remote buffers associated with gradient and weight streams
   * that are used in host side all reduce operations. Only populated if
   * \c hostAllReduce and \c hostAllReduceRemoteBuffer are enabled.
   */
  const std::map<std::string, poplar::RemoteBuffer> &
  getHostReduceRemoteBuffers() const;

  /**
   * Connect Poplar stream callbacks. In conjunction with
   * `getGradAndVarStreamIds` the streams can be used to copy gradients to the
   * host to perform collective operations after which the variables can be
   * streamed back after they have been updated to the device.
   * \p index referes to the replica index when using replicated graphs.
   */
  void connectStreamToCallback(const std::string &streamHandle,
                               std::function<void(void *)> callback,
                               unsigned index = 0);

  /**
   * Read from a RemoteBuffer object into a user space pointer \p w.
   * This can be useful when we run larger models with host side
   * reductions since HEXOPT is currently limited to 128 MB.
   */
  void copyFromRemoteBuffer(const std::string &buffer,
                            void *w,
                            int repeat_index,
                            unsigned replication_index = 0);

  /**
   * Write to a RemoteBuffer object from a user space pointer \p w.
   * This can be useful when we run larger models with host side
   * reductions since HEXOPT is currently limited to 128 MB.
   */
  void copyToRemoteBuffer(void *w,
                          const std::string &buffer,
                          int repeat_index,
                          unsigned replication_index = 0);
};

} // namespace popart

#endif
