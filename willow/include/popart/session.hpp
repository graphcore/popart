// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_NET_HPP
#define GUARD_NEURALNET_NET_HPP

#include <functional>
#include <memory>
#include <vector>

#include <pva/pva.hpp>
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

const std::string DefaultInferenceSessionName = "inference";
const std::string DefaultTrainingSessionName  = "training";

/**
 * Session is a runtime instance that provides an interface for executing ONNX
 * graphs on IPU hardware.
 */
class Session {
private:
  void ctorCommonLogic();

protected:
  Session(std::string name = "");
  Session(std::shared_ptr<Ir> ir,
          std::shared_ptr<DeviceInfo> deviceInfo,
          std::string name = "");

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
   * Sets the random number generator seed that explicitly seeds all random
   * operations and, as a side-effect, derive a new RNG state from the seed and
   * sets it on the device. This RNG state is used to resolve stochastic
   * rounding. Note that to deterministically store and restore the combined
   * random state for a session, do the following:
   *
   * C++:
   * ```
   * // Store random state (session s0).
   * auto seed = s0.getRandomSeed();
   * auto rngState = s0.getRNGState();
   *
   * // Restore random state (session s1).
   * s1.setRandomSeed(seed);   // <-- affects RNG state, order important
   * s1.setRNGState(rngState);
   * ```
   *
   * Python:
   * ```
   * # Store random state (session s0).
   * seed = s0.getRandomSeed()
   * rngState = s0.getRNGState()
   *
   * # Restore random state (session s1).
   * s1.setRandomSeed(seed)   // <-- affects RNG state, order important
   * s1.setRNGState(rngState)
   * ```
   *
   * \param The seed value.
   */
  void setRandomSeed(uint64_t seedValue);

  /**
   * Get the value of the random number seed. By later calling `setRandomSeed`
   * with this value you can reinstate the random state logic that seeds random
   * operations.
   *
   * \returns The seed value.
   */
  uint64_t getRandomSeed();

  /**
   * Compiles the graph and exports it to the specified path.
   *
   * This will create a \c snap::Graph and compile the \c poplar::Executable
   * before exporting the executable and metadata.

   * \param filename Name of the file where the compiled executable and
   *                 associated metadata will be saved.
   */
  void compileAndExport(const std::string &filename);

  /**
   * Compiles the graph and exports it to the specified stream.
   *
   * This will create a \c snap::Graph and compile the \c poplar::Executable
   * before exporting the executable and metadata.
   *
   * \param out Stream where the compiled executable and
   *            associated metadata will be written to.
   */
  void compileAndExport(std::ostream &out);

  /**
   * Save a compiled graph to the specified path.
   *
   * \pre prepareDevice() must have been called.
   *
   * \param filename Name of the file where the compiled executable and
   *                 associated metadata will be saved.
   *
   * This method automatically creates folders as needed
   * if filename is located in a folder which doesn't exist.
   */
  void saveExecutableToFile(const std::string &filename);

  /**
   * Save a compiled graph to the specified stream.
   *
   * \pre prepareDevice() must have been called.
   *
   * \param out Stream where the compiled executable and
   *            associated metadata will be written to.
   */
  void saveExecutableToStream(std::ostream &out);

  /**
   * Create an \c aliasModel for each graph and run the poprithms ambiguity
   * checker on it. This throws an error if the graph has an inplacing ambiguity
   * and will prompt the user to check the inplacing.
   *
   * See \c poprithms::memory::inplace::Graph::AmbiguityStatus for more info on
   * what constitutes an ambiguity.
   */
  void checkInplacingAmbiguity() const;

  /**
   * Load the \c poplar::Executable and the PopART metadata from the given
   * file. The file must have been created with compileAndExport()
   *
   * \param filename Name of the file to load the executable from.
   */
  void loadExecutableFromFile(const std::string &filename);

  /**
   * Load the \c poplar::Executable and the PopART metadata from the given
   * stream. The stream must have been created with compileAndExport()
   *
   * \param in Shared pointer to the std stream to load the executable from.
   */
  void loadExecutableFromStream(std::shared_ptr<std::istream> in);

  /**
   * Prepare the network for execution.
   *
   * This will create the \c snap::Graph and \c poplar::Engine.
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
   * Connect a given poplar stream handle with a buffer to copy the memory to or
   * from IPU.
   */
  void connectStream(const std::string &streamHandle, void *buffer);

  /**
   * Connect Poplar host function callbacks.
   * \p index referes to the replica index when using replicated graphs.
   */
  void connectHostFunction(
      const std::string &functionHandle,
      std::function<void(const void *const *, size_t, void *const *, size_t)>
          callback,
      unsigned index = 0);

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
   * Run one step of a custom program.
   *
   * Read input data from address in \p stepIO.in.
   *
   * Write the output data to addresses in \p stepIO.out.
   *
   * \param programHandle The handle of the custom program to run.
   * \param stepIO        The input and output data.
   * \param debugName     A debug string to identify this run in logs.
   */
  void
  run(std::string programHandle, IStepIO &stepIO, std::string debugName = "");

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
   * Retrieve the graph report from the \c poplar::Engine.
   *
   * The options which were given to the constructor will influence the
   * information in the report.
   *
   * This may only be called after the prepareDevice() call has been made.
   *
   * \return The pva report object
   */
  pva::Report getReport() const;

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

  const Ir &getIr() const { return *ir; }
  const popx::Devicex &getDevice() const { return *device_; }
  popx::Devicex &getDevice() { return *device_; }
  const popx::IrLowering &getIrLowering() const { return *lowering_; }
  const popx::Executablex &getExecutable() const { return *executable_; }

  /**
   * Update cacheEntries from engine cache directory
   * and update ir::hashMatched_ with the updated cacheEntries
   *
   */
  void updateEngineCache();

protected:
  /**
   * Select a device type.
   *
   * \param deviceInfo Defines the type of device to work on.
   */
  void setDevice(std::shared_ptr<DeviceInfo> deviceInfo);

  /**
   * Attempts to load a serialized executable. If successful then IR
   * preparation and \c snap::Graph compilation are skipped.
   */
  bool tryLoadExecutable();

  /**
   * Throws an error if there is no executable.
   */
  void assertExecutableLoaded() const;

  /**
   * Throws an error if the device cannot be used for offline compilation.
   */
  void assertDeviceCanCompileOffline() const;

  /**
   * Initializes the progress logger to zero
   */
  void initProgressLogger(const SessionOptions &userOptions);

  /**
   * Abstraction of the computation. The Ir is where
   * all the compute graph optimisations, backwards pass construction,
   * re-computation growing etc. happens.
   *
   * This is a shared_ptr rather than a unique_ptr as when binding using pybind,
   * it is impossible to use unique_ptrs as function arguments (see
   * https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html#std-unique-ptr).
   * This had minimal effect on tests etc passing unique_ptrs as converting
   * unique_ptrs to shared_ptrs is possible:
   * https://en.cppreference.com/w/cpp/memory/shared_ptr/operator%3D.
   * Furthermore memory increase would be minimal as only one ref to an Ir is
   * maintained per session.
   */
  std::shared_ptr<Ir> ir;

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

  /**
   *  The name of the session
   */
  std::string name;
};

/**
 * InferenceSession is a runtime instance that provides an interface for
 * executing ONNX graphs on IPU hardware, without any automatic differentiation
 * (backpropagation) or optimization.
 */
class InferenceSession : public Session {

  using Session::Session;

public:
  ~InferenceSession() override;

  static std::unique_ptr<InferenceSession>
  createFromIr(std::shared_ptr<Ir> ir,
               std::shared_ptr<DeviceInfo> deviceInfo,
               const std::string name = DefaultInferenceSessionName);

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
                      const Patterns &patterns             = Patterns(),
                      const std::string name = DefaultInferenceSessionName);
};

/**
 * TrainingSession is a runtime instance that provides an interface for
 * executing ONNX graphs on IPU hardware with training provided by optimizing
 * the specified loss tensor using the specified optimizer and automatic
 * differentiation (backpropagation)
 */
class TrainingSession : public Session {

  using Session::Session;

public:
  ~TrainingSession() override;

  static std::unique_ptr<TrainingSession>
  createFromIr(std::shared_ptr<Ir> ir,
               std::shared_ptr<DeviceInfo> deviceInfo,
               const std::string name = DefaultTrainingSessionName);

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
                      const Patterns &patterns             = Patterns(),
                      const std::string name = DefaultTrainingSessionName);

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
