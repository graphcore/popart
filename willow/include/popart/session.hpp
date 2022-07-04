// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_NET_HPP
#define GUARD_NEURALNET_NET_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>
#include <pva/pva.hpp>
#include <string>
#include <vector>
#include <popart/ir.hpp>
#include <popart/sessionoptions.hpp>

#include "popart/dataflow.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {

class DeviceInfo;
class IStepIO;
class IWeightsIO;
class InputShapeInfo;
class Optimizer;

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
  /**
   * Destructor for the Session class.
   */
  virtual ~Session() = 0;

  /**
   * Get state of the random number generator.
   */
  std::vector<uint32_t> getRNGState();

  /**
   * Set state of the random number generator.
   */
  void setRNGState(const std::vector<uint32_t>);

  /**
   * Set the value of the random number generator seed.

   * This method explicitly seeds all random operations. Additionally, this
   * method derives a new state for the random number generator (RNG) from the
   * seed and sets it on the device. This RNG state is used to resolve
   * stochastic rounding. Note that to deterministically store and restore the
   * combined random state for a session, do the following:
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
   * \param seedValue The value of the seed.
   */
  void setRandomSeed(uint64_t seedValue);

  /**
   * Get the value of the random number generator seed.

   * Calling setRandomSeed() with this value (at a later stage) reinstates the
   * random state logic that seeds random operations.
   *
   * \returns The value used to seed current random operations.
   */
  uint64_t getRandomSeed();

  /**
   * Compile the graph and export it to a file.
   *
   * This method will first create a \c snap::Graph and compile the \c
   * poplar::Executable. Next, it will export the executable and PopART metadata
   * to the file. The exported file will be in the <a
   * href="https://docs.graphcore.ai/projects/popef/en/latest/"> PopEF</a>
   * format. This means that the file can be used to run inference using the <a
   * href="https://developer.nvidia.com/nvidia-triton-inference-server"> Triton
   * Inference Server</a> with the Graphcore Triton backend. See the <a
   * href="https://docs.graphcore.ai/projects/poplar-triton-backend/en/latest/">
   * Poplar Triton Backend User Guide</a> for more information.
   *
   * \param filename The name of the file where the compiled executable and
   *      metadata will be saved.
   */
  void compileAndExport(const std::string &filename);

  /**
   * Compile the graph and export it to a stream.
   *
   * This method will first create a \c snap::Graph and compile the \c
   * poplar::Executable. Next, it will export the executable and PopART metadata
   * to the stream. The data will be streamed in the <a
   * href="https://docs.graphcore.ai/projects/popef/en/latest/"> PopEF</a>
   * format. This means that the file can be used to run inference using the <a
   * href="https://developer.nvidia.com/nvidia-triton-inference-server"> Triton
   * Inference Server</a> with the Graphcore Triton backend. See the <a
   * href="https://docs.graphcore.ai/projects/poplar-triton-backend/en/latest/">
   * Poplar Triton Backend User Guide</a> for more information.
   *
   * This method automatically creates folders as needed if \p filename is
   * located in a folder which does not exist.
   *
   * \param out The stream that the compiled executable and metadata will be
   *      written to.
   */
  void compileAndExport(std::ostream &out);

  /**
   * Save a compiled graph to a file.
   *
   * The file will be in the <a
   * href="https://docs.graphcore.ai/projects/popef/en/latest/"> PopEF</a>
   * format. This means that the file can be used to run inference using the <a
   * href="https://developer.nvidia.com/nvidia-triton-inference-server"> Triton
   * Inference Server</a> with the Graphcore Triton backend. See the <a
   * href="https://docs.graphcore.ai/projects/poplar-triton-backend/en/latest/">
   * Poplar Triton Backend User Guide</a> for more information.
   *
   * This method automatically creates folders as needed if \p filename is
   * located in a folder which does not exist.
   *
   * \pre prepareDevice() must have been called.
   *
   * \param filename The name of the file where the compiled executable and
   *        metadata will be saved.
   */
  void saveExecutableToFile(const std::string &filename);

  /**
   * Save a compiled graph to a stream.
   *
   * The data will be streamed in the <a
   * href="https://docs.graphcore.ai/projects/popef/en/latest/"> PopEF</a>
   * format. This means that the file can be used to run inference using the <a
   * href="https://developer.nvidia.com/nvidia-triton-inference-server"> Triton
   * Inference Server</a> with the Graphcore Triton backend. See the <a
   * href="https://docs.graphcore.ai/projects/poplar-triton-backend/en/latest/">
   * Poplar Triton Backend User Guide</a> for more information.
   *
   * \pre prepareDevice() must have been called.
   *
   * \param out The stream where the compiled executable and
   *      metadata will be written to.
   */
  void saveExecutableToStream(std::ostream &out);

  /**
   * Check for potential inplacing ambiguities.
   *
   * This method creates an \c AliasModel object for each graph and runs the
   * Poprithms ambiguity checker on it.

   * Throws an error if the graph has an inplacing ambiguity and will prompt the
   * user to check the inplacing.
   *
   * See \c poprithms::memory::inplace::Graph::AmbiguityStatus on the <a
   * href="https://github.com/graphcore/poprithms">Poprithms GitHub repo</a> for
   * more on what constitutes an ambiguity.
   */
  void checkInplacingAmbiguity() const;

  /**
   * Load the compiled executable and metadata from a file.
   *
   * The file must have been created with compileAndExport(const std::string).
   *
   * \param filename The name of the file to load the executable and metadata
   *      from.
   */
  void loadExecutableFromFile(const std::string &filename);

  /**
   * Load the compiled executable and from a stream.
   *
   * The stream must have been created with compileAndExport(std::ostream).
   *
   * \param in The shared pointer to the stream to load the executable from.
   */
  void loadExecutableFromStream(std::shared_ptr<std::istream> in);

  /**
   * Prepare the network for execution.
   *
   * This will create the \c snap::Graph and \c poplar::Engine.
   *
   * \param loadEngine If `true`, load the engine and connect the streams once
   *      the device is ready.
   */
  void prepareDevice(bool loadEngine = true);

  /**
   * Load the engine on the device and connect the streams.
   *
   * This will set up the \c poplar::Streams.
   *
   * Note: This call is optional. The engine will implicitly be loaded
   * on the device when required.
   */
  void loadEngineAndConnectStreams();

  /**
   * Copy weights from the host to the device.
   */
  void weightsFromHost();

  /**
   * Copy the weights from the device to the host steam memory.
   */
  void weightsToHost();

  /**
   * Copy the cycle count tensor from the device to the host.
   *
   * \param id The identifier of the cycle count tensor.
   */
  uint64_t getCycleCount(std::string id = "");

  /**
   * Connect a Poplar stream with a callback.
   *
   *  This method will be called whenever the stream will be read or was written
   *  to by the device. The memory location will only be valid for reading or
   *  writing for the duration of the callback.
   *
   * \param streamHandle The name of the stream to connect to.
   * \param callback The callback to be called whenever the stream is to be read
   *      or was written to by the device.
   * \param index The replica index to connect to, when using replicated
   *      graphs. Default=0.
   */
  void connectStreamToCallback(const std::string &streamHandle,
                               std::function<void(void *)> callback,
                               unsigned index = 0);

  /**
   * Connect a Poplar stream with a fixed location in memory.
   *
   * Each time data is copied to the stream, this location will be read and each
   * time data is copied from the stream, this location will be written.
   *
   * \param streamHandle The handle of the stream to connect to.
   * \param buffer The pointer to the memory location.
   */
  void connectStream(const std::string &streamHandle, void *buffer);

  /**
   * Connect a host function to a callback.
   *
   * The callback takes two arguments, which point to the locations in memory
   * for each of the function's input and output arguments, respectively.
   * During a host function call, first the device transfers the input data to
   * the host, then the callback is invoked, and finally the output data is
   * copied back to the device.
   * The memory pointed to by the callback arguments must only be accessed
   * during the duration of the callback.
   *
   * \param functionHandle The name of the host function.
   * \param callback The function to be called whenever new input data is
   *      available.
   * \param index The replica index to connect to, when using replicated
   *      graphs. Default=0.
   */
  void connectHostFunction(
      const std::string &functionHandle,
      std::function<void(const void *const *, size_t, void *const *, size_t)>
          callback,
      unsigned index = 0);

  /**
   * Run one step.
   *
   * Read input data from address in \p stepIO.in.
   *
   * Write the output data to addresses in \p stepIO.out.
   *
   * \param stepIO The input and output data.
   * \param debugName A debug string to identify this run in logs.
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
   * Update the tensor locations of tensors in the session's ONNX model.
   *
   * A new file will be created at this point, and written to when the ONNX
   * model is saved with a subsequent call to modelToHost().
   *
   * \param fromLocation All externally saved tensors with location \p
   *      fromLocation will have their location updated to \p toLocation.
   * \param toLocation The updated tensor locations. This must not already
   *      exist.
   */
  void updateExternallySavedTensorLocations(const std::string &fromLocation,
                                            const std::string &toLocation);

  /**
   * Write the current model to an ONNX file.
   *
   * \param fn The path to file. The path can be absolute or relative. If you
   *      plan to run your program in multiple processes simultaneously, you
   *      should avoid possible race conditions by writing to different files,
   *      for example by using temporary files.
   */
  void modelToHost(const std::string &fn);

  /**
   * Get the tensor information for a tensor.
   *
   * \param TensorId The identifier of the tensor to get the tensor information
   *      for.
   *
   * \returns The tensor information for the tensor.
   */
  TensorInfo getInfo(TensorId) const;

  /**
   * Check whether a tensor has information.
   *
   * \param TensorId The identifier of the tensor to get the tensor information
   *      for.
   *
   * \returns `true` if the tensor with identifier TensorId has tensor
   *      information and `false` if not.
   */
  bool hasInfo(TensorId) const;

  /**
   * Returns the ids of all tensors in the model.
   *
   * \pre prepareDevice() must have been called.
   */
  std::set<TensorId> getAllTensorIds() const;

  /**
   * Retrieve the summary from from the \c poplar::Engine.
   *
   * The options which were passed to the Session constructor will influence
   * the information in the report.
   *
   * This method may only be called after prepareDevice() has been called.
   *
   * \param resetProfile If `true`, resets the execution profile. Default =
   *      `true`.
   *
   * \return A string containing the report.
   */
  std::string getSummaryReport(bool resetProfile = true) const;

  /**
   * Retrieve the serialized graph from the \c poplar::Engine.
   *
   * A JSON format report is produced.
   *
   * This method may only be called after prepareDevice() has been called.
   *
   * \return A string containing the serialized graph.
   */
  std::string getSerializedGraph() const;

  /**
   * Retrieve the graph report from the \c poplar::Engine.
   *
   * The options which were passed to the Session constructor will influence
   * the information in the report.
   *
   * This method may only be called after prepareDevice() has been called.
   *
   * \return The PopVision Analysis report object.
   */
  pva::Report getReport() const;

  /**
   * Reset weights with weights in an ONNX model.
   *
   * Note that the only differences between the ONNX model and the current model
   * must be the weights. No other differences are allowed.
   *
   * This method only updates the weights on the host. weightsFromHost() must
   * be called after this method to update the weights on the device.
   *
   * \param model An ONNX model protobuf, or the name of a file containing an
   *      ONNX model protobuf.
   * \param ignoreWeightsInModelWithoutCorrespondingHostWeight If `true`, do not
   *      throw an error if there are initializers in the ONNX model without
   *      corresponding initializer tensor(s) in the session's IR.
   */
  void resetHostWeights(
      const std::string &model,
      const bool ignoreWeightsInModelWithoutCorrespondingHostWeight = false);

  /**
   * Read the weights from the host stream memory and write to the host.
   *
   * This method may only be called after weightsToHost() has been called.
   *
   * \param weightsIo The weight data that is read from the host stream memory
   *      is written to the addresses in \p weightsIo.out.
   */
  void readWeights(const IWeightsIO &weightsIo);

  /**
   * Write the weights from the host to the IR tensor memory.
   *
   * This method may only be called after weightsFromHost() has been called.
   *
   * \param weightsIo The weight data is written to the addresses in \p
   *      weightsIo.out.
   */
  void writeWeights(const IWeightsIO &weightsIo);

  /**
   * Serizalise the IR graph to a string.
   *
   * \param format The format to use for serializing.
   */
  std::string serializeIr(IrSerializationFormat format);

  /**
   * Get the IR associated with the Session.
   */
  const Ir &getIr() const { return *ir; }

  /**
   * Get the device associated with the Session.
   */
  const popx::Devicex &getDevice() const { return *device_; }

  /**
   * Get the device associated with the Session.
   */
  popx::Devicex &getDevice() { return *device_; }

  /**
   * Get the IR lowering associated with the Session.
   */
  const popx::IrLowering &getIrLowering() const { return *lowering_; }

  /**
   * Get the executable associated with the Session.
   */
  popx::Executablex &getExecutable() { return *executable_; }

  /**
   * Get the executable associated with the Session.
   */
  const popx::Executablex &getExecutable() const { return *executable_; }

  /**
   * Update cacheEntries from engine cache directory
   * and update ir::hashMatched_ with the updated cacheEntries
   *
   */
  void updateEngineCache();

  /**
   * Set the DeviceInfo of the Session.
   */
  void setDeviceInfo(std::shared_ptr<DeviceInfo> deviceInfo);

protected:
  /**
   * Select a device type.
   *
   * \param deviceInfo The type of device that this session uses.
   */
  void setDevice(std::shared_ptr<DeviceInfo> deviceInfo);

  /**
   * Attempt to load a serialized executable.
   *
   * If successful then IR preparation and \c snap::Graph compilation are
   * skipped.
   */
  bool tryLoadExecutable();

  /**
   * Throw an error if there is no executable.
   */
  void assertExecutableLoaded() const;

  /**
   * Throw an error if the device cannot be used for offline compilation.
   */
  void assertDeviceCanCompileOffline() const;

  /**
   * Initialize the progress logger to zero.
   */
  void initProgressLogger(const SessionOptions &userOptions);

  /**
   * Abstraction of the computation. The Ir is where
   * all the computational graph optimisations, backwards pass construction,
   * re-computation growing and so on happens.
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
   * Implementation of the computation.
   *
   * For the IPU back-end,  this is where calls to Poplar are made.
   */
  std::unique_ptr<popx::Devicex> device_;

  /**
   * Implementation of the lowering of the PopART Ir to the Poplar Graph.
   */
  std::unique_ptr<popx::IrLowering> lowering_;

  /**
   * The final executable which contains all the data, metadata and
   * configuration parameters necessary to start running the program on the
   * device.
   */
  std::unique_ptr<popx::Executablex> executable_;

  /**
   * Information about the device which this session uses.
   */
  std::shared_ptr<DeviceInfo> deviceInfo_;

  /**
   * Check if weightsFromHost() has been called.
   */
  bool weightsFromHostCalled = false;

  /**
   * Check if run() has been called.
   */
  bool runCalled = false;

  /**
   * The map of hashes or filenames of cached executables.
   */
  HashesMap cacheEntries;

  /**
   *  The name of the session.
   */
  std::string name;

private:
  /**
   * Get a hash seed for this session that incorporates various contributing
   * factors that could affect the compilation process (example: Poplar engine
   * options). Note, we pass in, e.g., userOptions instead of using the member
   * userOptions because the member is not guaranteed to be set yet at all
   * call sites.
   * \param userOptions The user options to use.
   * \param deviceInfo The device to use.
   * \return size_t The hash seed.
   */
  size_t getEngineCacheHashSeed(const SessionOptions &userOptions,
                                const DeviceInfo &deviceInfo) const;
};

/**
 * InferenceSession is a runtime instance that provides an interface for
 * executing ONNX graphs on IPU hardware, without any automatic differentiation
 * (backpropagation) or optimization.
 */
class InferenceSession : public Session {

  using Session::Session;

public:
  /**
   * Destructor for the InferenceSession class.
   */
  ~InferenceSession() override;

  /**
   * Create a session for inference from an IR.
   *
   * \param ir The IR to create the session from.
   * \param deviceInfo The type of device that this session uses.
   * \param name The name of this inference session. Default: "inference".
   */
  static std::unique_ptr<InferenceSession>
  createFromIr(std::shared_ptr<Ir> ir,
               std::shared_ptr<DeviceInfo> deviceInfo,
               const std::string name = DefaultInferenceSessionName);

  /**
   * Create a session for inference from an ONNX model.
   *
   * \param model An ONNX model protobuf, or the name of a file containing an
   *      ONNX model protobuf.
   * \param dataFlow Configuration for the data feeds and fetches.
   * \param deviceInfo The type of device that this session uses.
   * \param inputShapeInfo (Optional) The sizes and dtypes of the input
   *      tensors. This
   *      is used to specify the sizes of the input tensors in the case that
   *      the ONNX model does not include this information. The Poplar graph
   *      programmming framework uses statically allocated memory buffers and
   *      so it needs to know the size of tensors before the compilation.
   *      Default: InputShapeInfo().
   * \param userOptions (Optional) The user configuration options for the
   *      Session class. Default: SessionOptions().
   * \param patterns (Optional) A user-selected set of graph transformation
   *      patterns which will be applied to the graph. If this is not
   *      specified, a default set of optimisation transformations will be
   *      applied. Default: Patterns().
   * \param name (Optional) The name of this inference session. Default:
   *      "inference".
   */
  static std::unique_ptr<InferenceSession>
  createFromOnnxModel(const std::string &model,
                      const DataFlow &dataFlow,
                      std::shared_ptr<DeviceInfo> deviceInfo,
                      const InputShapeInfo &inputShapeInfo = InputShapeInfo(),
                      const SessionOptions &userOptions    = SessionOptions(),
                      const Patterns &patterns             = Patterns(),
                      const std::string name = DefaultInferenceSessionName);

  void popxlSetEngineIsLoaded(bool isLoaded);
};

/**
 * TrainingSession is a runtime instance that provides an interface for
 * executing ONNX graphs on IPU hardware with training provided by optimizing a
 * loss tensor using an optimizer and automatic differentiation
 * (backpropagation).
 */
class TrainingSession : public Session {

  using Session::Session;

public:
  /**
   * Destructor for the TrainingSession class.
   */
  ~TrainingSession() override;

  /**
   * Create a session for training from an IR.
   *
   * \param ir The IR to create the session from.
   * \param deviceInfo The type of device that this session uses.
   * \param name The name of this training session. Default: "training".
   */
  static std::unique_ptr<TrainingSession>
  createFromIr(std::shared_ptr<Ir> ir,
               std::shared_ptr<DeviceInfo> deviceInfo,
               const std::string name = DefaultTrainingSessionName);

  /**
   * Create a session for inference from an ONNX model.
   *
   * \param model An ONNX model protobuf, or the name of a file containing an
   *      ONNX model protobuf.
   * \param dataFlow Configuration for the data feeds and fetches.
   * \param loss The identifier of the final scalar loss tensor for training.
   * \param optimizer The name of an optimizer to use when training.
   * \param deviceInfo The type of device that this session uses.
   * \param inputShapeInfo (Optional) The sizes and dtypes of the input
   *      tensors. This
   *      is used to specify the sizes of the input tensors in the case that
   *      the ONNX model does not include this information. The Poplar graph
   *      programmming framework uses statically allocated memory buffers and
   *      so it needs to know the size of tensors before the compilation.
   *      Default: InputShapeInfo().
   * \param userOptions (Optional) The user configuration options for the
   *      Session class. Default: SessionOptions().
   * \param patterns (Optional) A user-selected set of graph transformation
   *      patterns which will be applied to the graph. If this is not
   *      specified, a default set of optimisation transformations will be
   *      applied. Default: Patterns().
   * \param name (Optional) The name of this inference session. Default:
   *      "training".
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
   * Update the optimizer from the host.
   *
   * This method updates the optimizer and the associated hyperparameters but
   * not the optimizer state tensors.
   *
   * **NOTE**: The optimizer parameter has to be compatible with the optimizer
   * passed to the TrainingSession constructor. For example, you cannot call
   * this function with an `SDG1` optimizer if you created the session with an
   * `SDG0` optimizer. This is because it is not possible to change the IR after
   * a session has been constructed.
   *
   * \param optimizer A pointer to a popart::Optimizer.
   */
  void updateOptimizerFromHost(const Optimizer *optimizer);

  /**
   * Copy from a remote butter into a user buffer.
   *
   * This can be useful when we run larger models with host side reductions
   * since HEXOPT is currently limited to 128 MB.
   *
   * \param buffer The name of the remote buffer to copy from.
   * \param w Pointer to a user buffer to copy to.
   * \param repeat_index The index in the remote buffer to copy from.
   * \param replication_index The replicated graph index when using replicated
   * graphs. Default=0.
   */
  void copyFromRemoteBuffer(const std::string &buffer,
                            void *w,
                            int repeat_index,
                            unsigned replication_index = 0);

  /**
   * Copy from a user buffer to a remote buffer.
   *
   * This can be useful when we run larger models with host side reductions
   * since HEXOPT is currently limited to 128 MB.
   *
   * \param w Pointer to a user buffer to copy from.
   * \param buffer The remote buffer to copy to.
   * \param repeat_index The index in the remote buffer to copy to.
   * \param replication_index The replicated graph index when using replicated
   * graphs. Default=0.
   */
  void copyToRemoteBuffer(void *w,
                          const std::string &buffer,
                          int repeat_index,
                          unsigned replication_index = 0);
};

} // namespace popart

#endif
