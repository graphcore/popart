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
 * Session is a runtime instance the provides an interface for executing ONNX
 * graphs on IPU hardware.
 */
class Session {

protected:
  Session();

public:
  virtual ~Session() = 0;

  std::vector<uint32_t> getRNGState();
  void setRNGState(const std::vector<uint32_t>);

  void setRandomSeed(uint64_t seedValue);

  /**
   * Compiles the graph and exports it to the specified path
   *
   * This will create a poplar::Graph and compile the poplar::Executable before
   * exporting the executable and metadata to allow offline running.

   * \arg executablePath path to output the compiled executable and associated
   *                     metadata: if empty, these will not be exported
   * \arg weightsPath path to output the weights: if empty, these will not be
   *                  exported
   */
  void compileAndExport(std::string executablePath, std::string weightsPath);

  /**
   * Prepare the network for execution.
   *
   * This will create the poplar::Graph, poplar::Engine, and setting up
   * poplar::Streams.
   */
  void prepareDevice();

  /**
   * write to device, from an ONNX model loaded from directory
   * Currently, the weights are taken from the onnx Model passed to the
   * constructor, but this should be relaxed so that the weights can
   * come from any Model
   */
  void weightsFromHost();

  /**
   * Copy the weights to host from the device
   */
  void weightsToHost();

  /**
   * Copy the cycle count tensor to host from the device
   */
  uint64_t getCycleCount(std::string id = "");

  /**
   * Perform one step.
   *
   * input data  : from address in stepIO.in
   * debug name  : debug string to identify this run in logs
   * output data : to addresses in stepIO.out
   */
  void run(IStepIO &stepIO, std::string debugName = "");

  /**
   * Export numElements from stepIO.in
   */
  void exportInputs(IStepIO &stepIO,
                    int64_t numElements,
                    const std::string &outputFilename);

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
   * Write current model to ONNX file
   */
  void modelToHost(const std::string &fn);

  /**
   * get the TensorInfo on a Tensor
   */
  TensorInfo getInfo(TensorId) const;

  /**
   * Retrieve the summary from from the poplar::Engine
   *
   * The options which were given to the constructor will influence the
   * information in the report.
   *
   * This may only be called after the `prepareDevice()` call has been made.
   * \arg resetProfile Resets the execution profile
   * \return a string containing the report
   */
  std::string getSummaryReport(bool resetProfile = true) const;

  /**
   * Retrieve the graph report from the poplar::Engine
   *
   * The options which were given to the constructor will influence the
   * information in the report.  By default a JSON format report is produced.
   *
   * This may only be called after the `prepareDevice()` call has been made.
   *
   * \arg useCbor Produce a CBOR formatted report
   * \return a string containing the graph (compilation) report
   */
  std::string getGraphReport(bool useCbor = false) const;

  /**
   * Retrieve the execution report from the poplar::Engine
   *
   * The options which were given to the constructor will influence the
   * information in the report.  By default a JSON format report is produced.
   *
   * This may only be called after the `prepareDevice()` call has been made.
   *
   * \arg useCbor Produce a CBOR formatted report
   * \arg resetProfile Resets the execution profile
   * \return a string containing the execution report
   */
  std::string getExecutionReport(bool useCbor      = false,
                                 bool resetProfile = true) const;

  /**
   * Retrieve the serialized graph from the poplar::Engine
   *
   * A JSON format report is produced.
   *
   * This may only be called after the `prepareDevice()` call has been made.
   *
   * \return a string containing the serialized graph
   */
  std::string getSerializedGraph() const;

  /**
   * Retrieve the tensor tile mapping from the poplar::Graph
   *
   * This may only be called after the `prepareDevice()` call has been made.
   *
   *  \return a TensorTileMap object for all tensors in the graph
   */
  TensorTileMap getTensorTileMap() const;

  /**
   * Reset the weights with the weights in a ONNX model that differs to the
   * current model only in weights. This only updates the weights on the host;
   * the user still needs to call weightsFromHost() after this to update the
   * weights on the device.
   *
   * \param model Either an ONNX model protobuf, or the name of a file
   *              containing an ONNX model protobuf
   * \param ignoreWeightsInModelWithoutCorrespondingHostWeight If true, do
   *        not error if there are initializers in the ONNX model with no
   *        corresponding initializer tensor in the session's IR
   */
  void resetHostWeights(
      const std::string &model,
      const bool ignoreWeightsInModelWithoutCorrespondingHostWeight = false);

  /**
   * Read the weights. Must have called weightsToHost first
   *
   * weight data : to addresses in weightsIo.out
   */
  void readWeights(const IWeightsIO &weightsIo);

  /**
   * Write the weights. Must call weightsFromHost after
   *
   * weight data : to addresses in weightsIo.out
   */
  void writeWeights(const IWeightsIO &weightsIo);

  /**
   * Serizalise the ir graph to a string
   *
   * format : the format to serialize
   */
  std::string serializeIr(IrSerializationFormat format);

  const Ir &getIr() const { return ir; }
  const popx::Devicex &getDevice() const { return *device_; }
  const popx::IrLowering &getIrLowering() const { return *lowering_; }
  const popx::Executablex &getExecutable() const { return *executable_; }

protected:
  /**
   * Select a device type.
   *
   * /param deviceInfo which defines the type of device to work on
   */
  void setDevice(std::shared_ptr<DeviceInfo> deviceInfo); /**

  * Attempts to load a serialized executable. If succesful then Ir
  * preparation and `poplar::Graph` compilation are skipped.
  *
  * \param modelProto An ONNX model protobuf
  * \param dataFlow Configuration for the data feeds and fetches
  * \param userOptions String to configure session options
  * \param deviceInfo which defines the type of device to work on
  */
  bool tryLoadExecutable(const ONNX_NAMESPACE::ModelProto &modelProto,
                         const DataFlow &dataFlow,
                         const SessionOptions &userOptions,
                         std::shared_ptr<DeviceInfo> deviceInfo);

  /**
   * abstraction of the computation, the Ir is where
   * all the compute graph optimisations, backwards pass construction,
   * re-computation growing etc. happens.
   */
  Ir ir;

  /**
   * Implementation of the computation, for IPU back-end this is
   * where calls to poplar are made.
   */
  std::unique_ptr<popx::Devicex> device_;

  /**
   * Implementation of the lowering of the PopART Ir to the
   * poplar Graph.
   */
  std::unique_ptr<popx::IrLowering> lowering_;

  /**
   * The final executable which contains all the data, metadata
   * and configuration parameters necessary to start running
   * the program on the device.
   */
  std::unique_ptr<popx::Executablex> executable_;

  /**
   * Flag to indicate if weightsFromHost has been called
   */
  bool weightsFromHostCalled = false;

  /**
   * Flag to indicate if run has been called
   */
  bool runCalled = false;
};

class InferenceSession : public Session {

  InferenceSession();

public:
  ~InferenceSession() override;

  /** Create a runtime class for executing an ONNX graph on a set of IPU
   *  hardware for inference
   *
   * \param model Either an ONNX model protobuf, or the name of a file
   *              containing an ONNX model protobuf
   * \param inputShapeInfo Information about the shapes of input and output
   *                       tensors
   * \param dataFlow Configuration for the data feeds and fetches
   * \param userOptions String to configure session options
   * \param patterns Optimization patterns to apply
   */

  static std::unique_ptr<InferenceSession>
  createFromOnnxModel(const std::string &model,
                      const DataFlow &dataFlow,
                      std::shared_ptr<DeviceInfo> deviceInfo,
                      const InputShapeInfo &inputShapeInfo = InputShapeInfo(),
                      const SessionOptions &userOptions    = SessionOptions(),
                      const Patterns &patterns             = Patterns());

private:
  void configureFromOnnx(const std::string &model,
                         const DataFlow &dataFlow,
                         const InputShapeInfo &inputShapeInfo,
                         std::shared_ptr<DeviceInfo> deviceInfo,
                         const SessionOptions &userOptions,
                         const Patterns &patterns);
};

class TrainingSession : public Session {

  TrainingSession();

public:
  ~TrainingSession() override;

  /** Create a runtime class for executing an ONNX graph on a set of IPU
   *  hardware for training
   *
   * \param model Either an ONNX model protobuf, or the name of a file
   *              containing an ONNX model protobuf
   * \param inputShapeInfo Information about the shapes of input and output
   *                       tensors
   * \param dataFlow Configuration for the data feeds and fetches
   * \param loss The TensorId of the final scalar loss tensor for training
   * \param optimizer The name of an optimizer to use when training
   * \param userOptions String to configure session options
   * \param patterns Optimization patterns to apply
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
   * reductions on the host. Only populated if hostAllReduce is enabled in the
   * SessionOptions
   */
  const std::vector<std::string> &getHostReduceStreamIds() const;

  /**
   * Access the remote buffers associated with gradient and weight streams
   * that are used in host side all reduce operations. Only populated if
   * hostAllReduce and hostAllReduceRemoteBuffer are enabled.
   */
  const std::map<std::string, poplar::RemoteBuffer> &
  getHostReduceRemoteBuffers() const;

  /**
   * Connect Poplar stream callbacks. In conjunction with
   * `getGradAndVarStreamIds` the streams can be used to copy gradients to the
   * host to perform collective operations after which the variables can be
   * streamed back after they have been updated to the device.
   * `index` referes to the replica index when using replicated graphs.
   */
  void connectStreamToCallback(const std::string &streamHandle,
                               std::function<void(void *)> callback,
                               unsigned index = 0);

  /**
   * Read from a RemoteBuffer object into a user space pointer w.
   * This can be useful when we run larger models with host side
   * reductions since HEXOPT is currently limited to 128 MB
   */
  void copyFromRemoteBuffer(const std::string &buffer,
                            void *w,
                            int repeat_index,
                            unsigned replication_index = 0);

  /**
   * Write to a RemoteBuffer object from a user space pointer w.
   * This can be useful when we run larger models with host side
   * reductions since HEXOPT is currently limited to 128 MB
   */
  void copyToRemoteBuffer(void *w,
                          const std::string &buffer,
                          int repeat_index,
                          unsigned replication_index = 0);

private:
  void configureFromOnnx(const std::string &model,
                         const DataFlow &dataFlow,
                         const TensorId &loss,
                         const Optimizer &optimizer,
                         const InputShapeInfo &inputShapeInfo,
                         std::shared_ptr<DeviceInfo> deviceInfo,
                         const SessionOptions &userOptions,
                         const Patterns &patterns);
};

} // namespace popart

#endif
