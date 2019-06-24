#ifndef GUARD_NEURALNET_NET_HPP
#define GUARD_NEURALNET_NET_HPP

#include <poponnx/ir.hpp>
#include <poponnx/names.hpp>

namespace poponnx {

struct SessionOptions;
class Patterns;
class DeviceInfo;

/**
 * Session is a runtime instance the provides an interface for executing ONNX
 * graphs on IPU hardware.
 */
class Session {

protected:
  Session();

public:
  virtual ~Session() = 0;

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
   * Perform one step.
   *
   * input data  : from address in stepIO.in
   * output data : to addresses in stepIO.out
   */
  void run(const IStepIO &stepIO);

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
   *
   * \return a string containing the report
   */
  std::string getSummaryReport() const;

  /**
   * Retrieve the graph report from the poplar::Engine
   *
   * The options which were given to the constructor will influence the
   * information in the report.  By default a JSON format report is produced.
   *
   * This may only be called after the `prepareDevice()` call has been made.
   *
   * \arg use_cbor Produce a CBOR formatted report
   * \return a string containing the graph (compilation) report
   */
  std::string getGraphReport(bool use_cbor = false) const;

  /**
   * Retrieve the execution report from the poplar::Engine
   *
   * The options which were given to the constructor will influence the
   * information in the report.  By default a JSON format report is produced.
   *
   * This may only be called after the `prepareDevice()` call has been made.
   *
   * \arg use_cbor Produce a CBOR formatted report
   * \return a string containing the execution report
   */
  std::string getExecutionReport(bool use_cbor = false) const;

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
   */
  void resetHostWeights(const std::string &model);

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

protected:
  /**
   * Select a device type.
   *
   * /param deviceInfo which defines the type of device to work on
   */
  void setDevice(std::shared_ptr<DeviceInfo> deviceInfo);

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
  std::unique_ptr<Device> device_;

  /**
   * Flag to indicate if weightsFromHost has been called
   */
  bool weightsFromHostCalled = false;
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
   * \param losses An optional list of loss layers to use after inference
   * \param dataFlow Configuration for the data feeds and fetches
   * \param userOptions String to configure session options
   * \param patterns Optimization patterns to apply
   */

  static std::unique_ptr<InferenceSession>
  createFromOnnxModel(const std::string &model,
                      const DataFlow &dataFlow,
                      std::shared_ptr<DeviceInfo> deviceInfo,
                      const std::vector<Loss *> &losses    = {},
                      const InputShapeInfo &inputShapeInfo = InputShapeInfo(),
                      const SessionOptions &userOptions    = SessionOptions(),
                      const Patterns &patterns             = Patterns());

private:
  void configureFromOnnx(const std::string &model,
                         const DataFlow &dataFlow,
                         const std::vector<Loss *> &losses,
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
   * \param losses A list of loss layers to use when training
   * \param optimizer The name of an optimizer to use when training
   * \param userOptions String to configure session options
   * \param patterns Optimization patterns to apply
   */

  static std::unique_ptr<TrainingSession>
  createFromOnnxModel(const std::string &model,
                      const DataFlow &dataFlow,
                      const std::vector<Loss *> &losses,
                      const Optimizer &optimizer,
                      std::shared_ptr<DeviceInfo> deviceInfo,
                      const InputShapeInfo &inputShapeInfo = InputShapeInfo(),
                      const SessionOptions &userOptions    = SessionOptions(),
                      const Patterns &patterns             = Patterns());

  /** Update the optimizer.
   *
   * Note that the optimizer passed in must be compatible with that passed to
   * the constructor. For example, you cannot update to an Optimizer which uses
   * momentum here, if the Optimizer passed to the constructor did not have
   * momentum. Reason: The Ir would need to change to incorporate momentum, but
   * the Ir is frozen once constructed. NB: Must call optimizerFromHost for this
   * update to take effect on the device.
   *
   * \param optimizer A pointer to a poponnx::Optimizer
   */
  void updateOptimizer(const Optimizer *optimizer);

  /**
   * write whatever optimizer tensors (learning rates,
   * momentum, initial momentum tensors (zero)) there are to device
   */
  void optimizerFromHost();

private:
  void configureFromOnnx(const std::string &model,
                         const DataFlow &dataFlow,
                         const std::vector<Loss *> &losses,
                         const Optimizer &optimizer,
                         const InputShapeInfo &inputShapeInfo,
                         std::shared_ptr<DeviceInfo> deviceInfo,
                         const SessionOptions &userOptions,
                         const Patterns &patterns);
};

} // namespace poponnx

#endif
