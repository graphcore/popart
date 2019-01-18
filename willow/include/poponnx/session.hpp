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

  Session();

public:
  ~Session();

  /** Create a runtime class for executing an ONNX graph on a set of IPU
   *  hardware.
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

  static std::unique_ptr<Session>
  createFromOnnxModel(const std::string &model,
                      const DataFlow &dataFlow,
                      const InputShapeInfo &inputShapeInfo = InputShapeInfo(),
                      const std::vector<Loss *> &losses    = {},
                      const Optimizer *optimizer           = nullptr,
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
   * Select a device type.
   *  TODO : This function should return a new class on which you can perform
   * operations which need the device
   *
   * /param deviceInfo which defines the type of device to work on
   */
  void setDevice(DeviceInfo &deviceInfo);

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
   * write whatever optimizer tensors (learning rates,
   * momentum, initial momentum tensors (zero)) there are to device
   */
  void optimizerFromHost();

  /**
   * Perform one training step.
   *
   * input data  : from address in stepIO.in
   * output data : to addresses in stepIO.out
   */
  void train(const IStepIO &stepIO);

  /**
   * Perform one evaluation step.
   *
   * input data  : from address in stepIO.in
   * output data : to addresses in stepIO.out
   */
  void evaluate(const IStepIO &stepIO);

  /**
   * Perform one inference step.
   *
   * input data  : from address in stepIO.in
   * output data : to addresses in stepIO.out
   */
  void infer(const IStepIO &stepIO);

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
   * information in the report.
   *
   * This may only be called after the `prepareDevice()` call has been made.
   *
   * \return a string containing the graph (compilation) report
   */
  std::string getGraphReport() const;

  /**
   * Retrieve the execution report from the poplar::Engine
   *
   * The options which were given to the constructor will influence the
   * information in the report.
   *
   * This may only be called after the `prepareDevice()` call has been made.
   *
   * \return a string containing the execution report
   */
  std::string getExecutionReport() const;

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

  std::vector<const Node &> listConstExprNodes(const onnx::ModelProto &) const;
  std::vector<const Node &>
  listNonConstExprNodes(const onnx::ModelProto &) const;

private:
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

  void configureFromOnnx(const std::string &model,
                         const DataFlow &dataFlow,
                         const InputShapeInfo &inputShapeInfo,
                         const std::vector<Loss *> &losses,
                         const Optimizer *optimizer,
                         const SessionOptions &userOptions,
                         const Patterns &patterns);
};
} // namespace poponnx

#endif
