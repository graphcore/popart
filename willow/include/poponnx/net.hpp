#ifndef GUARD_NEURALNET_NET_HPP
#define GUARD_NEURALNET_NET_HPP

#include <poponnx/names.hpp>

namespace willow {

/**
 * Net is a runtime instance the provides an interface for executing ONNX
 * graphs on IPU hardware.
 */
class Net {
public:
  /** Create a runtime class for executing an ONNX graph on a set of IPU
   *  hardware.
   *
   * \param model Either an ONNX model protobuf, or the name of a file
   *              containing an ONNX model protobuf
   * \param earlyInfo Information about the shapes of input and output tensors
   * \param dataFlow Configuration for the data feeds and fetches
   * \param losses A list of loss layers to use when training
   * \param optimizer The name of an optimizer to use when training
   * \param cTens List of weight tensors which are not to be updated
   * \param logdir Directory to dump logging information into
   * \param userOptions String to configure net options
   * \param patternNames
   */
  Net(const std::string &model,
      const EarlyInfo &earlyInfo,
      const DataFlow &dataFlow,
      const std::vector<Loss *> &losses,
      const Optimizer *optimizer,
      const std::vector<std::string> &cTens,
      std::string logdir,
      std::string userOptions,
      const std::vector<std::string> &patternNames);

  ~Net();

  /** Update the optimizer.
   *
   * Note that the optimizer passed in must be compatible with that passed to
   * the constructor. For example, you cannot update to an Optimizer which uses
   * momentum here, if the Optimizer passed to the constructor did not have
   * momentum. Reason: The Ir would need to change to incorporate momentum, but
   * the Ir is frozen once constructed. NB: Must call optimizerToDevice for this
   * update to take effect on the device.
   *
   * \param optimizer A pointer to a poponnx::Optimizer
   */
  void updateOptimizer(const Optimizer *optimizer);

  /**
   * Select a device type.
   *
   * /param deviceType One of 'IPU', 'IPU_MODEL', 'CPU'
   */
  void setDevice(const std::string &deviceType);

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
  void train(const StepIO &stepIO);

  /**
  * Perform one evaluation step.
  *
  * input data  : from address in stepIO.in
  * output data : to addresses in stepIO.out
  */
  void evaluate(const StepIO &stepIO);

  /**
  * Perform one inference step.
  *
  * input data  : from address in stepIO.in
  * output data : to addresses in stepIO.out
  */
  void infer(const StepIO &stepIO);

  /**
   * Write current model to ONNX file
   */
  void modelToHost(const std::string &fn);

  /**
   * get the TensorInfo on a Tensor
   */
  TensorInfo getInfo(TensorId) const;

private:
  /**
   * abstraction of the computation, the Ir is where
   * all the compute graph optimisations, backwards pass construction,
   * recomputation growing etc. happens.
   */
  std::unique_ptr<Ir> pir_;

  /**
   * Implementation of the computation, for IPU backend this is
   * where calls to poplar are made.
   */
  std::unique_ptr<Device> device_;
};
} // namespace willow

#endif
