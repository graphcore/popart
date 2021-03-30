
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_AUTODIFF_IR_INTERFACE_HPP
#define GUARD_NEURALNET_AUTODIFF_IR_INTERFACE_HPP

#include <popart/names.hpp>

namespace popart {

// Forward declarations.
class Ir;
class Op;
class Optimizer;
class Tensor;
class SessionOptions;

// NOTE: In this file we declare an interface that encapsulates the dependency
// that a collection of grad growing functions (that previously lived in
// popart::Ir) have on code in popart::Ir. This is the interface through which
// the autodiff transform communicates with the outside world.
//
// Capturing the dependency explicitly in an interface will allows us
// to provide an alternative implementation of these dependencies (such as mock
// classes) in future. This dependency breaking technique is referred to as
// 'Extract Interface' in 'Working Effectively with Legacy Code' and allows us
// to decouple the grad growing code from popart::Ir as much as possible.

/**
 * Abstract base class that encapsulates the interface that autodiff needs
 * to interact with popart::Ir.
 */
class AutodiffIrInterface {
public:
  // Get a reference to the main graph.
  virtual Graph &getMainGraph() = 0;
  // Get the main graph's tensors.
  virtual Tensors &getTensors() = 0;
  // Get a reference to session options.
  virtual const SessionOptions &getSessionOptions() = 0;
  // Get optimizer.
  virtual const Optimizer &getOptimizer() = 0;
  // Get the ID of the final loss tensor.
  virtual TensorId getFinalLossId() = 0;
  // Get the ID of the op that produces the final loss.
  virtual OpId getFinalLossOpId() = 0;
  // Get the op set version for a domain.
  virtual int getOpSetVersionFromModel(const std::string &domain) = 0;
  // Set 'path from loss' flag for all gradient tensors and ops producing them.
  virtual void setMainGraphPathFromLoss() = 0;
  // Final loss' pipeline stage.
  virtual PipelineStage getFinalLossPipelineStage() = 0;
};

} // namespace popart

#endif
