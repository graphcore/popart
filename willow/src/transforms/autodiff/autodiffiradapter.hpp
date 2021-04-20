// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_AUTODIFF_IR_ADAPTER_HPP
#define GUARD_NEURALNET_AUTODIFF_IR_ADAPTER_HPP

#include <functional>

#include <transforms/autodiff/autodiffirinterface.hpp>

namespace popart {

/**
 * Concrete class used to fulfill all dependencies for the autodiff classes.
 * It's basically a selective wrapper around popart::Ir.
 */
class AutodiffIrAdapter : public AutodiffIrInterface {
public:
  // Constructor.
  explicit AutodiffIrAdapter(Ir &ir);

  // Get a reference to the main graph.
  virtual Graph &getMainGraph() override;
  // Get schedule order of graphs where parent comes before child.
  virtual std::vector<const Graph *> getGraphSchedule() override;
  // Get schedule order of graphs where parent comes before child.
  virtual std::vector<const Graph *> getGraphSchedule(GraphId root) override;
  // Determine if graph exists.
  virtual bool hasGraph(const GraphId &) const override;
  // Get graph.
  virtual Graph &getGraph(const GraphId &) override;
  // Create a new graph.
  virtual Graph &createGraph(const GraphId &) override;
  // Get the main graph's tensors.
  virtual Tensors &getTensors() override;
  // Get a reference to session options.
  virtual const SessionOptions &getSessionOptions() override;
  // Get optimizer.
  virtual const Optimizer &getOptimizer() override;
  // Get the ID of the final loss tensor.
  virtual TensorId getFinalLossId() override;
  // Get the ID of the op that produces the final loss.
  virtual OpId getFinalLossOpId() override;
  // Get the op set version for a domain.
  virtual int getOpSetVersionFromModel(const std::string &domain) override;
  // Set 'path from loss' flag for all gradient tensors and ops producing them.
  virtual void setMainGraphPathFromLoss() override;
  // Final loss' pipeline stage.
  virtual PipelineStage getFinalLossPipelineStage() override;

private:
  // Reference to ir.
  std::reference_wrapper<Ir> ir;
};

} // namespace popart

#endif
