// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MAINLOOPS_HPP
#define GUARD_NEURALNET_MAINLOOPS_HPP

#include <cstddef>
#include <string>
#include <utility>
#include <popart/transforms/transform.hpp>

namespace popart {
class LoopOp;
class Graph;
class Ir;

class MainLoops : public Transform {
private:
  void validateAnchorReturnType(Graph &) const;
  std::pair<LoopOp *, LoopOp *> setupLoops(Graph &) const;
  void setupAnchors(Graph &, LoopOp *, LoopOp *) const;

public:
  static std::size_t id();

  MainLoops() : Transform() {}
  virtual ~MainLoops() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "MainLoops"; }

  /**
   * Return the name of the step subgraph.
   *
   * The step subgraph is the body of the \c LoopOp \c stepLoop .
   * The \c stepLoop is run when \c session.run(...) is called, and will run
   * \c batchesPerStep number of times (i.e. the \c trip_count of the loop
   * equals \c batchesPerStep ).
   * A \a step thus constitutes a call to \c session.run(...) .
   * As a call to \c session.run(...) involves a call to \c engine.run()
   * (which is expensive, and will involve returning to the host for more data)
   * we would like to have as large a \c batchesPerStep as possible.
   *
   * See https://github.com/onnx/onnx/blob/master/docs/Operators.md#Loop for
   * details about the loop operator
   *
   * \return The name of the step graph
   **/
  static std::string getStepGraphName() { return "stepGraph"; }
  /**
   * Return the name of the gradient accumulation subgraph.
   *
   * The gradient accumulation subgraph is the body of the \c LoopOp
   * \c accumLoop .
   * The \c accumLoop will run \c accumulationFactor number of times
   * (i.e. the \c trip_count of the loop equals \c batchesPerStep )
   * and will accumulate the gradients for each pass.
   * These accumulated gradients will be used to calculate the weigth update.
   *
   * See https://github.com/onnx/onnx/blob/master/docs/Operators.md#Loop for
   * details about the loop operator
   *
   * \return The name of the accumulation graph
   **/
  static std::string getAccumulationGraphName() { return "accumulationGraph"; }

  /**
   * Helper function for accessing the subgraph of the inner loop
   *
   * .. warning::
   *
   *    Should only be used after the transform has been applied
   *    (i.e. after call to apply() has been made).
   *
   * The inner loop depends on the values of \c accumulationFactor and
   * \c batchesPerStep.
   * The inner loop equals:
   * - The \c mainGraph if \c accumulationFactor = 1 and \c batchesPerStep = 1
   * - The \c accumulationGraph if \c accumulationFactor > 1 and
   *   \c batchesPerStep = 1
   * - The \c stepGraph if \c accumulationFactor = 1 and \c batchesPerStep > 1
   * - The \c accumulationGraph if \c accumulationFactor > 1 and
   *   \c batchesPerStep > 1
   *
   * .. note::
   *
   *    \c innerLoop and \c outerLoop are represented by the differnt graphs
   *    only when \c accumulationFactor > 1 and \c batchesPerStep > 1.
   *    In that case the \c outerLoop repeats the \c innerLoop
   *
   * \return The inner loop subgraph
   **/
  static Graph &getInnerLoopSubgraph(Ir &ir);
  /**
   * Helper function for accessing the subgraph of the outer loop
   *
   * .. warning::
   *
   *    Should only be used after the transform has been applied
   *    (i.e. after call to apply() has been made).
   *
   * The outer loop depends on the values of \c accumulationFactor and
   * \c batchesPerStep.
   * The outer loop equals:
   * - The \c mainGraph if \c accumulationFactor = 1 and \c batchesPerStep = 1
   * - The \c accumulationGraph if \c accumulationFactor > 1 and
   *   \c batchesPerStep = 1
   * - The \c stepGraph if \c accumulationFactor = 1 and \c batchesPerStep > 1
   * - The \c stepGraph if \c accumulationFactor > 1 and \c batchesPerStep > 1
   *
   * .. note::
   *
   *    \c innerLoop and \c outerLoop are represented by the differnt graphs
   *    only when \c accumulationFactor > 1 and \c batchesPerStep > 1.
   *    In that case the \c outerLoop repeats the \c innerLoop
   *
   * \return The outer loop subgraph
   **/
  static Graph &getOuterLoopSubgraph(Ir &ir);

  static LoopOp *getInnerLoopOp(Ir &ir);
};

} // namespace popart

#endif
