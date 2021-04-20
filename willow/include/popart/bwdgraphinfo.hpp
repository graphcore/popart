// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CALLED_SUBGRAPHS_GRAD_INFO_HPP
#define GUARD_NEURALNET_CALLED_SUBGRAPHS_GRAD_INFO_HPP

#include <ostream>

#include <popart/graphid.hpp>
#include <popart/names.hpp>

namespace popart {

/**
 * The type of tensor expected to connect to a graph input or output.
 **/
enum class ExpectedConnectionType {
  /// A tensor from a forward graph.
  Fwd = 0,
  /// The gradient of a tensor from a forward graph.
  FwdGrad = 1
};

/**
 * Description of tensor expected to connect to graph input or output.
 **/
struct ExpectedConnection {
  /// TensorId in the fwdGraph.
  TensorId fwdId;
  /// Either fwdId or getGradId(fwdId).
  ExpectedConnectionType type;

  /// Equality operator.
  bool operator==(const ExpectedConnection &rhs) const;
};

// Shorthand for vector of expected connections.
using ExpectedConnections = std::vector<ExpectedConnection>;

/**
 * A data structure that captures the result of applying autodiff to a graph.
 **/
struct BwdGraphInfo {
  /// A newly constructed backward graph.
  GraphId bwdGraphId;
  /// Expected connection details for each of bwdGraph's inputs.
  ExpectedConnections expectedInputs;
  /// Expected connection details for each of bwdGraph's outputs.
  ExpectedConnections expectedOutputs;

  /// Equality operator.
  bool operator==(const BwdGraphInfo &rhs) const;
};

/**
 * Mapping from fwdGraph to info on the bwdGraph.
 **/
using FwdGraphToBwdGraphInfo = std::map<GraphId, BwdGraphInfo>;

// Stream operators.
std::ostream &operator<<(std::ostream &out, ExpectedConnectionType);
std::ostream &operator<<(std::ostream &out, const ExpectedConnection &);
std::ostream &operator<<(std::ostream &out, const ExpectedConnections &);
std::ostream &operator<<(std::ostream &out, const BwdGraphInfo &);
std::ostream &operator<<(std::ostream &out, const FwdGraphToBwdGraphInfo &);

} // namespace popart

#endif
