#ifndef GUARD_NEURALNET_SUBGRAPHSNAMES_HPP
#define GUARD_NEURALNET_SUBGRAPHSNAMES_HPP

namespace fwtools {
namespace subgraph {

// To compare nodes in the graph for equivalence, use a std::string
using EquivId = std::string;

// The position in the schedule that a sequence (sub-graph) starts at
using Start = int;

// The input and output indices of a node
using InIndex  = int;
using OutIndex = int;

} // namespace subgraph
} // namespace fwtools

#endif
