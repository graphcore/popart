// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_TEST_GRAPHS_DUMMY_BUILDER_HPP_
#define POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_TEST_GRAPHS_DUMMY_BUILDER_HPP_

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "popart/names.hpp"

namespace popart {
class Graph;
class Op;
class Tensor;
class TensorInfo;
} // namespace popart

namespace test_graphs {
namespace dummy_builder {

const popart::TensorInfo &defaultTensorInfo();

/**
 * Initialises a graph with `DummyOp`s according to the topology specified
 * in `edges` and `topoCons`.
 *
 * `edges`: actual Op->Tensor->Op dependencies in the graph, which will be
 *         created.
 * `topoCons`:  explicit topological constraints that will be encoded in
 *              `graph.topoCons`.
 * `tensorInfo`: TensorInfo to use for ALL tensors created.
 *
 * The OpIds of the graph must be 0..nOps.
 * This is always (implictly) true anyway as the `edges` are specified as a
 * vector.
 *
 * To make the produced graph deterministic in a predictable way, if a
 * `DummyOp` has multiple inputs, they will be connected at indices
 * reflecting their relative topological (schedule) order (based on the
 * inherent graph topology only, not the extra topo cons). So, the input
 * tensor whose producer comes earliest in the order, will be connected at
 * index 0, and so on. If multiple inputs do not have a relative ordering
 * enforced by the topology (it would be valid for them to be scheduled in
 * any order), they will be connected in ascending order of Op::id. For
 * example:
 *
 * 0 ---> 2
 *       /
 * 1 ---/
 *
 * Will result in 0's output tensor being connected to 2 at index 0, and 1's
 * output tensor being connected to 2 at index 1; because they do not have a
 * strict topological order, so the order of Op::ids is used.
 */
void withEdges(popart::Graph &graph,
               const std::vector<std::vector<popart::OpId>> &edges,
               const std::multimap<popart::OpId, popart::OpId> &topoConps,
               const popart::TensorInfo &tensorInfo = defaultTensorInfo());

extern popart::InIndex NoneInIndex;
extern popart::OutIndex NoneOutIndex;

struct VerticesDisconnectedByReplacement {
  // Replaced op.
  // If `OpReplacement::eraseOldOp` was true, then this will be null as the op
  // has been erased and thus destroyed.
  popart::Op *op;

  // Tensors left disconnected by the replacement and the indices at which they
  // were connected to the old op.
  std::unordered_map<popart::InIndex, popart::Tensor *> inTensors;
  std::unordered_map<popart::OutIndex, popart::Tensor *> outTensors;
};

/**
 * In the given graph, perform the described OpReplacement.
 *
 * All properties, like Op::fromLoss, Op::Settings::batchSerializationPhase,
 * etc., will be copied from the old op to the new op; except where it is not
 * sensible to copy the property, like `Op::Settings::name`. Popart does not
 * provide a robust way of copying over only the "sensible" properties. Please
 * see the definition of this function for exactly what is copied and what is
 * not.
 *
 * Note, in particular, the Op::debugInfo of the new Op is preservered, so you
 * will lose some useful information held in the old op's Op::debugInfo that
 * ideally would be transferred over.
 *
 * `graph`: The graph in which to replace an Op with another Op.
 * `opIdToReplace`: Op::Id of op to replace.
 * `newOpUp`: New op to insert. Ownership will be moved to the graph.
 * `mapInputsToNewOp`: How to connect new op to the old op's input tensors.
 *                     mapInputsToNewOp[i] = j => input tensor connected to old
 *                     op at index i will be connected as an input tensor to new
 *                     op at index j.
 *                     You can drop a tensor and not reconnected it to the new
 *                     op by specifying destination index NoneInIndex.
 * `mapOutputsToNewOp`: Similarly for output tensors. Drop tensors with index
 *                      NoneOutIndex.
 *
 * Returns `VerticesDisconnectedByReplacement`:
 *   The op and tensors that were disconnected as a result of this replacement.
 *   See `VerticesDisconnectedByReplacement` definition.
 */
VerticesDisconnectedByReplacement
replaceOp(popart::Graph &graph,
          const popart::OpId opIdToReplace,
          std::unique_ptr<popart::Op> newOpUp,
          const std::vector<popart::InIndex> &mapInputsToNewOp,
          const std::vector<popart::OutIndex> &mapOutputsToNewOp);

} // namespace dummy_builder
} // namespace test_graphs

#endif // POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_TEST_GRAPHS_DUMMY_BUILDER_HPP_
