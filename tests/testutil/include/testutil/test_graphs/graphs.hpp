#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/tensor.hpp>

#include <map>
#include <vector>

/*
  Do not use these graphs as a reference for creating graphs/certain ops.
  Some of the dimensions, TensorTypes etc. may be malformed. The point here
  is just to have a graph with topological dependencies in it.
*/

/*
  Note, in these test graphs, we will manually overwrite the OpIds of the ops
  we create. This is so tests using these ops can statically construct the
  expected data they require corresponding to the test graph.

  For example, they may be testing for the edges between ops, so need to
  construct the "expected" edges using _known_ OpIds, so that the expected edges
  will actually be correct. See [comment-0] in `poprithmstransitiveclosure_test`
  as a full example of this.
*/
namespace test_graphs {

/**
 * add0
 *
 * With no dependencies.
 */
void initSingleOpTestGraph(popart::Graph &graph);

/**
 * add0 -> relu1 ---------> concat4 -> rss5
 *     \                /      \
 *      -> conv2 -> LRN3        -----> rssgrad6
 *
 * (rss is ReduceSumSquare)
 *
 * With extra topo cons:
 *   add0    -> LRN3
 *   relu1   -> LRN3
 *   conv2   -> concat4
 *   conv2   -> rssgrad6
 *   rss5    -> rssgrad6
 */
void initDiamondTestGraph(popart::Graph &graph);

const popart::TensorInfo &withEdgesDefaultTensorInfo();

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
 * tensor whose producer comes earliset in the order, will be connected at
 * index 0, and so on. If multiple inputs do not have a relative ordering
 * enforced by the topology (it would be valid for them to be scheduled in
 * any order), they will be connected in ascending order of Op::id. For
 * example:
 *
 * 0 ---> 2
 *       /
 * 1 ---/
 *
 * Will result in 0's ouput tensor being connected to 2 at index 0, and 1's
 * output tensor being connected to 2 at index 1; because they do not have a
 * strict topological order, so the order of Op::ids is used.
 */
void withEdges(
    popart::Graph &graph,
    const std::vector<std::vector<popart::OpId>> &edges,
    const std::multimap<popart::OpId, popart::OpId> &topoCons,
    const popart::TensorInfo &tensorInfo = withEdgesDefaultTensorInfo());

popart::InIndex NoneInIndex   = -1;
popart::OutIndex NoneOutIndex = -1;

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
          const std::vector<popart::InIndex> mapInputsToNewOp,
          const std::vector<popart::OutIndex> mapOutputsToNewOp);

/*
    -------------------------------------------------------------------> 14
    |
    ------------------------------ 13 ----------------------------------|
    |                                                                   |
    |                                                                   |
    ----------> 3 ------|    15    |----> 8 ----> 9 ----> 10 -----------|
    |                   |    |     |                      ^             |
    0 -> 1 -| ---       |    V     |                      |             |
    |       V   |       ---> 5 --> 6 ---> 7  -------------|             |
    | ----> 2   |       |          ^      ^               |             |
    |           V       |          |      |               |             V
    | --------> 4 ------| ---------| -----|               |---> 11 ---> 12
                                                                        ^
                                                  16 --> 17 --> 18 -----|
                                                  |             ^
                                                  |-------------|
  With additional topo cons:
    13 -> 8
    17 -> 7
    15 -> 6
    7 -> 13
 */
void initMultiInputMultiOutputComplexTestCase(popart::Graph &graph);

} // namespace test_graphs
