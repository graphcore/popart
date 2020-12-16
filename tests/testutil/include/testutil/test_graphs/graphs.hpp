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

/**
 * Initialises a graph with `DummyOp`s according to the topology specified in
 * `edges` and `topoCons`.
 *
 * `edges`: actual Op->Tensor->Op dependencies in the graph, which will be
 *         created.
 * `topoCons`:  explicit topological constraints that will be encoded in
 *              `graph.topoCons`.
 *
 * The OpIds of the graph must be 0..nOps.
 * This is always (implictly) true anyway as the `edges` are specified as a
 * vector.
 */
void withEdges(popart::Graph &graph,
               const std::vector<std::vector<popart::OpId>> &edges,
               const std::multimap<popart::OpId, popart::OpId> &topoCons);

popart::InIndex NoneInIndex   = -1;
popart::OutIndex NoneOutIndex = -1;

/**
 * Describes how to replace an Op within an existing topology.
 */
struct OpReplacement {
  // Id of the op to replace.
  popart::OpId opIdToReplace;

  // New Op. Ownership will be passed to the graph we are replacing in.
  std::unique_ptr<popart::Op> newOp;

  // How to connect replacement op to the old op's input/output tensors.
  // map[i] = j => tensor at index i will be connected at index j in new op.
  // All indices must be specified. `replaceOp` will not check this.
  // You can drop a tensor and not reconnect it to the new op by specifying
  // destination indices NoneInIndex and NoneOutIndex.
  std::vector<popart::InIndex> mapInputsToNewOp;
  std::vector<popart::OutIndex> mapOutputsToNewOp;
};

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
 * `replacement`: Describes the replacement. See `OpReplacement` definition.
 *
 * Returns `VerticesDisconnectedByReplacement`:
 *   The op and tensors that were disconnected as a result of this replacement.
 *   See `VerticesDisconnectedByReplacement` definition.
 */
VerticesDisconnectedByReplacement replaceOp(popart::Graph &graph,
                                            const OpReplacement &replacement);

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
