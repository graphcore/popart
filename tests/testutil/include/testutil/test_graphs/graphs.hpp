#include <popart/graph.hpp>

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
