// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PoprithmsTransitiveClosureTest
#include <boost/test/unit_test.hpp>

#include <popart/poprithmstransitiveclosure.hpp>

#include <popart/filereader.hpp>
#include <popart/graph.hpp>
#include <popart/graphid.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/opidentifier.hpp>
#include <popart/scheduler.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/topocons.hpp>

#include "dummy_test_graphs.hpp"

#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>

#include <memory>
#include <tuple>
#include <vector>

using namespace popart;
namespace rithmic = poprithms::schedule::transitiveclosure;

using EdgeMap = rithmic::Edges;

/*
  [comment-0]

  Terminology:
    rTC:   rithmic::TransitiveClosure
    PTC:   popart::PoprithmsTransitiveClosure
    rOpId: rithmic::OpId
    OpId:  popart::OpId

  To test the PTC class, we will need to hand-construct a graph; construct the
  PTC from it; and then compare the underyling actual rTC to an expected rTC we
  have hand-constructed. In other words, we need to be able to hand-construct a
  graph and the rTC we know corresponds to that graph.

  However, how can we construct an expected rTC that we know will be
  _isomorphic_ to the actual rTC constructed from creating a PTC from our test
  graph. That is, they represent the same topology but the vertices can have
  different indices:

    for some invertible f : N -> N; forall i,j : rOpId.
      actual->constrained(i, j) <=> expected.constrained(f(i), f(j))

  The answer is, not knowing anything about the internals of PTC, we do not
  know under what mapping `f` actual and expected are isomorphic. Instead, it is
  easier to have PTC enforce the semantics that its edges will be automorphic to
  the graph (in a way which will be explained below). That is, `f` is the
  identity mapping:

    (1) forall i,j : rOpId.
          actual->constrained(i, j) <=> expected.constrained(i, j)

        which is an equivalent statement to:
          actual.get() == expected

  We now show how to enforce this in PTC.

  First, we fix the ordering of the ops in the graph by exploiting the fact that
  `OpId Op::id` is a public field, so we can manually set them (2).

  Furthermore, we have PTC guarantee that it will preserve the ordering of OpIds
  when creating the rOpIds they correspond to; that the rOpIds will start at 0;
  and that the rOpIds will be contiguous natural numbers. Together, this means
  they will be 0...N, where N is the number of ops in the graph, and that each
  rOpId will be its corresponding OpId's position in their ordering (3).

  Together, (2) and (3) give us (1): they allow us to construct an expected rTC
  that we know is automorphic to the actual rTC underlying the PTC created from
  our graph (providing PTC is not buggy, which is the point!).

  To prove this, consider a graph that has ops `a` and `b` (among others), and
  that they have OpIds `i` and `j`, respectively. Let `t_i` and `t_j` be the
  positions of those OpIds in their ordering. E.g. `i` could be 100, and
  `t_i` could be 5 as it is the 5th smallest OpId. Let `r_i` and `r_j` be the
  rOpIds assigned to `i` and `j` by PTC.

  From (3), we know that `r_i` and `r_j` must in fact be `t_i` and `t_j`. From
  (2) we know what `t_i` and `t_j` are, as we explicitly set all OpIds and thus
  know their ordering. Now, if there is a dependency in the graph from `a` to
  `b`, we know that the actual edge map will have a dependency from `t_i` to
  `t_j`. Again, since we know what `t_i` and `t_j` are, we can statically create
  an expected rTC that we know will be automorphic to the actual one underlying
  the PTC.
*/

namespace {

template <typename TC> struct EdgeMapTestCase {
  Ir ir;
  Graph graph = {ir, GraphId::root()};

  // Subclasses implement `initTestGraph_` and `mkExpectedEdges_`.

  void initTestGraph() {
    TC &tc = static_cast<TC &>(*this);
    tc.initTestGraph_();
  }

  EdgeMap mkExpectedEdges() {
    TC &tc = static_cast<TC &>(*this);
    return tc.mkExpectedEdges_();
  }
};

struct SingleOpTestCase : EdgeMapTestCase<SingleOpTestCase> {
  void initTestGraph_() { test_graphs::initSingleOpTestGraph(graph); }

  EdgeMap mkExpectedEdges_() {
    return {{}}; // add0 is OpId 0; has no dependents.
  }
};

struct DiamondTestCase : EdgeMapTestCase<DiamondTestCase> {
  void initTestGraph_() { test_graphs::initDiamondTestGraph(graph); }

  EdgeMap mkExpectedEdges_() {
    return {
        {1, 2, 3}, // add0
        {3, 4},    // relu1
        {3, 4, 6}, // conv2
        {4},       // LRN3
        {5, 6},    // concat4 (no dependents)
        {6},       // nll5
        {}         // nllgrad6
    };
  }
};

struct ComplexTestCase : EdgeMapTestCase<ComplexTestCase> {
  void initTestGraph_() {
    test_graphs::initMultiInputMultiOutputComplexTestCase(graph);
  }

  EdgeMap mkExpectedEdges_() {
    return {
        {1, 2, 3, 4, 13, 14}, // 0
        {2, 4},               // 1
        {},                   // 2
        {5},                  // 3
        {5, 6, 7},            // 4
        {6},                  // 5
        {7, 8},               // 6
        {10, 11, 13},         // 7
        {9},                  // 8
        {10},                 // 9
        {12},                 // 10
        {12},                 // 11
        {},                   // 12
        {8, 12},              // 13
        {},                   // 14
        {5, 6},               // 15
        {17, 18},             // 16
        {7, 18},              // 17
        {12}                  // 18
    };
  }
};

} // namespace

using TestCaseTypes =
    std::tuple<SingleOpTestCase, DiamondTestCase, ComplexTestCase>;

BOOST_AUTO_TEST_CASE_TEMPLATE(PoprithmsTransitiveClosureTest,
                              TestCase,
                              TestCaseTypes) {
  TestCase tc;

  tc.initTestGraph();

  rithmic::TransitiveClosure expected{tc.mkExpectedEdges()};
  auto actual = PoprithmsTransitiveClosure::fromGraph(tc.graph);

  // 1. Check the mappings are exact inverses of each other and contain every
  // OpId in the graph, and only those OpIds.

  // 1.1. a in Graph ==> exists b s.t. toEdgeOpId(a) = b AND toPopartOpId(b) = a
  for (const auto &opid_op : tc.graph.getOps()) {
    const OpId opid_expected = opid_op.first;
    OpId opid_actual;

    BOOST_REQUIRE_NO_THROW(
        opid_actual = actual.popartOpId(actual.rithmicOpId(opid_expected)));

    BOOST_REQUIRE_EQUAL(opid_expected, opid_actual);
  }

  // 1.2. Check no spurious mappings were created (in either direction).
  {
    const std::size_t expected_size = tc.graph.getOps().size();
    std::size_t actual_size;

    // `numOps` throws ONLY WHEN BUILT IN DEBUG MODE if there is a discrepancy
    // between the forward and inverse mapping sizes. Decided this is better
    // than exposing the internals through `#define private public`.
    BOOST_REQUIRE_NO_THROW(actual_size = actual.numOps());

    BOOST_REQUIRE_EQUAL(expected_size, actual_size);
  }

  // 2. Check the underlying, actual `rithmic::TransitiveClosure` is isomorphic
  // to the expected `rithmic::TransitiveClosure`.
  //
  // Note we don't know under what function expected and actual are isomorphic,
  // but we have specifically constructed them such that they _should_ be
  // automorphic (if the class is not buggy), thus we can use
  // `rithmic::TransitiveClosure::operator==`. See [comment-0].

  BOOST_REQUIRE(actual.get() == expected);
}
