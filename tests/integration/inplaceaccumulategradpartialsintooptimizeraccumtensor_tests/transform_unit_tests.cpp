// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE                                                      \
  InplaceAccumulateGradPartialsIntoOptimizerAccumTensorTests

#include <boost/test/unit_test.hpp>

#include <testutil/test_graphs/builder.hpp>
#include <testutil/test_graphs/op/dummy.hpp>

#include <popart/transforms/inplaceaccumulategradpartialsintooptimizeraccumtensor.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/op/add.hpp>
#include <popart/op/init.hpp>
#include <popart/opidentifier.hpp>

#include <iostream>
#include <map>
#include <unordered_set>

using namespace popart;
using test_graphs::DummyOp;

namespace {

/**
 * \brief Helper class for testing equality between graphs.
 *
 * We do not consider TensorIds and OpIds when testing equality between graphs,
 * only the nature of the Tensors and Ops themselves - e.g. their surrounding
 * topoplogy, their pipeline stage, etc.
 */
class GraphEqualityTester {
public:
  // all default ctor/assignment/dtor

  GraphEqualityTester(const Graph &g, const Graph &h) {
    REQUIRE_EQUAL_GRAPH_STATE(g, h);
  }

  /**
   * Tests equality between two Ops. This includes testing equality of all input
   * and output tensors.
   */
  void REQUIRE_EQUAL_OP(const Op *o, const Op *p) {
    REQUIRE_EQUAL_VERTEX(o, p);

    BOOST_REQUIRE_EQUAL(o->getOptionalVGraphId(), p->getOptionalVGraphId());
    BOOST_REQUIRE_EQUAL(o->getOptionalExecutionPhase(),
                        p->getOptionalExecutionPhase());
    BOOST_REQUIRE_EQUAL(o->getOptionalBatchSerializedPhase(),
                        p->getOptionalBatchSerializedPhase());
    BOOST_REQUIRE_EQUAL(o->getOptionalPipelineStage(),
                        p->getOptionalPipelineStage());
    BOOST_REQUIRE_EQUAL(o->getScope(), p->getScope());

    REQUIRE_EQUAL_TENSORINDEXMAP(*o->input, *p->input);
    REQUIRE_EQUAL_TENSORINDEXMAP(*o->output, *p->output);
  }

  void REQUIRE_EQUAL_TENSOR(const Tensor *a, const Tensor *b) {
    REQUIRE_EQUAL_VERTEX(a, b);

    TensorIdPair pair = {a->id, b->id};

    if (verifiedTensorPairs.count(pair) > 0u) {
      return;
    }

    // Can't check all state comprehensively.
    BOOST_REQUIRE_EQUAL(a->tensorType(), b->tensorType());
    BOOST_REQUIRE(a->getReplicatedStreamMode() == b->getReplicatedStreamMode());
    BOOST_REQUIRE_EQUAL(a->getPipelineStages(), b->getPipelineStages());
    BOOST_REQUIRE_EQUAL(a->consumers.getTotal(), b->consumers.getTotal());
    BOOST_REQUIRE_EQUAL(a->info.isSet(), b->info.isSet());
    BOOST_REQUIRE_EQUAL(a->info.dataType(), b->info.dataType());
    BOOST_REQUIRE_EQUAL(a->info.shape(), b->info.shape());
    BOOST_REQUIRE_EQUAL(a->info.metaShape(), b->info.metaShape());
    BOOST_REQUIRE_EQUAL(a->isGraphInput(), b->isGraphInput());
    BOOST_REQUIRE_EQUAL(a->isGraphOutput(), b->isGraphOutput());
    BOOST_REQUIRE_EQUAL(a->isOptimizerTensor(), b->isOptimizerTensor());
    BOOST_REQUIRE_EQUAL(a->hasTensorData(), b->hasTensorData());
    BOOST_REQUIRE_EQUAL(a->hasVirtualGraphId(), b->hasVirtualGraphId());
    BOOST_REQUIRE_EQUAL(a->isUnmodifiable(), b->isUnmodifiable());
    BOOST_REQUIRE(a->tensorLocationInfo == b->tensorLocationInfo);

    verifiedTensorPairs.insert(pair);
  }

  void REQUIRE_EQUAL_INIT_OP(const Op *o, const Op *p) {
    REQUIRE_EQUAL_OP(o, p);

    const auto i = REQUIRE_CONVERTIBLE<InitOp>(o);
    const auto j = REQUIRE_CONVERTIBLE<InitOp>(p);

    BOOST_REQUIRE_EQUAL(i->getTensorInfo(), j->getTensorInfo());
    BOOST_REQUIRE_EQUAL(i->getTensorType(), j->getTensorType());
    BOOST_REQUIRE(i->getInitType() == j->getInitType());
  }

  void REQUIRE_EQUAL_ADDLHS_OP(const Op *o, const Op *p) {
    REQUIRE_EQUAL_OP(o, p);

    REQUIRE_CONVERTIBLE<AddLhsInplaceOp>(o);
    REQUIRE_CONVERTIBLE<AddLhsInplaceOp>(p);
  }

  void REQUIRE_EQUAL_ACCUMULATE_OP(const Op *o, const Op *p) {
    REQUIRE_EQUAL_OP(o, p);

    const auto a = REQUIRE_CONVERTIBLE<AccumulateOp>(o);
    const auto b = REQUIRE_CONVERTIBLE<AccumulateOp>(p);

    BOOST_REQUIRE(a->getAccumulationType() == b->getAccumulationType());
    BOOST_REQUIRE(a->getFactor() == b->getFactor());
  }

  void REQUIRE_EQUAL_DUMMY_OP(const Op *o, const Op *p) {
    REQUIRE_EQUAL_OP(o, p);
    REQUIRE_CONVERTIBLE<DummyOp>(o);
    REQUIRE_CONVERTIBLE<DummyOp>(p);
  }

private:
  using TensorIdPair = std::tuple<TensorId, TensorId>;

  struct TensorIdPairHash
      : public std::unary_function<TensorIdPair, std::size_t> {
    std::size_t operator()(const TensorIdPair &k) const {
      return std::hash<TensorId>{}(std::get<0>(k)) ^
             std::hash<TensorId>{}(std::get<1>(k));
    }
  };

  std::unordered_set<TensorIdPair, TensorIdPairHash> verifiedTensorPairs;

  /**
   * Performs some rudimentary testing of the state of the two graphs, e.g.
   * the same number of ops, the same number of inputs, etc.
   */
  void REQUIRE_EQUAL_GRAPH_STATE(const Graph &g, const Graph &h) {
    // In this function, avoid calling methods that massively increase the scope
    // of these tests, like Graph::isSchedulable, which would call into the
    // scheduler.

    // Check GraphId.
    BOOST_REQUIRE_EQUAL(g.id, h.id);

    // Check number ops and tensors.
    BOOST_REQUIRE_EQUAL(g.getOps().size(), h.getOps().size());
    BOOST_REQUIRE_EQUAL(g.getTensors().n(), h.getTensors().n());

    // Check graph inputs and outputs.
    BOOST_REQUIRE_EQUAL(g.getInputIds().size(), h.getInputIds().size());
    BOOST_REQUIRE_EQUAL(g.getOutputIds().size(), h.getOutputIds().size());

    // Check call site ops.
    BOOST_REQUIRE_EQUAL(g.getCallSiteOps().size(), h.getCallSiteOps().size());

    // Check virtual graph ids.
    BOOST_REQUIRE_EQUAL(g.getAllVirtualGraphIds(true),
                        h.getAllVirtualGraphIds(true));

    // Check Scope.
    BOOST_REQUIRE_EQUAL(g.getScope(), h.getScope());

    // Check called graphs.
    BOOST_REQUIRE_EQUAL(h.getCalledGraphs().size(), g.getCalledGraphs().size());
  }

  template <typename T> //
  const T *REQUIRE_CONVERTIBLE(const Op *op) {
    const auto downOp = dynamic_cast<const T *>(op);
    BOOST_REQUIRE(downOp);
    return downOp;
  }

  void REQUIRE_EQUAL_TENSORINDEXMAP(const TensorIndexMap &om,
                                    const TensorIndexMap &pm) {
    BOOST_REQUIRE_EQUAL(om.n(), pm.n());

    for (const auto &idx_ten : om.tensorMap()) {
      const auto idx = idx_ten.first;
      const auto t   = idx_ten.second;

      BOOST_REQUIRE(pm.hasIndex(idx));
      const auto u = pm.tensor(idx);

      REQUIRE_EQUAL_TENSOR(t, u);
    }
  }

  void REQUIRE_EQUAL_VERTEX(const Vertex *v, const Vertex *w) {
    BOOST_REQUIRE(v);
    BOOST_REQUIRE(w);
    BOOST_REQUIRE_EQUAL(v->fromLoss, w->fromLoss);
    BOOST_REQUIRE_EQUAL(v->toLoss, w->toLoss);
    BOOST_REQUIRE_EQUAL(v->scheduledPreLoss, w->scheduledPreLoss);
  }
};

template <bool ShouldLog = false>
void logSchedule(const Graph &graph, const std::string &title);

std::unique_ptr<AccumulateOp>
mkAccumulateOp(Graph &graph, const std::string &name, bool isConst = true);

std::unique_ptr<DummyOp> mkDummyOp(Graph &graph, const std::string &name);

// Unsafe, user must statically know they are passing an Op downcastable to
// DummyOp.
InIndex dummyNextInIndex(Op &dummy);

std::unique_ptr<AddLhsInplaceOp> mkAddLhsOp(Graph &graph,
                                            const std::string &name);

std::unique_ptr<InitOp>
mkInitOp(Graph &graph, const TensorInfo &tInfo, const std::string &name);

} // namespace

BOOST_AUTO_TEST_CASE(TestCanHandleEmptyGraph) {
  Ir irExpected;
  Graph &expected = irExpected.getMainGraph();

  Ir ir;
  Graph &graph = ir.getMainGraph();

  InplaceAccumulateGradPartialsIntoOptimizerAccumTensor transform;
  bool modified = transform.apply(graph);

  // Should be same as unmodified graph.
  BOOST_REQUIRE(!modified);
  GraphEqualityTester{graph, expected};
}

BOOST_AUTO_TEST_CASE(TestReplacesAddsWithAccumulateOpsOnOptimizerAccum) {
  /*
    We start with an inplace addition tree of length 3 on dw0, which is fed into
    an AccumulateOp that is inplace on accum.

    The transform will change this to be a tree of inplace AccumulateOps on
    accum of length 3.

    Original Graph:

    Init4
     |
    dW0              pW0
        \             /
        AddLhsInPlace0
              |
            dW1              pW1
                \             /
                AddLhsInPlace1
                      |
                    dW2              pW2
                        \             /
                        AddLhsInPlace2          Init5
                              |                   |
                              dW3   accum --------|
                                \    |
                                Accumulate3
                                     |
                                   accum1
                                     |
                                     |------------ Dummy6 -- d
    topo cons:
      Init7 -> AddLhsInplace1
      Init7 -> Accumulate3
      Accumulate3 -> Init8

    Where:
      Ti (upper camel case): Op of type `T` with id i.
      t (regular camel case): Tensor.

      dW: accumulator for inplace addition tree.
      pW: tensors to be summed by the addition tree.
      accum: optimizer's accum tensor.

    Becomes:

        |------------ Init5
        |
      accum   pw0
        |     /
    Accumulate
        |
      accum1        pw1
          \         /
          Accumulate'
                |
              accum2         pW2
                  \          /
                  Accumulate''
                      |
                    accum1
                      |
                      -------------- Dummy6 -- d
    topo cons:
      Init7 -> Accumulate'
      Init7 -> Accumulate
      Accumulate'' -> Init8

    NOTE: Ops without an Id are new Ops whose Id we do not care about.
  */

  Ir actualIr;
  Graph &actual = actualIr.getMainGraph();
  Ir expectedIr;
  Graph &expected = expectedIr.getMainGraph();

  const TensorInfo tInfo{DataType::FLOAT, Shape{4, 4, 4, 4}};

  const TensorId pW0 = "pW0";
  const TensorId pW1 = "pW1";
  const TensorId pW2 = "pW2";

  // We make these named input tensors, so they can be recovered later when
  // testing graph equality.
  const TensorId init7Out = "Init7-out";
  const TensorId init8Out = "Init8-out";

  // Build test (actual) graph.
  // clang-format off
  {
    actual.addInput(pW0, tInfo);
    actual.addInput(pW1, tInfo);
    actual.addInput(pW2, tInfo);

    actual.getTensors().addActGrad(init7Out);
    actual.getTensors().addActGrad(init8Out);

    std::vector<std::unique_ptr<Op>> ops;
    ops.reserve(9);
    ops.push_back(mkAddLhsOp(actual, "AddLhs0"));
    ops.push_back(mkAddLhsOp(actual, "AddLhs1"));
    ops.push_back(mkAddLhsOp(actual, "AddLhs2"));
    ops.push_back(mkAccumulateOp(actual, "Accumulate3"));
    ops.push_back(mkInitOp(actual, tInfo, "Init4"));
    ops.push_back(mkInitOp(actual, tInfo, "Init5"));
    ops.push_back(mkDummyOp(actual, "Dummy6"));
    ops.push_back(mkInitOp(actual, tInfo, "Init7"));
    ops.push_back(mkInitOp(actual, tInfo, "Init8"));

    const auto addLhsOutIdx = AddLhsInplaceOp::getOutIndex();
    const auto addLhs0InIdx = AddLhsInplaceOp::getArg0InIndex();
    const auto addLhs1InIdx = AddLhsInplaceOp::getArg1InIndex();
    const auto initOutIdx   = InitOp::getOutIndex();

    test_graphs::builder::withEdges(actual, ops,
      {
        {ops[4]->id, initOutIdx, ops[0]->id, addLhsOutIdx},
        {ops[0]->id, addLhsOutIdx, ops[1]->id, addLhs0InIdx},
        {ops[1]->id, addLhsOutIdx, ops[2]->id, addLhs0InIdx},
        {ops[2]->id, addLhsOutIdx, ops[3]->id, AccumulateOp::getUpdaterInIndex()},
        {ops[5]->id, initOutIdx, ops[3]->id, AccumulateOp::getVarToUpdateInIndex()},
        {ops[3]->id, AccumulateOp::getUpdatedVarOutIndex(), ops[6]->id, dummyNextInIndex(*ops[6])}
      },
      {
        {pW0, ops[0]->id, addLhs1InIdx},
        {pW1, ops[1]->id, addLhs1InIdx},
        {pW2, ops[2]->id, addLhs1InIdx},
      },
      {
        // We need to give these ops output tensors to make them valid.
        {ops[6]->id, DummyOp::getOutIndex(), {}},
        {ops[7]->id, initOutIdx, {init7Out}},
        {ops[8]->id, initOutIdx, {init8Out}}
      },
      {
        {ops[7]->id, ops[1]->id},
        {ops[7]->id, ops[3]->id},
        {ops[3]->id, ops[8]->id}
      }
    );
  }
  // clang-format on

  // Build expected graph.
  // clang-format off
  {
    expected.addInput(pW0, tInfo);
    expected.addInput(pW1, tInfo);
    expected.addInput(pW2, tInfo);
    
    expected.getTensors().addActGrad(init7Out);
    expected.getTensors().addActGrad(init8Out);

    std::vector<std::unique_ptr<Op>> ops;
    ops.reserve(7);
    ops.push_back(mkAccumulateOp(expected, "Accumulate"));   // idx 0
    ops.push_back(mkAccumulateOp(expected, "Accumulate'"));  // idx 1
    ops.push_back(mkAccumulateOp(expected, "Accumulate''")); // idx 2
    ops.push_back(mkInitOp(expected, tInfo, "Init5"));       // idx 3
    ops.push_back(mkDummyOp(expected, "Dummy6"));            // idx 4
    ops.push_back(mkInitOp(expected, tInfo, "Init7"));       // idx 5
    ops.push_back(mkInitOp(expected, tInfo, "Init8"));       // idx 6

    const auto toUpdateInIdx = AccumulateOp::getVarToUpdateInIndex();
    const auto updaterInIdx  = AccumulateOp::getUpdaterInIndex();
    const auto updatedOutIdx = AccumulateOp::getUpdatedVarOutIndex();
    const auto initOutIdx   = InitOp::getOutIndex();

    test_graphs::builder::withEdges(expected, ops,
        {
          {ops[3]->id, initOutIdx, ops[0]->id, toUpdateInIdx},
          {ops[0]->id, updatedOutIdx, ops[1]->id, toUpdateInIdx},
          {ops[1]->id, updatedOutIdx, ops[2]->id, toUpdateInIdx},
          {ops[2]->id, updatedOutIdx, ops[4]->id, dummyNextInIndex(*ops[4])}
        },
        {
          {pW0,  ops[0]->id, updaterInIdx},
          {pW1,  ops[1]->id, updaterInIdx},
          {pW2,  ops[2]->id, updaterInIdx}
        },
        {
          {ops[4]->id, DummyOp::getOutIndex(), {}},
          {ops[5]->id, initOutIdx, {init7Out}},
          {ops[6]->id, initOutIdx, {init8Out}}
        },
        {
          {ops[5]->id, ops[0]->id},
          {ops[5]->id, ops[1]->id},
          {ops[2]->id, ops[6]->id}
        }
    );
  }
  // clang-format on

  logSchedule(actual, "Actual op schedule before transform");

  // Run test
  InplaceAccumulateGradPartialsIntoOptimizerAccumTensor transform;
  bool modified = transform.apply(actual);

  logSchedule(expected, "Expected op schedule");
  logSchedule(actual, "Actual op schedule after transform");

  BOOST_REQUIRE(modified);
  {
    GraphEqualityTester tester{actual, expected};

    const auto a_Init7 = actual.getTensors().get(init7Out)->getProducerUnsafe();
    const auto e_Init7 =
        expected.getTensors().get(init7Out)->getProducerUnsafe();
    tester.REQUIRE_EQUAL_INIT_OP(a_Init7, e_Init7);

    const auto a_Init8 = actual.getTensors().get(init8Out)->getProducerUnsafe();
    const auto e_Init8 =
        expected.getTensors().get(init8Out)->getProducerUnsafe();
    tester.REQUIRE_EQUAL_INIT_OP(a_Init8, e_Init8);

    const auto a_pW0 = actual.getTensors().get(pW0);
    const auto e_pW0 = expected.getTensors().get(pW0);

    const auto a_Acc0 = a_pW0->consumers.getOps().at(0);
    const auto e_Acc0 = e_pW0->consumers.getOps().at(0);
    tester.REQUIRE_EQUAL_ACCUMULATE_OP(a_Acc0, e_Acc0);

    const auto a_Init5 = a_Acc0->inTensor(AccumulateOp::getVarToUpdateInIndex())
                             ->getProducerUnsafe();
    const auto e_Init5 = e_Acc0->inTensor(AccumulateOp::getVarToUpdateInIndex())
                             ->getProducerUnsafe();
    tester.REQUIRE_EQUAL_INIT_OP(a_Init5, e_Init5);

    const auto a_Acc1 = a_Acc0->outTensor(AccumulateOp::getUpdatedVarOutIndex())
                            ->consumers.getOps()
                            .at(0);
    const auto e_Acc1 = e_Acc0->outTensor(AccumulateOp::getUpdatedVarOutIndex())
                            ->consumers.getOps()
                            .at(0);
    tester.REQUIRE_EQUAL_ACCUMULATE_OP(a_Acc1, e_Acc1);

    const auto a_Acc2 = a_Acc1->outTensor(AccumulateOp::getUpdatedVarOutIndex())
                            ->consumers.getOps()
                            .at(0);
    const auto e_Acc2 = e_Acc1->outTensor(AccumulateOp::getUpdatedVarOutIndex())
                            ->consumers.getOps()
                            .at(0);
    tester.REQUIRE_EQUAL_ACCUMULATE_OP(a_Acc2, e_Acc2);

    const auto a_Dummy6 =
        a_Acc2->outTensor(AccumulateOp::getUpdatedVarOutIndex())
            ->consumers.getOps()
            .at(0);
    const auto e_Dummy6 =
        e_Acc2->outTensor(AccumulateOp::getUpdatedVarOutIndex())
            ->consumers.getOps()
            .at(0);
    tester.REQUIRE_EQUAL_DUMMY_OP(a_Dummy6, e_Dummy6);
  }
}

BOOST_AUTO_TEST_CASE(TestCanHandleWhenOptimizerAccumIsInputAndOutputTensor) {
  /*
    Init2
     |
    dW0             pW0
      \             /
      AddLhsInPlace0
            |
            dW1   accum
              \     |
              Accumulate1
                    |
                  accum1

    Where accum is a graph input and accum1 is a graph output.

    Becomes:

    accum         pW0
        \         /
        Accumulate
            |
          accum1

    Where accum is still a graph input and accum1 is still a graph output.
  */
  const TensorInfo tInfo{DataType::FLOAT, Shape{4, 4, 4, 4}};

  const TensorId pW0   = "pW0";
  const TensorId accum = "accum";

  Ir irActual;
  Graph &actual = irActual.getMainGraph();
  {
    actual.addInput(pW0, tInfo);
    actual.addInput(accum, tInfo);

    const TensorId accum1 = "accum1";
    actual.getTensors().addActGrad(accum1);
    actual.markAsOutput(accum1);

    std::vector<std::unique_ptr<Op>> ops;
    ops.reserve(3);
    ops.push_back(mkAddLhsOp(actual, "AddLhs0"));
    ops.push_back(mkAccumulateOp(actual, "Accumulate1"));
    ops.push_back(mkInitOp(actual, tInfo, "Init2"));

    // clang-format off
    test_graphs::builder::withEdges(actual, ops,
        {
          {ops[2]->id, InitOp::getOutIndex(),          ops[0]->id, AddLhsInplaceOp::getArg0InIndex()},
          {ops[0]->id, AddLhsInplaceOp::getOutIndex(), ops[1]->id, AccumulateOp::getUpdaterInIndex()}
        },
        {
          {pW0,  ops[0]->id, AddLhsInplaceOp::getArg1InIndex()},
          {accum, ops[1]->id, AccumulateOp::getVarToUpdateInIndex()}
        },
        {
          {ops[1]->id, AccumulateOp::getUpdatedVarOutIndex(), {accum1}}
        },
        {}
    );
    // clang-format on
  }

  Ir expectedIr;
  Graph &expected = expectedIr.getMainGraph();
  {
    expected.addInput(pW0, tInfo);
    expected.addInput(accum, tInfo);

    const TensorId accum1 = "accum1";
    expected.getTensors().addActGrad(accum1);
    expected.markAsOutput(accum1);

    std::vector<std::unique_ptr<Op>> ops;
    ops.reserve(1);
    ops.push_back(mkAccumulateOp(expected, "Accumulate"));

    // clang-format off
    test_graphs::builder::withEdges(expected, ops,
        {},
        {
          {accum, ops[0]->id, AccumulateOp::getVarToUpdateInIndex()},
          {pW0,  ops[0]->id, AccumulateOp::getUpdaterInIndex()}
        },
        {
          {ops[0]->id, AccumulateOp::getUpdatedVarOutIndex(), {accum1}}
        },
        {}
    );
    // clang-format on
  }

  InplaceAccumulateGradPartialsIntoOptimizerAccumTensor transform;
  bool modified = transform.apply(actual);

  BOOST_REQUIRE(modified);

  logSchedule(actual, "Actual after transform:");

  {
    GraphEqualityTester tester{actual, expected};

    const auto a_pW0 = actual.getTensors().get(pW0);
    const auto e_pW0 = expected.getTensors().get(pW0);

    const auto a_Acc = a_pW0->consumers.getOps().at(0);
    const auto e_Acc = e_pW0->consumers.getOps().at(0);
    tester.REQUIRE_EQUAL_ACCUMULATE_OP(a_Acc, e_Acc);
  }
}

BOOST_AUTO_TEST_CASE(TestDoesNotModifyGraphWhenTreeAccumIsNotProducedByInitOp) {
  /*
    dW0             pW0
      \             /
      AddLhsInPlace0
            |
            dW1   accum
              \     |
              Accumulate1
                    |
                  accum1

    Unmodified.
  */
  Ir irExpected;
  Graph &expected = irExpected.getMainGraph();

  Ir irActual;
  Graph &actual = irActual.getMainGraph();

  const TensorId pW0   = "pW0";
  const TensorId accum = "accum";

  const auto buildTestGraph = [&pW0, &accum](Graph &graph) {
    const TensorInfo tInfo{DataType::FLOAT, Shape{4, 4, 4, 4}};

    const auto dW0 = graph.addInput(tInfo);
    graph.addInput(pW0, tInfo);
    graph.addInput(accum, tInfo);

    std::vector<std::unique_ptr<Op>> ops;
    ops.reserve(2);
    ops.push_back(mkAddLhsOp(graph, "AddLhs0"));
    ops.push_back(mkAccumulateOp(graph, "Accumulate1"));

    // clang-format off
    test_graphs::builder::withEdges(graph, ops,
        {
          {ops[0]->id, AddLhsInplaceOp::getOutIndex(), ops[1]->id, AccumulateOp::getUpdaterInIndex()}
        },
        {
          {dW0,  ops[0]->id, AddLhsInplaceOp::getArg0InIndex()},
          {pW0,  ops[0]->id, AddLhsInplaceOp::getArg1InIndex()},
          {accum, ops[1]->id, AccumulateOp::getVarToUpdateInIndex()}
        },
        {
          {ops[1]->id, AccumulateOp::getUpdatedVarOutIndex(), {}}
        },
        {}
    );
    // clang-format on
  };

  buildTestGraph(actual);
  buildTestGraph(expected);

  InplaceAccumulateGradPartialsIntoOptimizerAccumTensor transform;
  bool modified = transform.apply(actual);

  BOOST_REQUIRE(!modified);

  {
    GraphEqualityTester tester{actual, expected};

    const auto a_pW0 = actual.getTensors().get(pW0);
    const auto e_pW0 = expected.getTensors().get(pW0);

    const auto a_AddLhs0 = a_pW0->consumers.getOps().at(0);
    const auto e_AddLhs0 = e_pW0->consumers.getOps().at(0);
    tester.REQUIRE_EQUAL_ADDLHS_OP(a_AddLhs0, e_AddLhs0);

    const auto a_accum = actual.getTensors().get(accum);
    const auto e_accum = expected.getTensors().get(accum);

    const auto a_Acc1 = a_accum->consumers.getOps().at(0);
    const auto e_Acc1 = e_accum->consumers.getOps().at(0);
    tester.REQUIRE_EQUAL_ACCUMULATE_OP(a_Acc1, e_Acc1);
  }
}

BOOST_AUTO_TEST_CASE(TestDoesNotModifyGraphWhenInitOpHasNonZeroInitType) {
  /*
    Init2
     |
    dW0             pW0
      \             /
      AddLhsInPlace0
            |
            dW1   accum
              \     |
              Accumulate1
                    |
                  accum1

    Where Init2->getInitType() != InitType::Zero

    Unmodified.
  */
  Ir irExpected;
  Graph &expected = irExpected.getMainGraph();

  Ir irActual;
  Graph &actual = irActual.getMainGraph();

  const TensorId pW0   = "pW0";
  const TensorId accum = "accum";

  const auto mkNonZeroInitOp =
      [](Graph &graph,
         const TensorInfo &tInfo,
         const std::string &name) -> std::unique_ptr<InitOp> {
    return std::make_unique<InitOp>(Onnx::CustomOperators::Init_1,
                                    tInfo,
                                    TensorType::ActGrad,
                                    InitType::NoInit,
                                    Op::Settings{graph, name});
  };

  const auto buildTestGraph = [&mkNonZeroInitOp, &pW0, &accum](Graph &graph) {
    const TensorInfo tInfo{DataType::FLOAT, Shape{4, 4, 4, 4}};

    const auto dW0 = graph.addInput(tInfo);
    graph.addInput(pW0, tInfo);
    graph.addInput(accum, tInfo);

    std::vector<std::unique_ptr<Op>> ops;
    ops.reserve(3);
    ops.push_back(mkAddLhsOp(graph, "AddLhs0"));
    ops.push_back(mkAccumulateOp(graph, "Accumulate1"));
    ops.push_back(mkNonZeroInitOp(graph, tInfo, "Init2"));

    // clang-format off
    test_graphs::builder::withEdges(graph, ops,
        {
          {ops[2]->id, InitOp::getOutIndex(),          ops[0]->id, AddLhsInplaceOp::getArg0InIndex()},
          {ops[0]->id, AddLhsInplaceOp::getOutIndex(), ops[1]->id, AccumulateOp::getUpdaterInIndex()}
        },
        {
          {pW0,  ops[0]->id, AddLhsInplaceOp::getArg1InIndex()},
          {accum, ops[1]->id, AccumulateOp::getVarToUpdateInIndex()}
        },
        {
          {ops[1]->id, AccumulateOp::getUpdatedVarOutIndex(), {}}
        },
        {}
    );
    // clang-format on
  };

  buildTestGraph(actual);
  buildTestGraph(expected);

  InplaceAccumulateGradPartialsIntoOptimizerAccumTensor transform;
  bool modified = transform.apply(actual);

  BOOST_REQUIRE(!modified);

  {
    GraphEqualityTester tester{actual, expected};

    const auto a_pW0 = actual.getTensors().get(pW0);
    const auto e_pW0 = expected.getTensors().get(pW0);

    const auto a_AddLhs0 = a_pW0->consumers.getOps().at(0);
    const auto e_AddLhs0 = e_pW0->consumers.getOps().at(0);
    tester.REQUIRE_EQUAL_ADDLHS_OP(a_AddLhs0, e_AddLhs0);

    const auto a_Init2 = a_AddLhs0->inTensor(AddLhsInplaceOp::getArg0InIndex())
                             ->getProducerUnsafe();
    const auto e_Init2 = e_AddLhs0->inTensor(AddLhsInplaceOp::getArg0InIndex())
                             ->getProducerUnsafe();
    tester.REQUIRE_EQUAL_INIT_OP(a_Init2, e_Init2);

    const auto a_accum = actual.getTensors().get(accum);
    const auto e_accum = expected.getTensors().get(accum);

    const auto a_Acc1 = a_accum->consumers.getOps().at(0);
    const auto e_Acc1 = e_accum->consumers.getOps().at(0);
    tester.REQUIRE_EQUAL_ACCUMULATE_OP(a_Acc1, e_Acc1);
  }
}

/*
  It is only mathematically correct to decompose certain AccumulationTypes.
 */
BOOST_AUTO_TEST_CASE(
    TestDoesNotModifyGraphWhenOptimizerAccumTypeNotDecomposable) {
  /*
    Init2
     |
    dW0             pW0
      \             /
      AddLhsInPlace0
            |
            dW1   accum
              \     |
              Accumulate1
                    |
                  accum1

    Where Accumulate1->getAccumulationType() == AccumulationType::MovingAverage

    Unmodified.
  */
  Ir irExpected;
  Graph &expected = irExpected.getMainGraph();

  Ir irActual;
  Graph &actual = irActual.getMainGraph();

  const TensorId pW0   = "pW0";
  const TensorId accum = "accum";

  const auto mkMovingAvgAccumulateOp =
      [](Graph &graph,
         const std::string &name) -> std::unique_ptr<AccumulateOp> {
    return std::make_unique<AccumulateOp>(AccumulationType::MovingAverage,
                                          OptimizerValue{0.02},
                                          Op::Settings{graph, name});
  };

  const auto buildTestGraph =
      [&mkMovingAvgAccumulateOp, &pW0, &accum](Graph &graph) {
        const TensorInfo tInfo{DataType::FLOAT, Shape{4, 4, 4, 4}};

        const auto dW0 = graph.addInput(tInfo);
        graph.addInput(pW0, tInfo);
        graph.addInput(accum, tInfo);

        std::vector<std::unique_ptr<Op>> ops;
        ops.reserve(3);
        ops.push_back(mkAddLhsOp(graph, "AddLhs0"));
        ops.push_back(mkMovingAvgAccumulateOp(graph, "Accumulate1"));
        ops.push_back(mkInitOp(graph, tInfo, "Init2"));

        // clang-format off
        test_graphs::builder::withEdges(graph, ops,
            {
              {ops[2]->id, InitOp::getOutIndex(),          ops[0]->id, AddLhsInplaceOp::getArg0InIndex()},
              {ops[0]->id, AddLhsInplaceOp::getOutIndex(), ops[1]->id, AccumulateOp::getUpdaterInIndex()}
            },
            {
              {pW0,  ops[0]->id, AddLhsInplaceOp::getArg1InIndex()},
              {accum, ops[1]->id, AccumulateOp::getVarToUpdateInIndex()}
            },
            {
              {ops[1]->id, AccumulateOp::getUpdatedVarOutIndex(), {}}
            },
            {}
        );
        // clang-format on
      };

  buildTestGraph(actual);
  buildTestGraph(expected);

  InplaceAccumulateGradPartialsIntoOptimizerAccumTensor transform;
  bool modified = transform.apply(actual);

  BOOST_REQUIRE(!modified);

  {
    GraphEqualityTester tester{actual, expected};

    const auto a_pW0 = actual.getTensors().get(pW0);
    const auto e_pW0 = expected.getTensors().get(pW0);

    const auto a_AddLhs0 = a_pW0->consumers.getOps().at(0);
    const auto e_AddLhs0 = e_pW0->consumers.getOps().at(0);
    tester.REQUIRE_EQUAL_ADDLHS_OP(a_AddLhs0, e_AddLhs0);

    const auto a_Init2 = a_AddLhs0->inTensor(AddLhsInplaceOp::getArg0InIndex())
                             ->getProducerUnsafe();
    const auto e_Init2 = e_AddLhs0->inTensor(AddLhsInplaceOp::getArg0InIndex())
                             ->getProducerUnsafe();
    tester.REQUIRE_EQUAL_INIT_OP(a_Init2, e_Init2);

    const auto a_accum = actual.getTensors().get(accum);
    const auto e_accum = expected.getTensors().get(accum);

    const auto a_Acc1 = a_accum->consumers.getOps().at(0);
    const auto e_Acc1 = e_accum->consumers.getOps().at(0);
    tester.REQUIRE_EQUAL_ACCUMULATE_OP(a_Acc1, e_Acc1);
  }
}

/*
  When an AccumulateOp has a non-const OptimizerValue as its factor, this
  means there is an extra tensor input for the factor. We test that this tensor
  is correctly connected to every new AccumulateOp.
 */
BOOST_AUTO_TEST_CASE(TestCanHandleAccumulateOpNonConstFactor) {
  /*
    Init2
     |
    dW0             pW0
      \             /
      AddLhsInPlace0
            |
            dW1   accum     factor
              \     |        /
              Accumulate1 --/
                    |
                  accum1

    Becomes:

    accum         pW0    factor
        \         /       /
        Accumulate ------/
            |
          accum1

  */
  const TensorInfo tInfo{DataType::FLOAT, Shape{4, 4, 4, 4}};

  const TensorId pW0   = "pW0";
  const TensorId accum = "accum";

  Ir irActual;
  Graph &actual = irActual.getMainGraph();
  {
    actual.addInput(pW0, tInfo);
    actual.addInput(accum, tInfo);
    const auto factor = actual.addInput(TensorInfo{DataType::FLOAT, {1}});

    std::vector<std::unique_ptr<Op>> ops;
    ops.reserve(3);
    ops.push_back(mkAddLhsOp(actual, "AddLhs0"));
    ops.push_back(mkAccumulateOp(actual, "Accumulate1", false));
    ops.push_back(mkInitOp(actual, tInfo, "Init2"));

    // clang-format off
    test_graphs::builder::withEdges(actual, ops,
        {
          {ops[2]->id, InitOp::getOutIndex(),          ops[0]->id, AddLhsInplaceOp::getArg0InIndex()},
          {ops[0]->id, AddLhsInplaceOp::getOutIndex(), ops[1]->id, AccumulateOp::getUpdaterInIndex()}
        },
        {
          {pW0,    ops[0]->id, AddLhsInplaceOp::getArg1InIndex()},
          {accum,   ops[1]->id, AccumulateOp::getVarToUpdateInIndex()},
          {factor, ops[1]->id, AccumulateOp::getFactorInIndex()}
        },
        {
          {ops[1]->id, AccumulateOp::getUpdatedVarOutIndex(), {}}
        },
        {}
    );
    // clang-format on
  }

  Ir expectedIr;
  Graph &expected = expectedIr.getMainGraph();
  {
    expected.addInput(pW0, tInfo);
    expected.addInput(accum, tInfo);
    const auto factor = expected.addInput(TensorInfo{DataType::FLOAT, {1}});

    std::vector<std::unique_ptr<Op>> ops;
    ops.reserve(1);
    ops.push_back(mkAccumulateOp(expected, "Accumulate", false));

    // clang-format off
    test_graphs::builder::withEdges(expected, ops,
        {},
        {
          {accum,   ops[0]->id, AccumulateOp::getVarToUpdateInIndex()},
          {pW0,    ops[0]->id, AccumulateOp::getUpdaterInIndex()},
          {factor, ops[0]->id, AccumulateOp::getFactorInIndex()}
        },
        {
          {ops[0]->id, AccumulateOp::getUpdatedVarOutIndex(), {}}
        },
        {}
    );
    // clang-format on
  }

  InplaceAccumulateGradPartialsIntoOptimizerAccumTensor transform;
  bool modified = transform.apply(actual);

  BOOST_REQUIRE(modified);

  logSchedule(actual, "Actual after transform:");

  {
    GraphEqualityTester tester{actual, expected};

    const auto a_pW0 = actual.getTensors().get(pW0);
    const auto e_pW0 = expected.getTensors().get(pW0);

    const auto a_Acc = a_pW0->consumers.getOps().at(0);
    const auto e_Acc = e_pW0->consumers.getOps().at(0);
    tester.REQUIRE_EQUAL_ACCUMULATE_OP(a_Acc, e_Acc);
  }
}

BOOST_AUTO_TEST_CASE(
    TestDoesNotModifyGraphWithAccumulateOpsButNoInplaceAdditionTree) {
  /*

       accum0
         |
    Accumulate0 --- t0                      accum1
         |                                    |
      accum0_1 ------ Dummy1 ------ t2 ---- Accumulate2
                       |                      |
                       t1                   accum1_1

    Remains unmodified, as there are no in-place adds going into either of the
    Accumulate ops.
  */
  Ir expectedIr;
  Graph &expected = expectedIr.getMainGraph();

  Ir irActual;
  Graph &actual = irActual.getMainGraph();

  // Save the name of this input so the graphs' ops can be recovered later when
  // we test graph equality.
  const TensorId accum0 = "accum0";

  // clang-format off
  const auto buildTestGraph = [&accum0](Graph &graph) {
      const TensorInfo tInfo{DataType::FLOAT, Shape{4, 4, 4, 4}};

      graph.addInput(accum0, tInfo);
      const auto t0    = graph.addInput(tInfo);
      const auto accum1 = graph.addInput(tInfo);
      const auto t1    = graph.addInput(tInfo);

      std::vector<std::unique_ptr<Op>> ops;
      ops.reserve(3);
      ops.push_back(mkAccumulateOp(graph, "Accumulate0"));
      ops.push_back(mkDummyOp(graph, "Dummy1"));
      ops.push_back(mkAccumulateOp(graph, "Accumulate2"));

      const auto toUpdateInIdx = AccumulateOp::getVarToUpdateInIndex();
      const auto updaterInIdx  = AccumulateOp::getUpdaterInIndex();
      const auto updatedOutIdx = AccumulateOp::getUpdatedVarOutIndex();

      test_graphs::builder::withEdges(graph, ops,
          {
            {ops[0]->id, updatedOutIdx,        ops[1]->id, dummyNextInIndex(*ops[1])},
            {ops[1]->id, AddOp::getOutIndex(), ops[2]->id, updaterInIdx},
          },
          {
            {accum0, ops[0]->id, toUpdateInIdx},
            {t0,    ops[0]->id, updaterInIdx},
            {t1,    ops[1]->id, dummyNextInIndex(*ops[1])},
            {accum1, ops[2]->id, toUpdateInIdx}
          },
          {
            {ops[2]->id, updatedOutIdx, {}}
          },
          {}
      );
  };
  // clang-format on

  buildTestGraph(expected);
  buildTestGraph(actual);

  InplaceAccumulateGradPartialsIntoOptimizerAccumTensor transform;
  bool modified = transform.apply(actual);

  BOOST_REQUIRE(!modified);
  {
    GraphEqualityTester tester{actual, expected};

    const auto a_accum0 = actual.getTensors().get(accum0);
    const auto e_accum0 = expected.getTensors().get(accum0);

    const auto a_Acc0 = a_accum0->consumers.getOps().at(0);
    const auto e_Acc0 = e_accum0->consumers.getOps().at(0);
    tester.REQUIRE_EQUAL_ACCUMULATE_OP(a_Acc0, e_Acc0);

    const auto a_Dummy1 =
        a_Acc0->outTensor(AccumulateOp::getUpdatedVarOutIndex())
            ->consumers.getOps()
            .at(0);
    const auto e_Dummy1 =
        e_Acc0->outTensor(AccumulateOp::getUpdatedVarOutIndex())
            ->consumers.getOps()
            .at(0);
    tester.REQUIRE_EQUAL_DUMMY_OP(a_Dummy1, e_Dummy1);

    const auto a_Acc2 =
        a_Dummy1->outTensor(DummyOp::getOutIndex())->consumers.getOps().at(0);
    const auto e_Acc2 =
        e_Dummy1->outTensor(DummyOp::getOutIndex())->consumers.getOps().at(0);
    tester.REQUIRE_EQUAL_ACCUMULATE_OP(a_Acc2, e_Acc2);
  }
}

/*
  In a graph where there is a transformable inplace addition tree, but the
  addition tree's accumulation tensor has other consumers, we cannot optimise it
  away.
*/
BOOST_AUTO_TEST_CASE(
    TestDoesNotModifyGraphWhenAddTreeAccumTensorHasOtherConsumers) {
  /*
    Init3
     |
    dW0 ---             pW0
     |     \            /
     |     AddLhsInplace0
     |           |
     |           dW1   accum
     |             \     |
    Dummy1         Accumulate2
     |                   |
    dOut               accum1

    Unmodified, as cannot lose side-effects on dW0 if it's used outside of this
    addition tree.
  */

  Ir irExpected;
  Graph &expected = irExpected.getMainGraph();
  Ir irActual;
  Graph &actual = irActual.getMainGraph();

  const TensorId accum = "accum";
  const TensorId dOut  = "dOut";

  const auto buildTestGraph = [&accum, &dOut](Graph &graph) {
    // clang-format off
      const TensorInfo tInfo{DataType::FLOAT, Shape{4, 4, 4, 4}};

      const auto pw0  = graph.addInput(tInfo);
      graph.addInput(accum, tInfo);
      graph.getTensors().addActGrad(dOut);

      std::vector<std::unique_ptr<Op>> ops;
      ops.reserve(4);
      ops.push_back(mkAddLhsOp(graph, "AddLhs0"));
      ops.push_back(mkDummyOp(graph, "Dummy1"));
      ops.push_back(mkAccumulateOp(graph, "Accumulate2"));
      ops.push_back(mkInitOp(graph, tInfo, "Init3"));

      test_graphs::builder::withEdges(graph, ops, 
          {
            {ops[3]->id, InitOp::getOutIndex(), ops[0]->id, AddLhsInplaceOp::getArg0InIndex()},
            {ops[3]->id, InitOp::getOutIndex(), ops[1]->id, dummyNextInIndex(*ops[1])},
            {ops[0]->id, AddLhsInplaceOp::getOutIndex(), ops[2]->id, AccumulateOp::getUpdaterInIndex()}
          },
          {
            {pw0,  ops[0]->id, AddLhsInplaceOp::getArg1InIndex()},
            {accum, ops[2]->id, AccumulateOp::getVarToUpdateInIndex()}
          },
          {
            {ops[1]->id, DummyOp::getOutIndex(), {dOut}},
            {ops[2]->id, AccumulateOp::getUpdatedVarOutIndex(), {}}
          },
          {}
      );

    // clang-format on
  };

  buildTestGraph(expected);
  buildTestGraph(actual);

  InplaceAccumulateGradPartialsIntoOptimizerAccumTensor transform;
  bool modified = transform.apply(actual);

  BOOST_REQUIRE(!modified);
  {
    GraphEqualityTester tester{actual, expected};

    const auto a_accum = actual.getTensors().get(accum);
    const auto e_accum = expected.getTensors().get(accum);

    const auto a_Acc2 = a_accum->consumers.getOps().at(0);
    const auto e_Acc2 = e_accum->consumers.getOps().at(0);
    tester.REQUIRE_EQUAL_ACCUMULATE_OP(a_Acc2, e_Acc2);

    const auto a_AddLhs0 = a_Acc2->inTensor(AccumulateOp::getUpdaterInIndex())
                               ->getProducerUnsafe();
    const auto e_AddLhs0 = e_Acc2->inTensor(AccumulateOp::getUpdaterInIndex())
                               ->getProducerUnsafe();
    tester.REQUIRE_EQUAL_ADDLHS_OP(a_AddLhs0, e_AddLhs0);

    const auto a_Init3 = a_AddLhs0->inTensor(AddLhsInplaceOp::getArg0InIndex())
                             ->getProducerUnsafe();
    const auto e_Init3 = e_AddLhs0->inTensor(AddLhsInplaceOp::getArg0InIndex())
                             ->getProducerUnsafe();
    tester.REQUIRE_EQUAL_INIT_OP(a_Init3, e_Init3);

    const auto a_dOut = actual.getTensors().get(dOut);
    const auto e_dOut = expected.getTensors().get(dOut);

    const auto a_Dummy1 = a_dOut->getProducerUnsafe();
    const auto e_Dummy1 = e_dOut->getProducerUnsafe();
    tester.REQUIRE_EQUAL_DUMMY_OP(a_Dummy1, e_Dummy1);
  }
}

namespace {

std::unique_ptr<AccumulateOp>
mkAccumulateOp(Graph &graph, const std::string &name, const bool isConst) {
  return std::make_unique<AccumulateOp>(AccumulationType::DampenedAdd,
                                        OptimizerValue{0.02, isConst},
                                        Op::Settings{graph, name});
}

std::unique_ptr<InitOp>
mkInitOp(Graph &graph, const TensorInfo &tInfo, const std::string &name) {
  return std::make_unique<InitOp>(Onnx::CustomOperators::Init_1,
                                  tInfo,
                                  TensorType::ActGrad,
                                  InitType::Zero,
                                  Op::Settings{graph, name});
}

std::unique_ptr<DummyOp> mkDummyOp(Graph &graph, const std::string &name) {
  return std::make_unique<DummyOp>(graph, Op::Settings(graph, name));
}

InIndex dummyNextInIndex(Op &dummy) {
  return static_cast<DummyOp &>(dummy).getNextInIndex();
}

std::unique_ptr<AddLhsInplaceOp> mkAddLhsOp(Graph &graph,
                                            const std::string &name) {
  return std::make_unique<AddLhsInplaceOp>(Op::Settings{graph, "AddLhs0"});
}

template <bool ShouldLog>
void logSchedule(const Graph &graph, const std::string &title) {
  if (!ShouldLog) {
    return;
  }

  std::cout << title << ":" << std::endl;

  for (const auto op : graph.getOpSchedule({}, RequireOptimalSchedule::No)) {
    std::cout << "-> " << op->getName() << ": (" << op->id << ")" << std::endl
              << "      " << op->debugName() << std::endl;
  }
}

} // namespace
