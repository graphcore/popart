// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE GraphEdgeMapTest
#include <boost/test/unit_test.hpp>

#include <map>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include <popart/filereader.hpp>
#include <popart/graph.hpp>
#include <popart/graphid.hpp>
#include <popart/ir.hpp>
#include <popart/opidentifier.hpp>
#include <popart/scheduler.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/topocons.hpp>

// Used in test graphs
#include <popart/op/add.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/conv.hpp>
#include <popart/op/lrn.hpp>
#include <popart/op/reducesumsquare.hpp>
#include <popart/op/relu.hpp>

using namespace popart;

using EdgeMap = std::map<OpId, std::unordered_set<OpId>>;

namespace {

std::string mkFailureMsg(const EdgeMap &expected, const EdgeMap &actual) {

  const auto append = [](std::ostringstream &oss, const EdgeMap &edgeMap) {
    oss << "[ ";

    for (const auto &consumersOfOpId : edgeMap) {
      oss << "{ " << consumersOfOpId.first << ": [ ";
      for (const auto &opid : consumersOfOpId.second) {
        oss << opid << " ";
      }
      oss << "] }, ";
    }

    oss << " ]";
  };

  std::ostringstream oss;

  oss << "critical check expectedMap == actualMap has failed  ";
  append(oss, expected);
  oss << "  !=  ";
  append(oss, actual);

  return oss.str();
}

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

protected:
  EdgeMap::value_type mp(const EdgeMap::value_type::first_type &&a,
                         const EdgeMap::value_type::second_type &&b) {
    return std::make_pair(std::move(a), std::move(b));
  }
};

/**
 * add
 *
 * With no dependencies.
 */
struct SingleOpTestCase : EdgeMapTestCase<SingleOpTestCase> {

  void initTestGraph_() {
    // Make an empty AddOp.
    Op::Settings settings{graph, ""};
    auto addOp = std::make_unique<AddOp>(Onnx::Operators::Add_7, settings);

    // We will manually overwrite the OpIds of the Ops we create in order to
    // fix the ordering of the ops in the map returned by `g.getOps`. See
    // [comment-0]. This field will be used for those OpIds.
    addOp->id = 0;

    TensorId t0{"add-in-0"};
    TensorId t1{"add-in-1"};
    TensorId t2{"add-out"};

    // Make the input tensors and mark them as inputs/outputs of the graph.
    // This is also the point at which we give the inputs their TensorInfo.
    // TensorInfo of all other tensors will be inferred from the op that
    // produces them.
    TensorInfo tInfo{"FLOAT16", std::vector<int64_t>{4, 4}};
    graph.addInput(t0, tInfo);
    graph.addInput(t1, tInfo);

    // Make the output tensors and connect the input/output tensors to the
    // AddOp at the correct indices.
    addOp->connectInTensor(AddOp::getArg0InIndex(), t0);
    addOp->connectInTensor(AddOp::getArg1InIndex(), t1);
    addOp->createAndConnectOutTensor(AddOp::getOutIndex(), t2);

    // Sets the TensorInfo of all outputs.
    addOp->setup();

    // Mark addOp's output tensor as an output of the graph.
    graph.markAsOutput(t2);

    // Move the AddOp into the graph.
    graph.moveIntoGraph(std::move(addOp));
  }

  EdgeMap mkExpectedEdges_() {
    return {mp(0, {})}; // add0 is OpId 0; has no dependents.
  }
};

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
struct DiamondTestCase : EdgeMapTestCase<DiamondTestCase> {

  void initTestGraph_() {
    // Do not use this graph as a reference for creating graphs/certain ops.
    // Some of the dimensions, TensorTypes etc. may be malformed. The point here
    // is just to have a graph with topological dependencies in it.

    // Batch size and number channels for tensor we will create.
    constexpr int64_t B  = 8;
    constexpr int64_t C  = 4;
    constexpr int64_t Nx = 4;
    constexpr int64_t Ny = 4;

    /********* add0 ********/

    auto addOp_0 = std::make_unique<AddOp>(Onnx::Operators::Add_7,
                                           Op::Settings{graph, ""});
    addOp_0->id  = 0;

    TensorId add_0_in_0{"add-0-in-0"};
    TensorId add_0_in_1{"add-0-in-1"};

    TensorInfo add0_in_tInfo{"FLOAT", std::vector<int64_t>{B, C, Nx, Ny}};
    graph.addInput(add_0_in_0, add0_in_tInfo);
    graph.addInput(add_0_in_1, add0_in_tInfo);

    addOp_0->connectInTensor(AddOp::getArg0InIndex(), add_0_in_0);
    addOp_0->connectInTensor(AddOp::getArg1InIndex(), add_0_in_1);

    TensorId add_0_out{"add-0-out"};

    addOp_0->createAndConnectOutTensor(AddOp::getOutIndex(), add_0_out);

    addOp_0->setup();

    graph.moveIntoGraph(std::move(addOp_0));

    /********* relu1 ********/

    auto reluOp_1 = std::make_unique<ReluOp>(Onnx::Operators::Relu_6,
                                             Op::Settings{graph, ""});
    reluOp_1->id  = 1;

    reluOp_1->connectInTensor(ReluOp::getInIndex(), add_0_out);

    TensorId relu_1_out{"relu-1-out"};
    reluOp_1->createAndConnectOutTensor(ReluOp::getOutIndex(), relu_1_out);

    reluOp_1->setup();

    graph.moveIntoGraph(std::move(reluOp_1));

    /********* conv2 ********/

    constexpr int64_t convOp_2_group = 1;

    auto convOp_2 =
        std::make_unique<ConvOp>(Onnx::Operators::Conv_11,
                                 Op::Settings{graph, ""},
                                 std::vector<int64_t>{}, // strides
                                 std::vector<int64_t>{}, // pads
                                 std::vector<int64_t>{}, // dilations
                                 convOp_2_group,
                                 AutoPad::VALID,
                                 MultiConvOptions{{}, {}});
    convOp_2->id = 2;

    convOp_2->connectInTensor(ConvOp::getDataInIndex(), add_0_out);

    TensorId convOp_2_in_W{"conv-2-in-W"};

    // Create C feature maps
    std::vector<float> convOp_2_in_W_data{C * C * 1 * 1};
    graph.getTensors().addVarInit(
        convOp_2_in_W,
        TensorInfo{"FLOAT", std::vector<int64_t>{C, C / convOp_2_group, 1, 1}},
        convOp_2_in_W_data.data());

    convOp_2->connectInTensor(ConvOp::getWeightsInIndex(), convOp_2_in_W);

    TensorId conv_2_out{"conv-2-out"};
    convOp_2->createAndConnectOutTensor(ConvOp::getOutIndex(), conv_2_out);

    convOp_2->setup();

    graph.moveIntoGraph(std::move(convOp_2));

    /********* LRN3 ********/

    auto lrnOp_3 = std::make_unique<LRNOp>(Onnx::Operators::LRN_1,
                                           0.001, // alpha
                                           0.75,  // beta,
                                           1.0,   // bias
                                           4,     // size
                                           Op::Settings{graph, ""});
    lrnOp_3->id  = 3;

    lrnOp_3->connectInTensor(LRNOp::getInIndex(), conv_2_out);

    TensorId lrn_3_out{"lrn-3-out"};
    lrnOp_3->createAndConnectOutTensor(LRNOp::getOutIndex(), lrn_3_out);

    lrnOp_3->setup();

    graph.moveIntoGraph(std::move(lrnOp_3));

    /********* concat4 ********/

    auto concatOp_4 =
        std::make_unique<ConcatOp>(Onnx::Operators::Concat_11,
                                   1, // Concat on channels dimension
                                   Op::Settings{graph, ""});
    concatOp_4->id = 4;

    concatOp_4->connectInTensor(ConcatOp::getInIndex(0), relu_1_out);
    concatOp_4->connectInTensor(ConcatOp::getInIndex(1), lrn_3_out);

    TensorId concat_4_out{"concat-4-out"};
    concatOp_4->createAndConnectOutTensor(ConcatOp::getOutIndex(),
                                          concat_4_out);

    concatOp_4->setup();

    graph.moveIntoGraph(std::move(concatOp_4));

    /********* rss5 ********/

    auto rssOp_5 = std::make_unique<ReduceSumSquareOp>(
        Onnx::Operators::ReduceSumSquare_11,
        nonstd::optional<std::vector<int64_t>>{{1, 2, 3}}, // axes
        0,                                                 // keepdims
        Op::Settings{graph, ""});
    rssOp_5->id = 5;

    rssOp_5->connectInTensor(ReduceSumSquareOp::getInIndex(), concat_4_out);

    TensorId rss_5_out{"rss-5-out"};
    rssOp_5->createAndConnectOutTensor(ReduceSumSquareOp::getOutIndex(),
                                       rss_5_out);
    graph.setLoss(rss_5_out);

    rssOp_5->setup();

    graph.moveIntoGraph(std::move(rssOp_5));

    /********* rssgrad6 ********/

    std::unique_ptr<Op> rssGradOp_6;

    // Internal error (not test failure): ReduceSumSquareOp has one grad op at
    // index 0 that is a ReduceSumSquareGradOp.
    BOOST_REQUIRE_NO_THROW(
        (rssGradOp_6 = std::move(graph.getOp(5)->getGradOps().at(0))));
    BOOST_REQUIRE_NO_THROW(dynamic_cast<ReduceSumSquareGradOp &>(*rssGradOp_6));

    rssGradOp_6->id = 6;

    // Looking at `ReduceSumSquareOp::gradInputInfo()`, we see what the inputs
    // should be and manually connect them.

    // Input 1: Gradient of rss output (which is the loss) wrt loss. This is 1.

    TensorId rssGrad_6_in_grad_of_op_out{"rssGrad-6-in-grad-of-op-out"};

    std::vector<float> lossGradData{B, 1.0f};
    graph.getTensors().addConstInit(
        rssGrad_6_in_grad_of_op_out,
        TensorInfo{"FLOAT", std::vector<int64_t>{B}},
        lossGradData.data());

    rssGradOp_6->connectInTensor(ReduceSumSquareGradOp::getInIndex(),
                                 rssGrad_6_in_grad_of_op_out);

    // Input 2. rss input (which is concat output).

    rssGradOp_6->connectInTensor(
        ReduceSumSquareGradOp::getFwdInInIndex(),
        graph.getOp(5)->input->tensor(ReduceSumSquareOp::getInIndex())->id);

    TensorId rssGrad_6_out{"rssGrad-6-out"};
    rssGradOp_6->createAndConnectOutTensor(ReduceSumSquareGradOp::getOutIndex(),
                                           rssGrad_6_out);
    graph.markAsOutput(rssGrad_6_out);

    rssGradOp_6->setup();

    graph.moveIntoGraph(std::move(rssGradOp_6));

    /********* TopoCons ********/

    auto &topoCons = graph.topoCons;

    topoCons->insert(graph.getOp(0), graph.getOp(3));
    topoCons->insert(graph.getOp(1), graph.getOp(3));
    topoCons->insert(graph.getOp(2), graph.getOp(4));
    topoCons->insert(graph.getOp(2), graph.getOp(6));
    topoCons->insert(graph.getOp(5), graph.getOp(6));
  }

  EdgeMap mkExpectedEdges_() {
    return {
        mp(0, {1, 2, 3}), // add0
        mp(1, {3, 4}),    // relu1
        mp(2, {3, 4, 6}), // conv2
        mp(3, {4}),       // LRN3
        mp(4, {5, 6}),    // concat4 (no dependents)
        mp(5, {6}),       // nll5
        mp(6, {})         // nllgrad6
    };
  }
};

} // namespace

using TestCaseTypes = std::tuple<SingleOpTestCase, DiamondTestCase>;

BOOST_AUTO_TEST_CASE_TEMPLATE(GraphEdgeMapTest, TestCase, TestCaseTypes) {
  TestCase tc;

  tc.initTestGraph();
  const auto expectedMap = tc.mkExpectedEdges();
  const auto actualMap   = tc.graph.getEdgeMap();

  // NB: BOOST_REQUIRE_EQUAL can't handle printing a map<OpId, unordered_set>,
  // so we construct a nice error message manually.
  if (expectedMap != actualMap) {
    BOOST_FAIL(mkFailureMsg(expectedMap, actualMap));
  }
}
