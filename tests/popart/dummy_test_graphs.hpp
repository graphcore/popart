#include <popart/error.hpp>
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

#include <popart/op/add.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/conv.hpp>
#include <popart/op/lrn.hpp>
#include <popart/op/reducesumsquare.hpp>
#include <popart/op/relu.hpp>

#include <limits>
#include <map>
#include <memory>
#include <string>
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
void initSingleOpTestGraph(popart::Graph &graph) {
  using namespace popart;

  // Make an empty AddOp.
  Op::Settings settings{graph, ""};
  auto addOp = std::make_unique<AddOp>(Onnx::Operators::Add_7, settings);
  addOp->id  = 0;

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
void initDiamondTestGraph(popart::Graph &graph) {
  using namespace popart;

  // Batch size and number channels for tensor we will create.
  constexpr int64_t B  = 8;
  constexpr int64_t C  = 4;
  constexpr int64_t Nx = 4;
  constexpr int64_t Ny = 4;

  /********* add0 ********/

  auto addOp_0 =
      std::make_unique<AddOp>(Onnx::Operators::Add_7, Op::Settings{graph, ""});
  addOp_0->id = 0;

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

  auto convOp_2 = std::make_unique<ConvOp>(Onnx::Operators::Conv_11,
                                           Op::Settings{graph, ""},
                                           std::vector<int64_t>{}, // strides
                                           std::vector<int64_t>{}, // pads
                                           std::vector<int64_t>{}, // dilations
                                           convOp_2_group,
                                           AutoPad::VALID,
                                           MultiConvOptions{{}, {}});
  convOp_2->id  = 2;

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
  concatOp_4->createAndConnectOutTensor(ConcatOp::getOutIndex(), concat_4_out);

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
  BOOST_REQUIRE_NO_THROW(rssGradOp_6->isConvertibleTo<ReduceSumSquareGradOp>());

  rssGradOp_6->id = 6;

  // Looking at `ReduceSumSquareOp::gradInputInfo()`, we see what the inputs
  // should be and manually connect them.

  // Input 1: Gradient of rss output (which is the loss) wrt loss. This is 1.

  TensorId rssGrad_6_in_grad_of_op_out{"rssGrad-6-in-grad-of-op-out"};

  std::vector<float> lossGradData{B, 1.0f};
  graph.getTensors().addConstInit(rssGrad_6_in_grad_of_op_out,
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

class DummyOp : public popart::Op {
public:
  static inline const popart::OperatorIdentifier OnnxId = {
      "dummy_domain",
      "dummy_type",
      1, // version
      popart::NumInputs{0, std::numeric_limits<int>::max()},
      1 // num outputs
  };

  DummyOp(popart::Graph &graph)
      : Op(OnnxId,
           popart::Op::Settings{graph, "test_graph::DummyOp::Settings"}) {}

  popart::InIndex getNextInIndex() { return nextInIndex++; }
  static popart::OutIndex getOutIndex() { return 0; }

  void setup() final {
    if (nextInIndex == 0) {
      throw popart::error(
          "DummyOp::setup(): DummyOp requires at least one input.");
    }

    outInfo(getOutIndex()) = inInfo(0);
  }

  float getSubgraphValue() const override { return getLowSubgraphValue(); }

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<DummyOp>(*this);
  }

private:
  popart::InIndex nextInIndex{0};
};

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
               const std::multimap<popart::OpId, popart::OpId> &topoCons) {
  using namespace popart;

  const auto nOps = edges.size();

  /* 1. Create all the ops, assign their opIds, and move into graph. */

  for (auto opId = 0; opId < nOps; opId++) {
    auto op = std::make_unique<DummyOp>(graph);
    op->id  = opId;
    graph.moveIntoGraph(std::move(op));
  }

  /*
    2. Construct the graph.

    To do this, we essentially do a simple scheduling pass (kahn's algo) over
    the edges, constructing the tensors and connecting them to the ops as we go
    along.
  */

  // First, some helpers.

  const TensorInfo defaultTensorInfo{"FLOAT", std::vector<int64_t>{4, 4, 4, 4}};

  const auto connectInTensor = [&graph](const OpId opId, const TensorId tId) {
    auto *op = dynamic_cast<DummyOp *>(graph.getOp(opId));
    op->connectInTensor(op->getNextInIndex(), tId);
  };

  const auto createAndConnectOutTensorAndConnectToConsumers =
      [&graph, &edges, connectInTensor](const OpId opId) {
        auto *op = dynamic_cast<DummyOp *>(graph.getOp(opId));

        const auto tOutId = TensorId{std::to_string(opId) + "-output"};

        op->createAndConnectOutTensor(DummyOp::getOutIndex(), tOutId);
        op->setup();

        for (const auto out : edges[opId]) {
          connectInTensor(out, tOutId);
        }
      };

  // Initialise the scheduling pass.

  std::vector<OpId> outstanding(nOps, 0);
  std::vector<OpId> ready;

  // Compute, for each op, how many incoming edges (dependencies) it has.
  for (const auto consumersOfOp : edges) {
    for (const auto j : consumersOfOp) {
      outstanding[j]++;
    }
  }

  // Mark those ops with 0 dependencies as ready.
  // Note, in our scheduling pass, when an op gets scheduled, we create its
  // output tensor and hook it up as an input to the op's consumers. We do not
  // create the inputs; the invariant is that the input tensors will already
  // exist, created when their producer was scheduled. Thus, as part of
  // initialising the scheduling pass, we need to create the input ops' input
  // tensors.
  for (OpId i = 0; i < nOps; i++) {
    if (outstanding[i] == 0) {
      ready.push_back(i);

      const TensorId tId{std::to_string(i) + "-input"};
      graph.addInput(tId, defaultTensorInfo);
      connectInTensor(i, tId);
    }
  }

  int nScheduled = 0;

  while (!ready.empty()) {
    const OpId i = ready.back();
    ready.pop_back();

    nScheduled++;

    createAndConnectOutTensorAndConnectToConsumers(i);

    for (const auto j : edges[i]) {
      --outstanding[j];

      if (outstanding[j] == 0) {
        ready.push_back(j);
      }
    }
  }

  // Done!

  if (nScheduled != static_cast<int>(nOps)) {
    throw error("test_graphs::withEdges: Proposed graph is not schedulable.");
  }

  /* 3. Now we've built the graph, add all the topo cons. */

  for (const auto tc : topoCons) {
    graph.topoCons->insert(graph.getOp(tc.first), graph.getOp(tc.second));
  }
}

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
void initMultiInputMultiOutputComplexTestCase(popart::Graph &graph) {
  using namespace popart;

  const auto edges = std::vector<std::vector<OpId>>{
      {1, 2, 3, 4, 13, 14}, // 0
      {2, 4},               // 1
      {},                   // 2
      {5},                  // 3
      {5, 6, 7},            // 4
      {6},                  // 5
      {7, 8},               // 6
      {10, 11},             // 7
      {9},                  // 8
      {10},                 // 9
      {12},                 // 10
      {12},                 // 11
      {},                   // 12
      {12},                 // 13
      {},                   // 14
      {5},                  // 15
      {17, 18},             // 16
      {18},                 // 17
      {12}                  // 18
  };

  const auto topoCons =
      std::multimap<OpId, OpId>{{13, 8}, {17, 7}, {15, 6}, {7, 13}};

  return withEdges(graph, edges, topoCons);
}

} // namespace test_graphs
