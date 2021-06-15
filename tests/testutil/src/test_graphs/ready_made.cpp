#include <testutil/test_graphs/ready_made.hpp>

#include <testutil/test_graphs/dummy_builder.hpp>

#include <filereader.hpp>
#include <popart/error.hpp>
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

#include <memory>
#include <string>
#include <vector>

using namespace popart;

namespace test_graphs {
namespace ready_made {

void initSingleOp(Graph &graph) {
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

void initDiamond(Graph &graph) {
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

  try {

    // `.at` throws.
    rssGradOp_6 = std::move(graph.getOp(5)->getGradOps().at(0));

  } catch (std::out_of_range const &ex) {
    throw internal_error("rss5 should have one grad op at index 0.");
  }

  if (!rssGradOp_6->isConvertibleTo<ReduceSumSquareGradOp>()) {
    throw internal_error("rss5's grad op is not a ReduceSumSquareGradOp.");
  }

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

void initComplexMultiInputMultiOutput(Graph &graph) {
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

  return dummy_builder::withEdges(graph, edges, topoCons);
}

} // namespace ready_made
} // namespace test_graphs
