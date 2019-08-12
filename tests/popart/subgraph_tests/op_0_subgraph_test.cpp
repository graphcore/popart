#define BOOST_TEST_MODULE Op0SubgraphTest

// tests using the Op class as the subgraph template parameter

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <vector>
#include <popart/builder.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/op.hpp>
#include <popart/subgraph/outliner.hpp>
#include <popart/tensornames.hpp>

#include <popart/filereader.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensordata.hpp>

namespace {

using namespace fwtools::subgraph;
using namespace popart;

std::vector<std::set<Match>> getSets(const std::vector<Op *> &sched,
                                     const std::vector<Match> &expected_matches,
                                     float threshold,
                                     int algo) {

  // get the matches
  //
  std::vector<Match> matches;
  if (algo == 0) {
    matches = getRinseMatches<Op>(sched, threshold, OutlinerAlgorithm::ALGO0);
  } else if (algo == 1) {
    matches = getRinseMatches<Op>(sched, threshold, OutlinerAlgorithm::ALGO1);
  } else {
    throw std::runtime_error("invalid algo number");
  }

  std::set<Match> s_expected_matches;
  std::set<Match> s_matches;

  std::stringstream ss;
  ss << "\nExpected matches:";
  for (auto x : expected_matches) {
    ss << "\n" << x;
    setValue(x, sched);
    s_expected_matches.emplace(x);
  }

  ss << "\nComputed matches using subgraph outlining algo " << algo << ":";
  for (auto x : matches) {
    ss << "\n" << x;
    setValue(x, sched);
    s_matches.insert(x);
  }

  popart::logging::debug(ss.str());

  std::vector<std::set<Match>> sets = {s_expected_matches, s_matches};
  return sets;
}

} // namespace

BOOST_AUTO_TEST_CASE(Op0_Subgraph) {

  using namespace fwtools::subgraph;
  using namespace popart;

  auto test = [](const std::vector<Op *> &sched,
                 const std::vector<Match> &expected_matches,
                 float threshold,
                 int algo) {
    auto sets = getSets(sched, expected_matches, threshold, algo);
    std::cout << "testing with threshold = " << threshold
              << " and algo = " << algo << std::endl;
    BOOST_CHECK(sets[0] == sets[1]);
  };

  // ----------------------------------------------------
  auto testWithTrain = [&test](bool train,
                               float threshold,
                               std::vector<Match> expected_matches_algo0,
                               std::vector<Match> expected_matches_algo1) {
    auto builder = Builder::create();
    auto aiOnnx  = builder->aiOnnxOpset9();
    TensorInfo info0{"FLOAT", std::vector<int64_t>{4, 4}};
    float weight_vals[4 * 4]  = {1.0f};
    ConstVoidData weight_data = {weight_vals, info0};

    auto in0 = builder->addInputTensor(info0);
    std::vector<TensorId> weightIds;
    std::vector<TensorId> mmOutIds;
    std::vector<TensorId> reluIds{in0};
    int reps = 4;
    for (int i = 0; i < reps; ++i) {
      weightIds.push_back(builder->addInitializedInputTensor(weight_data));
    }
    for (int i = 0; i < reps; ++i) {
      mmOutIds.push_back(aiOnnx.matmul({reluIds.back(), weightIds[i]}));
      reluIds.push_back(aiOnnx.relu({mmOutIds.back()}));
    }
    auto out = aiOnnx.reducesum({reluIds.back()});
    builder->addOutputTensor(out);

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);
    std::unique_ptr<Optimizer> optimizer;
    std::vector<std::unique_ptr<L1Loss>> up_losses;
    std::vector<Loss *> losses{};
    auto dataFlow  = DataFlow(1, {{out, AnchorReturnType("ALL")}});
    auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

    if (train) {
      optimizer.reset(new ConstSGD(0.01));
      up_losses.push_back(std::unique_ptr<L1Loss>(
          new L1Loss(out, "l1LossVal", 0.1, ReductionType::SUM)));
      losses = {up_losses[0].get()};
    }

    auto opts = SessionOptions();
    // This test tests the functionality of fwtools::subgraph::getRinseMatches,
    // not the actual outlining of the Ir
    opts.enableOutlining   = false;
    opts.autoRecomputation = RecomputationType::None;

    Ir ir;
    opts.mergeVarUpdate = MergeVarUpdateType::None;
    ir.prepare({modelProto,
                InputShapeInfo(),
                dataFlow,
                losses,
                optimizer.get(),
                *cpuDevice,
                opts,
                Patterns(PatternsLevel::ALL).enableInPlace(false)});

    auto sched = ir.getOpSchedule({});

    int i = 0;
    for (auto op : sched) {
      std::cout << i++ << " " << op->opid.type << std::endl;
    }

    test(sched, expected_matches_algo0, threshold, 0);
    test(sched, expected_matches_algo1, threshold, 1);
  };

  /*
  0  Reshape  | [@  [*
  1  MatMul   |  @   *
  2  Reshape  |  @   *
  3  Relu     |  @   *]
  4  Reshape  |  @  [*
  5  MatMul   |  @   *
  6  Reshape  |  @   *
  7  Relu     |  @]  *]
  8  Reshape  | [@  [*
  9  MatMul   |  @   *
  10 Reshape  |  @   *
  11 Relu     |  @   *]
  12 Reshape  |  @  [*
  13 MatMul   |  @   *
  14 Reshape  |  @   *
  15 Relu     |  @]  *]
  16 Identity |
  */

  std::vector<Match> expected_test_matches = {{{0, 8}, 8}, {{0, 4, 8, 12}, 4}};

  /*
    0 Reshape             |                          [1   2
    1 Reshape             |                           1]  2
    2 Reshape             |                          [1   2
    3 Reshape             |                           1]  2
    4 Reshape             |    [&    [%
    5 MatMul              |     &     %           @
    6 Reshape             |     &     %
    7 Relu                |     &     %]
    8 Reshape             |     &    [%                   2
    9 MatMul              |     &     %           @
    10 Reshape            |     &     %
    11 Relu               |     &]    %]
    12 Reshape            |    [&    [%                   2
    13 MatMul             |     &     %           @
    14 Reshape            |     &     %
    15 Relu               |     &     %]
    16 Reshape            |     &    [%                   2
    17 MatMul             |     &     %           @
    18 Reshape            |     &     %
    19 Relu               |     &]    %]
    20 Identity           |
    21 L1Grad             |
    22 ReluGrad           | [*                                        5
    23 ReshapeGrad        |  *                                    4
    24 Transpose          |  *                                3
    25 MatMul             |  *    [£    [!   [#   @
    26 ReshapeGrad        |  *     £     !    #]
    27 ConstSGDVarUpdate  |  *     £     !]
    28 Transpose          |  *     £
    29 MatMul             |  *     £         [#   @
    30 ReshapeGrad        |  *     £]         #]
    31 ReluGrad           |  *                                         5
    32 ReshapeGrad        |  *                                    4
    33 Transpose          |  *]                               3
    34 MatMul             |       [£    [!   [#   @
    35 ReshapeGrad        |        £     !    #]
    36 ConstSGDVarUpdate  |        £     !]
    37 Transpose          |        £                          3
    38 MatMul             |        £         [#   @
    39 ReshapeGrad        |        £]         #]
    40 ReluGrad           | [*                                         5
    41 ReshapeGrad        |  *                                    4
    42 Transpose          |  *    [£                          3
    43 MatMul             |  *     £    [!   [#   @
    44 ReshapeGrad        |  *     £     !    #]
    45 ConstSGDVarUpdate  |  *     £     !]
    46 Transpose          |  *     £                          3
    47 MatMul             |  *     £]        [#   @
    48 ReshapeGrad        |  *                #]                       5
    49 ReluGrad           |  *
    50 ReshapeGrad        |  *                                    4
    51 Transpose          |  *]                               3
    52 MatMul             |             [!   [#   @
    53 ReshapeGrad        |              !    #]
    54 ConstSGDVarUpdate  |              !]
  */

  std::vector<Match> expected_train_matches_algo0 = {
      {{22, 40}, 12},
      {{4, 12}, 8},
      {{25, 34, 43}, 6},
      {{4, 8, 12, 16}, 4},
      {{25, 34, 43, 52}, 3},
      {{25, 29, 34, 38, 43, 47, 52}, 2},
      {{5, 9, 13, 17, 25, 29, 34, 38, 43, 47, 52}, 1},
      {{0, 2}, 2},
      {{0, 1, 2, 3, 4, 8, 12, 16}, 1},
      {{24, 28, 33, 37, 42, 46, 51}, 1},
      {{23, 32, 41, 50}, 1},
      {{22, 31, 40, 49}, 1}};

  std::vector<Match> expected_train_matches_algo1 = {
      {{22, 40}, 12},
      {{4, 12}, 8},
      {{25, 34, 43}, 6},
      {{4, 8, 12, 16}, 4},
      {{25, 34, 43, 52}, 3},
      {{25, 29, 34, 38, 43, 47, 52}, 2},
      {{5, 9, 13, 17, 25, 29, 34, 38, 43, 47, 52}, 1},
      {{0, 2}, 2},
      {{24, 28, 33, 37, 42, 46, 51}, 1},
      {{0, 1, 2, 3, 4, 8, 12, 16}, 1}};

  popart::logging::info("simple case of an Op schedule. Is TEST, threshold -1");
  testWithTrain(false, -1.0, expected_test_matches, expected_test_matches);

  popart::logging::info(
      "simple case of an Op schedule. Is TRAIN, threshold -1");
  testWithTrain(
      true, -1.0, expected_train_matches_algo0, expected_train_matches_algo1);

  expected_test_matches = {{{0, 4, 8, 12}, 4}};

  // remove completely saturated at threshold 0.0f
  popart::logging::info("simple case of an Op schedule. Is TEST, threshold 0");
  testWithTrain(false, 0.0, expected_test_matches, expected_test_matches);

  expected_train_matches_algo0 = {
      {{4, 8, 12, 16}, 4},
      {{25, 34, 43, 52}, 3},
      {{25, 29, 34, 38, 43, 47, 52}, 2},
      {{5, 9, 13, 17, 25, 29, 34, 38, 43, 47, 52}, 1},
      {{0, 1, 2, 3, 4, 8, 12, 16}, 1},
      {{24, 28, 33, 37, 42, 46, 51}, 1},
      {{23, 32, 41, 50}, 1},
      {{22, 31, 40, 49}, 1}};

  expected_train_matches_algo1 = {
      {{22, 40}, 12},
      {{25, 34, 43}, 6},
      {{4, 8, 12, 16}, 4},
      {{25, 34, 43, 52}, 3},
      {{25, 29, 34, 38, 43, 47, 52}, 2},
      {{5, 9, 13, 17, 25, 29, 34, 38, 43, 47, 52}, 1},
      {{24, 28, 33, 37, 42, 46, 51}, 1},
      {{0, 1, 2, 3, 4, 8, 12, 16}, 1}};

  popart::logging::info("simple case of an Op schedule. Is TRAIN, threshold 0");
  testWithTrain(
      true, 0.0, expected_train_matches_algo0, expected_train_matches_algo1);

  // at threshold 1.0f, all matmul ops are always cached
  expected_test_matches = {{{0, 4, 8, 12}, 4}};
  popart::logging::info("simple case of an Op schedule. Is TEST, threshold 1");
  testWithTrain(false, 1.0, expected_test_matches, expected_test_matches);

  expected_train_matches_algo0 = {
      {{5, 9, 13, 17, 25, 29, 34, 38, 43, 47, 52}, 1}};

  expected_train_matches_algo1 = {
      {{22, 40}, 12}, {{5, 9, 13, 17, 25, 29, 34, 38, 43, 47, 52}, 1}};

  popart::logging::info("simple case of an Op schedule. Is TRAIN, threshold 1");
  testWithTrain(
      true, 1.0, expected_train_matches_algo0, expected_train_matches_algo1);
}

BOOST_AUTO_TEST_CASE(Anchor0_Subgraph) {

  using namespace fwtools::subgraph;
  using namespace popart;

  auto test = [](const std::vector<Op *> &sched,
                 const std::vector<Match> &expected_matches,
                 float threshold,
                 int algo) {
    auto sets = getSets(sched, expected_matches, threshold, algo);
    std::cout << "Anchor0 test with threshold = " << threshold
              << " and algo = " << algo << std::endl;
    BOOST_CHECK(sets[0] == sets[1]);
  };

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo data_info{"FLOAT", std::vector<int64_t>{2, 3, 5, 5}};
  TensorInfo weight_info{"FLOAT", std::vector<int64_t>{3, 3, 1, 1}};

  auto in0 = builder->addInputTensor(data_info);
  auto w0  = builder->addInputTensor(weight_info);
  auto w1  = builder->addInputTensor(weight_info);
  auto w2  = builder->addInputTensor(weight_info);
  auto w3  = builder->addInputTensor(weight_info);
  auto w4  = builder->addInputTensor(weight_info);
  auto w5  = builder->addInputTensor(weight_info);

  // dilations, group, kernel_shape, pads, strides
  auto o0  = aiOnnx.conv({in0, w0}, {1, 1}, 1, {1, 1}, {0, 0, 0, 0}, {1, 1});
  auto o1  = aiOnnx.conv({o0, w1}, {1, 1}, 1, {1, 1}, {0, 0, 0, 0}, {1, 1});
  auto o2  = aiOnnx.conv({o1, w2}, {1, 1}, 1, {1, 1}, {0, 0, 0, 0}, {1, 1});
  auto o3  = aiOnnx.conv({o2, w3}, {1, 1}, 1, {1, 1}, {0, 0, 0, 0}, {1, 1});
  auto o4  = aiOnnx.conv({o3, w4}, {1, 1}, 1, {1, 1}, {0, 0, 0, 0}, {1, 1});
  auto out = aiOnnx.conv({o4, w5}, {1, 1}, 1, {1, 1}, {0, 0, 0, 0}, {1, 1});

  // auto out = aiOnnx.reducesum({o1});
  builder->addOutputTensor(out);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  std::unique_ptr<Optimizer> optimizer;
  std::vector<std::unique_ptr<L1Loss>> up_losses;
  std::vector<Loss *> losses{};
  auto dataFlow =
      DataFlow(1,
               {{out, AnchorReturnType("ALL")},
                {reservedGradientPrefix() + in0, AnchorReturnType("ALL")},
                {reservedGradientPrefix() + o2, AnchorReturnType("ALL")},
                {reservedGradientPrefix() + out, AnchorReturnType("ALL")},
                {o2, AnchorReturnType("ALL")}});
  auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

  optimizer.reset(new ConstSGD(0.01));
  up_losses.push_back(std::unique_ptr<L1Loss>(
      new L1Loss(out, "l1LossVal", 0.1, ReductionType::SUM)));
  losses = {up_losses[0].get()};

  std::vector<Match> expected_train_matches = {
      {{7, 13}, 6},
      {{0, 3}, 3},
      {{7, 9, 11, 13, 15, 17}, 2},
      {{0, 1, 2, 3, 4, 5, 8, 10, 12, 14, 16, 18}, 1}};

  auto opts = SessionOptions();
  // This test tests the functionality of fwtools::subgraph::getRinseMatches,
  // not the actual outlining of the Ir
  opts.enableOutlining   = false;
  opts.autoRecomputation = RecomputationType::None;

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              optimizer.get(),
              *cpuDevice,
              opts,
              Patterns(PatternsLevel::DEFAULT).enableInPlace(false)});

  std::vector<Match> expected_matches{};
  auto sched = ir.getOpSchedule({});

  popart::logging::debug("Testing Anchor0_Subgraph, algo 0, threshold -1");
  test(sched, expected_train_matches, -1.0f, 0);

  popart::logging::debug("Testing Anchor0_Subgraph, algo 1, threshold -1");
  test(sched, expected_train_matches, -1.0f, 1);

  for (int i = 0; i < sched.size(); ++i) {
    auto x = sched[i];
    std::cout << i << " : " << x->getSubgraphValue() << " : "
              << x->getSubgraphEquivId() << std::endl;
  }
}
