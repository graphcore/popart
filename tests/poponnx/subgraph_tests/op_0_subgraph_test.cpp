#define BOOST_TEST_MODULE Op0SubgraphTest

// tests using the Op class as the subgraph template parameter

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <vector>
#include <poponnx/builder.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/logging.hpp>
#include <poponnx/op.hpp>
#include <poponnx/subgraph/outliner.hpp>
#include <poponnx/tensornames.hpp>

#include <poponnx/filereader.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/tensordata.hpp>

namespace {

using namespace fwtools::subgraph;
using namespace poponnx;

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

  poponnx::logging::debug(ss.str());

  std::vector<std::set<Match>> sets = {s_expected_matches, s_matches};
  return sets;
}

} // namespace

BOOST_AUTO_TEST_CASE(Op0_Subgraph) {

  using namespace fwtools::subgraph;
  using namespace poponnx;

  auto test = [](const std::vector<Op *> &sched,
                 const std::vector<Match> &expected_matches,
                 float threshold,
                 int algo) {
    auto sets = getSets(sched, expected_matches, threshold, algo);
    BOOST_CHECK(sets[0] == sets[1]);
  };

  // ----------------------------------------------------
  auto testWithTrain = [&test](bool train,
                               float threshold,
                               std::vector<Match> expected_matches) {
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
      up_losses.push_back(
          std::unique_ptr<L1Loss>(new L1Loss(out, "l1LossVal", 0.1)));
      losses = {up_losses[0].get()};
    }

    Ir ir;
    ir.prepare({modelProto,
                InputShapeInfo(),
                dataFlow,
                losses,
                optimizer.get(),
                *cpuDevice,
                {},
                Patterns(PatternsLevel::DEFAULT)});

    auto sched = ir.getOpSchedule({});

    test(sched, expected_matches, threshold, 0);
    test(sched, expected_matches, threshold, 1);
  };

  // test mode:
  // ---------/
  // 0  1    2  3    4  5    6  7    8
  // mm relu mm relu mm relu mm relu reduce
  std::vector<Match> expected_test_matches = {{{0, 4}, 4}, {{0, 2, 4, 6}, 2}};

  // train mode:
  // 0  MatMul              |   [X         [w
  // 1  ReluInplace         |    X          w]
  // 2  MatMul              |    X         [w
  // 3  ReluInplace         |    X]         w]
  // 4  MatMul              |   [X         [w
  // 5  ReluInplace         |    X          w]
  // 6  MatMul              |    X         [w
  // 7  ReluInplace         |    X]         w]
  // 8  Identity            |
  // 9  L1Grad              |
  // 10 ReluGrad            | %                *
  // 11 MatMulRhsGrad       | %      [@             ^
  // 12 MatMulLhsGrad       | %       @
  // 13 ConstSGDVarUpdate   | %       @]                $
  // 14 ReluGrad            | %                *
  // 15 MatMulRhsGrad       |        [@             ^
  // 16 MatMulLhsGrad       |         @
  // 17 ConstSGDVarUpdate   |         @]                $
  // 18 ReluGrad            | %                *
  // 19 MatMulRhsGrad       | %      [@             ^
  // 20 MatMulLhsGrad       | %       @
  // 21 ConstSGDVarUpdate   | %       @]                $
  // 22 ReluGrad            | %                *
  // 23 MatMulRhsGrad       |                       ^
  // 24 ConstSGDVarUpdate   |                           $
  std::vector<Match> expected_train_matches = {
      //
      {{10, 18}, 5}, // RR, MM, MM, RR,  VU
                     //
      {{0, 4}, 4},   // MM, RI, MM, RI
      {{11, 15, 19}, 3},
      {{0, 2, 4, 6}, 2},
      {{11, 15, 19, 23}, 1},
      {{10, 14, 18, 22}, 1},
      {{13, 17, 21, 24}, 1}};

  poponnx::logging::info(
      "simple case of an Op schedule. Is TEST, threshold -1");
  testWithTrain(false, -1.0, expected_test_matches);

  poponnx::logging::info(
      "simple case of an Op schedule. Is TRAIN, threshold -1");
  testWithTrain(true, -1.0, expected_train_matches);

  // remove completely saturated at threshold 0.0f
  expected_test_matches = {{{0, 2, 4, 6}, 2}};
  poponnx::logging::info("simple case of an Op schedule. Is TEST, threshold 0");
  testWithTrain(false, 0.0, expected_test_matches);

  expected_train_matches = {{{11, 15, 19}, 3},
                            {{0, 2, 4, 6}, 2},
                            {{11, 15, 19, 23}, 1},
                            {{10, 14, 18, 22}, 1},
                            {{13, 17, 21, 24}, 1}};
  poponnx::logging::info(
      "simple case of an Op schedule. Is TRAIN, threshold 0");
  testWithTrain(true, 0.0, expected_train_matches);

  // at threshold 1.0f, all matmul ops are always cached
  expected_test_matches = {{{0, 2, 4, 6}, 2}};
  poponnx::logging::info("simple case of an Op schedule. Is TEST, threshold 1");
  testWithTrain(false, 1.0, expected_test_matches);

  expected_train_matches = {
      {{11, 15, 19}, 3},
      {{0, 2, 4, 6}, 2},
      {{11, 15, 19, 23}, 1},

  };
  poponnx::logging::info(
      "simple case of an Op schedule. Is TRAIN, threshold 1");
  testWithTrain(true, 1.0, expected_train_matches);
}

BOOST_AUTO_TEST_CASE(Anchor0_Subgraph) {

  using namespace fwtools::subgraph;
  using namespace poponnx;

  auto test = [](const std::vector<Op *> &sched,
                 const std::vector<Match> &expected_matches,
                 float threshold,
                 int algo) {
    auto sets = getSets(sched, expected_matches, threshold, algo);
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
  up_losses.push_back(
      std::unique_ptr<L1Loss>(new L1Loss(out, "l1LossVal", 0.1)));
  losses = {up_losses[0].get()};

  std::vector<Match> expected_train_matches = {
      {{7, 13}, 6},
      {{0, 3}, 3},
      {{7, 9, 11, 13, 15, 17}, 2},
      {{0, 1, 2, 3, 4, 5, 8, 10, 12, 14, 16, 18}, 1}};
  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              optimizer.get(),
              *cpuDevice,
              {},
              Patterns(PatternsLevel::DEFAULT)});

  std::vector<Match> expected_matches{};
  auto sched = ir.getOpSchedule({});

  poponnx::logging::debug("Testing Anchor0_Subgraph, algo 0, threshold -1");
  test(sched, expected_train_matches, -1.0f, 0);

  poponnx::logging::debug("Testing Anchor0_Subgraph, algo 1, threshold -1");
  test(sched, expected_train_matches, -1.0f, 1);

  for (int i = 0; i < sched.size(); ++i) {
    auto x = sched[i];
    std::cout << i << " : " << x->getSubgraphValue() << " : "
              << x->getSubgraphEquivId() << std::endl;
  }
}
