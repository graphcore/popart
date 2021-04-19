// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Op0SubgraphTest

// tests using the Op class as the subgraph template parameter

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <memory>
#include <vector>
#include <popart/builder.hpp>
#include <popart/filereader.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/op.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/mul.hpp>
#include <popart/op/relu.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/optimizer.hpp>
#include <popart/subgraph/outliner.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensornames.hpp>
#include <popart/testdevice.hpp>

namespace {

using namespace fwtools::subgraph;
using namespace popart;

std::vector<std::set<Match>> getSets(const std::vector<Op *> &sched,
                                     const std::vector<Match> &expected_matches,
                                     float threshold,
                                     int algo) {

  std::vector<std::pair<size_t, size_t>> sequences(sched.size());
  float sequenceBreakCost = 0.0f;

  // get the matches
  //
  std::vector<Match> matches;
  if (algo == 0) {
    matches = getRinseMatches<Op>(sched,
                                  sequences,
                                  threshold,
                                  sequenceBreakCost,
                                  OutlinerAlgorithm::ALGO0);
  } else if (algo == 1) {
    matches = getRinseMatches<Op>(sched,
                                  sequences,
                                  threshold,
                                  sequenceBreakCost,
                                  OutlinerAlgorithm::ALGO1);
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
    std::vector<TensorId> mulOutIds;
    std::vector<TensorId> reluIds{in0};
    int reps = 4;
    for (int i = 0; i < reps; ++i) {
      weightIds.push_back(builder->addInitializedInputTensor(weight_data));
    }
    for (int i = 0; i < reps; ++i) {
      mulOutIds.push_back(aiOnnx.mul({reluIds.back(), weightIds[i]}));
      reluIds.push_back(aiOnnx.relu({mulOutIds.back()}));
    }
    auto out = aiOnnx.reducesum({reluIds.back()}, std::vector<int64_t>{});
    auto l1  = builder->aiGraphcoreOpset1().l1loss({out}, 0.1);

    builder->addOutputTensor(out);

    std::unique_ptr<Optimizer> optimizer;
    TensorId loss;

    if (train) {
      loss = l1;
      optimizer.reset(new ConstSGD(0.01));
    }

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);
    auto dataFlow   = DataFlow(1, {{out, AnchorReturnType("All")}});
    auto device     = createTestDevice(TEST_TARGET);

    auto opts = SessionOptions();
    // This test tests the functionality of fwtools::subgraph::getRinseMatches,
    // not the actual outlining of the Ir
    opts.enableOutlining                        = false;
    opts.autoRecomputation                      = RecomputationType::None;
    opts.kahnTieBreaker                         = "fifo";
    opts.swapLimitScheduler                     = -1;
    opts.transitiveClosureOptimizationThreshold = 0;

    Ir ir;
    opts.mergeVarUpdate = MergeVarUpdateType::None;

    ir.prepare({modelProto,
                InputShapeInfo(),
                dataFlow,
                loss,
                optimizer.get(),
                *device,
                opts,
                Patterns(PatternsLevel::All).enableInPlace(false)});

    // pin down the scheduler in a few places, to reduce test shadiness;
    OpsBeforeKey topoCons;

    auto insertBefores = [&topoCons](const std::vector<Op *> &ops) {
      for (auto op : ops) {
        topoCons.insert({op, {}});
        for (auto t : op->input->tensors()) {
          for (auto c : t->consumers.getOps()) {
            if (c != op) {
              topoCons[op].push_back(c);
            }
          }
        }
      }
    };

    // The four VarUpdates are constrained to be final consumers of their
    // inputs. There are 1 or 2 other consumers of their inputs.
    auto sgd0VarUpdates = ir.opsOfType(Onnx::CustomOperators::SGD0VarUpdate);
    insertBefores(sgd0VarUpdates);
    BOOST_CHECK(topoCons.size() == sgd0VarUpdates.size());
    for (const auto &x : topoCons) {
      BOOST_CHECK(x.second.size() == 1 || x.second.size() == 2);
    }

    // ReluGrads are constrained to be final consumers
    auto reluGrads = ir.opsOfType(Onnx::GradOperators::ReluGrad);
    insertBefores(reluGrads);

    // pin-down the relative order of the matmuls in the bwd pass
    uint64_t pole{0};
    for (auto x : reluGrads) {
      auto out       = x->output->tensors()[0];
      auto consumers = out->consumers.getOps();
      if (consumers.size() == 2) {
        topoCons[consumers[pole]].push_back(consumers[1 - pole]);
      }
      pole = pole == 0 ? 1 : 0;
    }
    auto sched = ir.getOpSchedule(topoCons, RequireOptimalSchedule::Yes);

    // The training schedule looks like this (05 / September / 2019)
    //
    // 0 Mul
    // 1 Relu
    // 2 Mul
    // 3 Relu
    // 4 Mul
    // 5 Relu
    // 6 Mul
    // 7 Relu
    // 8 Identity
    // 9 L1Grad
    // 10 ReluGrad
    // 11 Mul
    // 12 Mul
    // 13 ConstSGDVarUpdate
    // 14 ReluGrad
    // 15 Mul
    // 16 Mul
    // 17 ConstSGDVarUpdate
    // 18 ReluGrad
    // 19 Mul
    // 20 Mul
    // 21 ConstSGDVarUpdate
    // 22 ReluGrad
    // 23 Mul
    // 24 ConstSGDVarUpdate
    //
    // The scheduler might change in the future, or patterns might modify the
    // Ops. if so, this test needs to be redesigned, or patterns disabled, or
    // further scheduler pins inserted

    // verify that a change to the scheduler or a transform hasn't made this
    // test invalid
    if (train) {
      BOOST_CHECK(sched.size() == 25);
      for (auto i : {0, 2, 4, 6, 11, 12, 15, 16, 19, 20, 23}) {
        BOOST_CHECK(dynamic_cast<MulOp *>(sched[i]));
      }

      for (auto i : {13, 17, 21, 24}) {
        BOOST_CHECK(dynamic_cast<VarUpdateOp *>(sched[i]));
      }

      for (auto i : {1, 3, 5, 7}) {
        BOOST_CHECK(dynamic_cast<ReluOp *>(sched[i]));
      }

      for (auto i : {10, 14, 18, 22}) {
        BOOST_CHECK(dynamic_cast<ReluGradOp *>(sched[i]));
      }
    }

    int i = 0;
    for (auto op : sched) {
      std::cout << i++ << " " << op->opid.type << std::endl;
    }

    test(sched, expected_matches_algo0, threshold, 0);
    test(sched, expected_matches_algo1, threshold, 1);
  };

  std::vector<Match> expected_test_matches{{{0, 2, 4, 6}, 2}};

  // expected with threshold  = a very small value (just slightly positive to
  // prevent redundant matches)
  std::vector<Match> expected_train_matches_algo0 = {
      {{0, 2, 4, 6}, 2},
      {{0, 2, 4, 6, 11, 12, 15, 16, 19, 20, 23}, 1},
      {{13, 17, 21, 24}, 1},
      {{10, 14, 18, 22}, 1}};

  std::vector<Match> expected_train_matches_algo1 =
      expected_train_matches_algo0;

  popart::logging::info("simple case of an Op schedule. Is TEST, threshold 1");
  testWithTrain(false, 0.01, expected_test_matches, expected_test_matches);

  popart::logging::info("simple case of an Op schedule. Is TRAIN, threshold 1");
  testWithTrain(
      true, 0.01, expected_train_matches_algo0, expected_train_matches_algo1);

  // expected threshold with a negative threshold
  expected_train_matches_algo0 = {
      {{10, 18}, 5},
      {{0, 4}, 4},
      {{11, 15, 19}, 2},
      {{0, 2, 4, 6}, 2},
      {{13, 17, 21, 24}, 1},
      {{10, 14, 18, 22}, 1},
      {{0, 2, 4, 6, 11, 12, 15, 16, 19, 20, 23}, 1}};

  expected_train_matches_algo1 = expected_train_matches_algo0;

  popart::logging::info(
      "simple case of an Op schedule. Is TRAIN, threshold -1");
  testWithTrain(
      true, -1.0, expected_train_matches_algo0, expected_train_matches_algo1);
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

  // auto out = aiOnnx.reducesum({o1}, std::vector<int64_t>{});
  auto l1 = builder->aiGraphcoreOpset1().l1loss({out}, 0.1);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  std::unique_ptr<Optimizer> optimizer;
  auto dataFlow =
      DataFlow(1,
               {{out, AnchorReturnType("All")},
                {reservedGradientPrefix() + in0, AnchorReturnType("All")},
                {reservedGradientPrefix() + o2, AnchorReturnType("All")},
                {reservedGradientPrefix() + out, AnchorReturnType("All")},
                {o2, AnchorReturnType("All")}});
  auto device = createTestDevice(TEST_TARGET);

  optimizer.reset(new ConstSGD(0.01));

  std::vector<Match> expected_train_matches = {
      {{7, 13}, 6},
      {{0, 3}, 3},
      {{7, 9, 11, 13, 15, 17}, 2},
      {{0, 1, 2, 3, 4, 5, 8, 10, 12, 14, 16, 18}, 1}};

  auto opts = SessionOptions();
  // This test tests the functionality of fwtools::subgraph::getRinseMatches,
  // not the actual outlining of the Ir
  opts.enableOutlining                        = false;
  opts.autoRecomputation                      = RecomputationType::None;
  opts.kahnTieBreaker                         = "fifo";
  opts.swapLimitScheduler                     = -1;
  opts.transitiveClosureOptimizationThreshold = 0;

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              l1,
              optimizer.get(),
              *device,
              opts,
              Patterns(PatternsLevel::Default).enableInPlace(false)});

  std::vector<Match> expected_matches{};
  auto sched = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);

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
