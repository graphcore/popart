#define BOOST_TEST_MODULE Op0SubgraphTest

// tests using the Op class as the subgraph template parameter

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <vector>
#include <poponnx/builder.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/logging.hpp>
#include <poponnx/op.hpp>
#include <poponnx/subgraph/subgraph.hpp>

#include <poponnx/filereader.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/tensordata.hpp>

using namespace fwtools::subgraph;
using namespace poponnx;

BOOST_AUTO_TEST_CASE(Op0_Subgraph) {

  auto test = [](const std::vector<Op *> &sched,
                 const std::vector<Match> &expected_matches) {
    // get the matches
    auto matches = getMatches<Op>(sched);

    // compare the final matches to those expected in this test
    std::stringstream ss;
    ss << "\nExpected matches:";
    for (auto &x : expected_matches) {
      ss << "\n" << x;
    }
    ss << "\nComputed matches:";
    for (auto &x : matches) {
      ss << "\n" << x;
    }
    poponnx::logging::debug(ss.str());
    BOOST_CHECK(matches == expected_matches);
  };

  // ----------------------------------------------------
  auto testWithTrain = [&test](bool train) {
    poponnx::logging::info("simple case of an Op schedule. Is train ? {}",
                           train);

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
    auto dataFlow = DataFlow(1, {{out, AnchorReturnType("ALL")}});

    if (train) {
      optimizer.reset(new ConstSGD(0.01));
      up_losses.push_back(
          std::unique_ptr<L1Loss>(new L1Loss(out, "l1LossVal", 0.1)));
      losses = {up_losses[0].get()};
    }

    // TODO : this test will fail if saturated sub-graphs are removed
    // see T7255

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
    // 11 MatMulRhsGrad       | %      [@
    // 12 MatMulLhsGrad       | %       @
    // 13 ConstSGDVarUpdate   | %       @]
    // 14 ReluGrad            | %                *
    // 15 MatMulRhsGrad       |        [@
    // 16 MatMulLhsGrad       |         @
    // 17 ConstSGDVarUpdate   |         @]
    // 18 ReluGrad            | %                *
    // 19 MatMulRhsGrad       | %      [@
    // 20 MatMulLhsGrad       | %       @
    // 21 ConstSGDVarUpdate   | %       @]
    // 22 ReluGrad            | %                *
    // 23 MatMulRhsGrad       |
    // 24 ConstSGDVarUpdate   |
    std::vector<Match> expected_train_matches = {{{10, 18}, 5},
                                                 {{0, 4}, 4},
                                                 {{11, 15, 19}, 3},
                                                 {{0, 2, 4, 6}, 2},
                                                 {{13, 17, 21, 24}, 1},
                                                 {{11, 15, 19, 23}, 1},
                                                 {{10, 14, 18, 22}, 1}};
    Ir ir;
    ir.prepare({modelProto,
                InputShapeInfo(),
                dataFlow,
                losses,
                optimizer.get(),
                {},
                Patterns(PatternsLevel::DEFAULT)});

    std::vector<Match> expected_matches{};
    auto sched = ir.getOpSchedule({});

    if (train) {
      test(sched, expected_train_matches);
    } else {
      test(sched, expected_test_matches);
    }
  };

  testWithTrain(false);
  testWithTrain(true);
}
