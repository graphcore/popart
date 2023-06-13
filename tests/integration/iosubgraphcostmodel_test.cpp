// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "popart/onnxoperators.gen.hpp"
#include "popart/popx/irlowering.hpp"
#include "popart/scheduler_requireoptimal.hpp"
#include <boost/test/tools/old/interface.hpp>
#define BOOST_TEST_MODULE OutliningIrTest

#include "test_runner.hpp"

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/tensorinfo.hpp>

#include "popart/graphid.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/subgraph/iosubgraphcostmodel.hpp"
#include "popart/vendored/any.hpp"
#include "popart/voiddata.hpp"

float referenceVals[23][23] = {
    {0,           207533,      207539,      207544,      415190,
     415196,      415202,      622848,      622853,      622859,
     830505,      830511,      830517,      1.03816e+06, 1.03817e+06,
     1.03817e+06, 1.24582e+06, 1.24583e+06, 1.24583e+06, 1.45348e+06,
     1.45348e+06, 1.45349e+06, 1.66114e+06},
    {0,           0,           -107.48,     -101.823,    207544,
     207550,      207556,      415202,      415207,      415213,
     622859,      622865,      622870,      830517,      830522,
     830528,      1.03817e+06, 1.03818e+06, 1.03819e+06, 1.24583e+06,
     1.24584e+06, 1.24584e+06, 1.45349e+06},
    {0,           0,           0,           -107.48,     207539,
     207544,      207550,      415196,      415202,      415207,
     622853,      622859,      622865,      830511,      830517,
     830522,      1.03817e+06, 1.03817e+06, 1.03818e+06, 1.24583e+06,
     1.24583e+06, 1.24584e+06, 1.45348e+06},
    {0,           0,           0,           0,           207533,
     207539,      207544,      415190,      415196,      415202,
     622848,      622853,      622859,      830505,      830511,
     830517,      1.03816e+06, 1.03817e+06, 1.03817e+06, 1.24582e+06,
     1.24583e+06, 1.24583e+06, 1.45348e+06},
    {0,        0,           0,           0,           0,          -107.48,
     -101.823, 207544,      207550,      207556,      415202,     415207,
     415213,   622859,      622865,      622870,      830517,     830522,
     830528,   1.03817e+06, 1.03818e+06, 1.03819e+06, 1.24583e+06},
    {0,       0,           0,           0,           0,          0,
     -107.48, 207539,      207544,      207550,      415196,     415202,
     415207,  622853,      622859,      622865,      830511,     830517,
     830522,  1.03817e+06, 1.03817e+06, 1.03818e+06, 1.24583e+06},
    {0,      0,           0,           0,           0,          0,
     0,      207533,      207539,      207544,      415190,     415196,
     415202, 622848,      622853,      622859,      830505,     830511,
     830517, 1.03816e+06, 1.03817e+06, 1.03817e+06, 1.24582e+06},
    {0,       0,        0,      0,      0,      0,      0,          0,
     -107.48, -101.823, 207544, 207550, 207556, 415202, 415207,     415213,
     622859,  622865,   622870, 830517, 830522, 830528, 1.03817e+06},
    {0,      0,       0,      0,      0,      0,      0,          0,
     0,      -107.48, 207539, 207544, 207550, 415196, 415202,     415207,
     622853, 622859,  622865, 830511, 830517, 830522, 1.03817e+06},
    {0,      0,      0,      0,      0,      0,      0,          0,
     0,      0,      207533, 207539, 207544, 415190, 415196,     415202,
     622848, 622853, 622859, 830505, 830511, 830517, 1.03816e+06},
    {0,      0,      0,      0,       0,        0,      0,      0,
     0,      0,      0,      -107.48, -101.823, 207544, 207550, 207556,
     415202, 415207, 415213, 622859,  622865,   622870, 830517},
    {0,      0,      0,      0,      0,       0,      0,      0,
     0,      0,      0,      0,      -107.48, 207539, 207544, 207550,
     415196, 415202, 415207, 622853, 622859,  622865, 830511},
    {0,      0,      0,      0,      0,      0,      0,      0,
     0,      0,      0,      0,      0,      207533, 207539, 207544,
     415190, 415196, 415202, 622848, 622853, 622859, 830505},
    {0,      0,      0,      0,      0,      0,      0,       0,
     0,      0,      0,      0,      0,      0,      -107.48, -101.823,
     207544, 207550, 207556, 415202, 415207, 415213, 622859},
    {0, 0, 0, 0,       0,      0,      0,      0,      0,      0,      0,     0,
     0, 0, 0, -107.48, 207539, 207544, 207550, 415196, 415202, 415207, 622853},
    {0, 0, 0, 0, 0,      0,      0,      0,      0,      0,      0,     0,
     0, 0, 0, 0, 207533, 207539, 207544, 415190, 415196, 415202, 622848},
    {0, 0, 0, 0, 0, 0,       0,        0,      0,      0,      0,     0,
     0, 0, 0, 0, 0, -107.48, -101.823, 207544, 207550, 207556, 415202},
    {0, 0, 0, 0, 0, 0, 0,       0,      0,      0,      0,     0,
     0, 0, 0, 0, 0, 0, -107.48, 207539, 207544, 207550, 415196},
    {0, 0, 0, 0, 0, 0, 0, 0,      0,      0,      0,     0,
     0, 0, 0, 0, 0, 0, 0, 207533, 207539, 207544, 415190},
    {0, 0, 0, 0, 0, 0, 0, 0, 0,       0,        0,     0,
     0, 0, 0, 0, 0, 0, 0, 0, -107.48, -101.823, 207544},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       0,     0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, -107.48, 207539},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 207533},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

BOOST_AUTO_TEST_CASE(TestOutliningCopyCost) {

  // This test checks if, for a given model, a specific cost function,
  // IoSubgraphCostModel::value returns a specific set of values. This test was
  // introduced to test that changes introduced to speed up this cost function
  // did not change the values that it generates. If in future the logic of the
  // cost function is changed it would be valid to delete this test.

  auto test = [](int numOutliningContexts = 1) {
    TestRunner runner;
    runner.isTraining = true;
    int N             = 8;
    int B             = 8;
    int size          = 100;

    runner.buildModel([&](auto &builder) {
      auto aiOnnx = builder.aiOnnxOpset9();
      TensorInfo inInfo{"FLOAT", std::vector<int64_t>{B, 1, size}};
      auto act = builder.addInputTensor(inInfo);
      // N layers
      for (int n = 0; n < N; ++n) {
        auto attribute = std::vector<std::string>{
            "context", std::to_string(n % numOutliningContexts)};
        TensorInfo wInfo{"FLOAT", std::vector<int64_t>{1, size, size}};
        std::vector<TestTensor> inputs;
        std::vector<TestTensor> outputs;
        std::vector<float> wData(wInfo.nelms(), 0);
        ConstVoidData wCVData{wData.data(), wInfo};
        auto w = builder.addInitializedInputTensor(wCVData);
        act = aiOnnx.matmul({act, w}, logging::format("CHECKOP_MM: [{}]", n));
        builder.addNodeAttribute(sOutlineAttribute, attribute, {act});
        builder.virtualGraph(act, n % 2);
        act = aiOnnx.relu({act}, logging::format("CHECKOP_RELU: [{}]", n));
        builder.addNodeAttribute(sOutlineAttribute, attribute, {act});
        builder.virtualGraph(act, n % 2);
      }

      runner.isTraining                 = false;
      runner.opts.explicitRecomputation = false;
      // Don't enable outlining as we want ot calculate potential subgraph copy
      // costs.
      runner.opts.enableOutlining                = false;
      runner.opts.outlineThreshold               = 1.0;
      runner.opts.enableOutliningCopyCostPruning = false;
      runner.opts.virtualGraphMode               = VirtualGraphMode::Manual;
      runner.patterns = Patterns(PatternsLevel::Default);
      // Disable so that no false negatives (rhs vs. lhs inplace) exist
      runner.patterns.enableInPlace(false);

      return act;
    });

    // Testing that the schedule is as expected for batch serialization:
    runner.checkIr([&](Ir &ir) {
      auto ioSCM = outline::IoSubgraphCostModel();
      auto schedule =
          ir.getMainGraph().getOpSchedule({}, RequireOptimalSchedule::Yes);
      std::map<Op *, int> schedule_index;
      for (int i = 0; i < schedule.size(); ++i) {
        schedule_index[schedule[i]] = i;
      }

      auto N           = schedule.size();
      auto actual_vals = std::vector<std::tuple<int, int, float>>();
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

          auto val = ioSCM.value(i, j, schedule, schedule_index);

          std::cout << val << " vs " << referenceVals[i][j] << std::endl;

          // I copied the ref values from std out, so there may be some
          // rounding.
          BOOST_CHECK_CLOSE(val, referenceVals[i][j], 0.01);
        }
      }
    });
  };
  test(4);
}