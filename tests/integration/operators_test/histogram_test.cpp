// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE HistogramOpTest

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <popart/builder.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/session.hpp>
#include <popart/testdevice.hpp>

#include "popart/dataflow.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/half.hpp"
#include "popart/inputshapeinfo.hpp"
#include "popart/names.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/stepio.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/vendored/any.hpp"

namespace popart {
class IArray;
} // namespace popart

using namespace popart;

template <typename T> std::string getTypeName();
template <> std::string getTypeName<float>() { return "FLOAT"; }
template <> std::string getTypeName<float16_t>() { return "FLOAT16"; }

template <typename T> void run_test(bool absoluteOfInput) {
  auto builder = Builder::create();
  TensorInfo t0_info{getTypeName<T>(), std::vector<int64_t>{10}};
  auto t0 = builder->addInputTensor(t0_info);

  std::vector<float> levels = {0.1, 7, 3.1};

  // The HistogramOp has not been exposed directly in the Builder class,
  // but can still be added to an Onnx model via the customOp method.
  auto t1 = builder->customOp(
      Onnx::CustomOperators::Histogram,
      1,
      {t0},
      1,
      {{"levels", levels}, {"absoluteOfInput", (int)absoluteOfInput}})[0];

  std::map<std::string, std::string> deviceOpts{{"numIPUs", "1"}};

  auto session = popart::InferenceSession::createFromOnnxModel(
      builder->getModelProto(),
      DataFlow(1, {{t1, AnchorReturnType("All")}}),
      createTestDevice(TEST_TARGET),
      popart::InputShapeInfo(),
      SessionOptions(),
      popart::Patterns(PatternsLevel::Default));

  auto outElms = static_cast<int64_t>(levels.size() + 1);
  std::vector<uint32_t> anchor_data(outElms);
  popart::NDArrayWrapper<uint32_t> anchor_wrapper(anchor_data.data(),
                                                  {outElms});
  session->prepareDevice();

  // anchor
  std::map<popart::TensorId, popart::IArray &> anchors = {{t1, anchor_wrapper}};

  // input
  std::vector<T> input_vals{-10, -0.1, 0.01, 0.09, 1.1, 4, 6.9, 7, 8.0, 900};
  popart::NDArrayWrapper<T> input_wrapper(input_vals.data(), t0_info);
  std::map<popart::TensorId, popart::IArray &> inputs = {{t0, input_wrapper}};
  popart::StepIO stepio(inputs, anchors);
  session->run(stepio);

  std::vector<uint32_t> expected;

  if (absoluteOfInput) {
    // expected result:
    // x < 0.1        : 2
    // 0.1 <= x < 3.1 : 2
    // 3.1 <= x < 7   : 2
    // x >= 7         : 4
    expected = {2, 2, 2, 4};
  } else {
    // expected result:
    // x < 0.1        : 4
    // 0.1 <= x < 3.1 : 1
    // 3.1 <= x < 7   : 2
    // x >= 7         : 3
    expected = {4, 1, 2, 3};
  }

  for (size_t i = 0; i < anchor_data.size(); ++i) {
    std::cout << anchor_data[i] << std::endl;
    BOOST_CHECK_EQUAL(anchor_data[i], expected[i]);
  }
}

BOOST_AUTO_TEST_CASE(HistogramOp_test) {
  run_test<float>(false);
  run_test<float16_t>(false);

  run_test<float>(true);
  run_test<float16_t>(true);
}
