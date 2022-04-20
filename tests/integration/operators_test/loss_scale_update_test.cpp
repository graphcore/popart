// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE LossScaleUpdateOpTest

#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <onnxutil.hpp>
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

template <typename T> void run_test() {
  auto builder = Builder::create();

  // loss scale update factor
  TensorInfo ls_info{getTypeName<T>(), std::vector<int64_t>{}};
  auto ls_update_factor = builder->addInputTensor(ls_info);

  // gradient statistics, 2 bins:
  // (0) not saturated
  // (1) saturated
  TensorInfo statistics_info{"UINT32", std::vector<int64_t>{2}};
  auto statistics_tensor_id = builder->addInputTensor(statistics_info);

  std::vector<TensorId> all_ls_update_inputs = {ls_update_factor,
                                                statistics_tensor_id};

  // The LossScaleUpdateOp has not been exposed directly in the Builder class,
  // but can still be added to an Onnx model via the customOp method.
  std::map<std::string, popart::any> attributes;
  attributes["updateFactorDType"] =
      static_cast<int64_t>(onnxutil::getTPDataType(getDataType<T>()));
  auto out = builder->customOp(Onnx::CustomOperators::LossScaleUpdate,
                               1,
                               all_ls_update_inputs,
                               1,
                               attributes)[0];

  std::map<std::string, std::string> deviceOpts{{"numIPUs", "1"}};

  auto session = popart::InferenceSession::createFromOnnxModel(
      builder->getModelProto(),
      DataFlow(1, {out}, AnchorReturnType("All")),
      createTestDevice(TEST_TARGET),
      popart::InputShapeInfo(),
      SessionOptions(),
      popart::Patterns(PatternsLevel::Default));

  std::vector<T> anchor_data0(1);
  popart::NDArrayWrapper<T> anchor_wrapper0(anchor_data0.data(), {1});
  session->prepareDevice();

  // anchor
  std::map<popart::TensorId, popart::IArray &> anchors = {
      {out, anchor_wrapper0}};

  // inputs
  std::map<popart::TensorId, popart::IArray &> inputs;

  std::vector<T> ls_update_val{4.0};
  popart::NDArrayWrapper<T> ls_update_wrapper(ls_update_val.data(), ls_info);
  inputs.emplace(ls_update_factor, ls_update_wrapper);

  std::vector<uint32_t> grad_stats_vals{8, 56};
  popart::NDArrayWrapper<uint32_t> grad_stats_wrapper(grad_stats_vals.data(),
                                                      statistics_info);

  inputs.emplace(statistics_tensor_id, grad_stats_wrapper);

  popart::StepIO stepio(inputs, anchors);
  session->run(stepio);

  // The upper bin count is much higher than the lower bin count -
  // the loss scale will be scaled down by a factor of 2, and the inverse
  // loss scale scaled up by the same factor.
  std::vector<T> expected_update_factor = {2.0};
  for (size_t i = 0; i < anchor_data0.size(); ++i) {
    BOOST_CHECK_EQUAL(anchor_data0[i], expected_update_factor[i]);
  }
}

BOOST_AUTO_TEST_CASE(LossScaleUpdateOp_test) {
  run_test<float>();
  run_test<float16_t>();
}
