// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE LossScaleUpdateOpTest

#include <boost/test/unit_test.hpp>

#include <popart/builder.hpp>
#include <popart/filereader.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/lossscaleupdate.hpp>
#include <popart/opidentifier.hpp>
#include <popart/session.hpp>
#include <popart/testdevice.hpp>

using namespace popart;

template <typename T> std::string getTypeName();
template <> std::string getTypeName<float>() { return "FLOAT"; }
template <> std::string getTypeName<float16_t>() { return "FLOAT16"; }

template <typename T> void run_test() {
  auto builder = Builder::create();

  // loss scale
  TensorInfo ls_info{getTypeName<T>(), std::vector<int64_t>{}};
  auto ls                                    = builder->addInputTensor(ls_info);
  std::vector<TensorId> all_ls_update_inputs = {ls};

  // gradient statistics, each with 2 bins:
  // (0) not saturated
  // (1) saturated
  int num_histograms = 3;
  std::vector<TensorId> histogram_ids;
  TensorInfo hist_info{"UINT32", std::vector<int64_t>{2}};
  for (int i = 0; i < num_histograms; i++) {
    histogram_ids.push_back(builder->addInputTensor(hist_info));
    all_ls_update_inputs.push_back(histogram_ids[i]);
  }

  // The LossScaleUpdateOp has not been exposed directly in the Builder class,
  // but can still be added to an Onnx model via the customOp method.
  auto t1 = builder->customOp(Onnx::CustomOperators::LossScaleUpdate,
                              1,
                              all_ls_update_inputs,
                              1,
                              {})[0];

  std::map<std::string, std::string> deviceOpts{{"numIPUs", "1"}};

  auto session = popart::InferenceSession::createFromOnnxModel(
      builder->getModelProto(),
      DataFlow(1, {{t1, AnchorReturnType("All")}}),
      createTestDevice(TEST_TARGET),
      popart::InputShapeInfo(),
      SessionOptions(),
      popart::Patterns(PatternsLevel::Default));

  std::vector<T> anchor_data(1);
  popart::NDArrayWrapper<T> anchor_wrapper(anchor_data.data(), {1});
  session->prepareDevice();

  // anchor
  std::map<popart::TensorId, popart::IArray &> anchors = {{t1, anchor_wrapper}};

  // inputs
  std::map<popart::TensorId, popart::IArray &> inputs;

  std::vector<T> loss_scale_val{10.0};
  popart::NDArrayWrapper<T> loss_scale_wrapper(loss_scale_val.data(), ls_info);
  inputs.emplace(ls, loss_scale_wrapper);

  std::vector<uint32_t> grad_stats_vals{8, 56};
  popart::NDArrayWrapper<uint32_t> grad_stats_wrapper(grad_stats_vals.data(),
                                                      hist_info);
  for (int i = 0; i < num_histograms; i++) {
    inputs.emplace(histogram_ids[i], grad_stats_wrapper);
  }

  popart::StepIO stepio(inputs, anchors);
  session->run(stepio);

  // The upper bin count is much higher than the lower bin count -
  // the loss scale will be scaled down by a factor of 2.
  std::vector<T> expected = {5.0};
  for (size_t i = 0; i < anchor_data.size(); ++i) {
    BOOST_CHECK_EQUAL(anchor_data[i], expected[i]);
  }
}

BOOST_AUTO_TEST_CASE(LossScaleUpdateOp_test) {
  run_test<float>();
  run_test<float16_t>();
}