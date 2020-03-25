// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE AddBias0InplaceTest

#include <../test_runner.hpp>
#include <boost/test/unit_test.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/tensors.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(Inplace_addBias0) {
  auto tinfo = [](const std::vector<int64_t> &shape) -> TensorInfo {
    return {"FLOAT", shape};
  };

  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  auto add_input_tensor = [&](const std::string &tensorId,
                              const std::vector<int64_t> shape) {
    std::vector<float> data;
    int size = 1;
    for (auto s : shape) {
      size = size * s;
    }
    for (int i = 0; i < size; i++) {
      data.push_back(i + 1);
    }
    inputs.push_back(TestTensor::create<float>(tensorId, data, shape));
  };

  bool inplaceEnabled = false;

  auto buildModel = [&](Builder &builder) {
    auto aiOnnx = builder.aiOnnxOpset9();
    auto in0    = builder.addInputTensor(tinfo({1, 1, 8, 8}));
    add_input_tensor(in0, {1, 1, 8, 8});
    auto w0 = builder.addInputTensor(tinfo({2, 1, 2, 2}));
    add_input_tensor(w0, {2, 1, 2, 2});
    auto b0 = builder.addInputTensor(tinfo({2}));
    add_input_tensor(b0, {2});

    auto c0 =
        aiOnnx.conv({in0, w0, b0}, {1, 1}, 1, {2, 2}, {0, 0, 0, 0}, {2, 2});
    auto out = aiOnnx.identity({c0});

    outputs.push_back(TestTensor::create<float>(out, {1, 2, 4, 4}));
    builder.addOutputTensor(out);

    return out;
  };

  auto checkIr = [&inplaceEnabled](Ir &ir) {
    if (inplaceEnabled) {
      BOOST_CHECK_EQUAL(ir.opsOfType(Onnx::CustomOperators::AddBias).size(), 0);
      BOOST_CHECK_EQUAL(
          ir.opsOfType(Onnx::CustomOperators::AddBiasInplace).size(), 1);
    } else {
      BOOST_CHECK_EQUAL(ir.opsOfType(Onnx::CustomOperators::AddBias).size(), 1);
      BOOST_CHECK_EQUAL(
          ir.opsOfType(Onnx::CustomOperators::AddBiasInplace).size(), 0);
    }
  };

  auto checkResult = [](TestTensor &result) {};

  auto run_test = [&]() {
    TestRunner runner;
    runner.patterns = Patterns({PreAliasPatternType::SPLITCONVBIAS});
    runner.patterns.enableInPlace(inplaceEnabled);
    runner.patterns.enableUpdateInplacePrioritiesForIpu(true);
    runner.buildModel(buildModel);
    runner.checkIr(checkIr);
    runner.checkResult(checkResult, inputs, outputs);
  };

  inplaceEnabled = false;
  run_test();
  auto r1 = outputs[0].getDataCopy<float>();

  inplaceEnabled = true;
  run_test();
  auto r2 = outputs[0].getDataCopy<float>();

  BOOST_CHECK_EQUAL_COLLECTIONS(r1.begin(), r1.end(), r2.begin(), r2.end());
}
