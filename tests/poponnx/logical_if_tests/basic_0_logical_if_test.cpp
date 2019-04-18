#define BOOST_TEST_MODULE Basic0LogicalIf

#include <../test_runner.hpp>
#include <boost/test/unit_test.hpp>
#include <poponnx/builder.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/inputshapeinfo.hpp>
#include <poponnx/op/add.hpp>
#include <poponnx/op/if.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/tensornames.hpp>
#include <poponnx/tensors.hpp>

using namespace poponnx;

BOOST_AUTO_TEST_CASE(LogicalIf_basic0) {
  TensorInfo info{"FLOAT", std::vector<int64_t>{2, 2}};
  TensorInfo infoBool{"BOOL", std::vector<int64_t>{}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  TestRunner runner;
  runner.enableInPlace = false;

  runner.buildModel([&inputs, &outputs, info, infoBool](Builder &builder) {
    auto aiOnnx       = builder.aiOnnxOpset9();
    auto in0          = builder.addInputTensor(info);
    auto in1          = builder.addInputTensor(info);
    auto in_condition = builder.addInputTensor(infoBool);

    auto then_branch = [&info, in0, in1](Builder &parent_builder) {
      Builder &builder = parent_builder.createSubgraphBuilder();
      // auto builder = Builder::create();
      auto aiOnnx = builder.aiOnnxOpset9();
      builder.addInputTensorFromHigherScope(in0);
      builder.addInputTensorFromHigherScope(in1);
      auto out = aiOnnx.add({in0, in1});
      builder.addOutputTensor(out);
      return io::getModelFromString(builder.getModelProto()).graph();
    }(builder);

    auto else_branch = [&info, in0, in1](Builder &parent_builder) {
      Builder &builder = parent_builder.createSubgraphBuilder();
      // auto builder = Builder::create();
      auto aiOnnx = builder.aiOnnxOpset9();
      builder.addInputTensorFromHigherScope(in0);
      builder.addInputTensorFromHigherScope(in1);
      auto out = aiOnnx.sub({in0, in1});
      builder.addOutputTensor(out);
      return io::getModelFromString(builder.getModelProto()).graph();
    }(builder);

    auto out =
        aiOnnx.logical_if({in_condition}, 1, else_branch, then_branch)[0];
    builder.addOutputTensor(out);

    inputs.push_back(
        TestTensor::create<float>(in0, {1, 2, 3, 4}, info.shape()));
    inputs.push_back(
        TestTensor::create<float>(in1, {2, 3, 4, 5}, info.shape()));
    inputs.push_back(TestTensor::create<bool>(in_condition, infoBool.shape()));
    outputs.push_back(TestTensor::create<float>(out, info.shape()));

    return out;
  });

  // Check true branch
  inputs.back().setData<char>({1});
  runner.checkResult(
      [](TestTensor &result) {
        auto data = result.getDataCopy<float>();
        std::vector<float> expected{3, 5, 7, 9};
        BOOST_CHECK(data == expected);
      },
      inputs,
      outputs);

  // Check false branch
  inputs.back().setData<char>({0});
  runner.checkResult(
      [](TestTensor &result) {
        auto data = result.getDataCopy<float>();
        std::vector<float> expected{-1, -1, -1, -1};
        BOOST_CHECK(data == expected);
      },
      inputs,
      outputs);
}

BOOST_AUTO_TEST_CASE(LogicalIf_scopes0) {
  TensorInfo info{"FLOAT", std::vector<int64_t>{2, 2}};
  TensorInfo infoBool{"BOOL", std::vector<int64_t>{}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  TestRunner runner;
  runner.enableInPlace = false;

  runner.buildModel([&inputs, &outputs, info, infoBool](Builder &builder) {
    auto aiOnnx       = builder.aiOnnxOpset9();
    auto in0          = builder.addInputTensor(info);
    auto in1          = builder.addInputTensor(info);
    auto in_condition = builder.addInputTensor(infoBool);

    auto then_branch = [&info, in0, in1](Builder &parent_builder) {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromHigherScope(in0);
      builder.addInputTensorFromHigherScope(in1);
      auto out = aiOnnx.add({in0, in1});
      builder.addOutputTensor(out);
      return io::getModelFromString(builder.getModelProto()).graph();
    }(builder);

    auto else_branch = [&info, in0, in1](Builder &parent_builder) {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromHigherScope(in0);
      builder.addInputTensorFromHigherScope(in1);
      auto out = aiOnnx.add({in0, in1});
      builder.addOutputTensor(out);
      return io::getModelFromString(builder.getModelProto()).graph();
    }(builder);

    auto out =
        aiOnnx.logical_if({in_condition}, 1, else_branch, then_branch)[0];
    builder.addOutputTensor(out);

    inputs.push_back(
        TestTensor::create<float>(in0, {1, 2, 3, 4}, info.shape()));
    inputs.push_back(
        TestTensor::create<float>(in1, {2, 3, 4, 5}, info.shape()));
    inputs.push_back(TestTensor::create<bool>(in_condition, infoBool.shape()));
    outputs.push_back(TestTensor::create<float>(out, info.shape()));

    return out;
  });

  // Check true branch
  inputs.back().setData<char>({1});
  runner.checkResult(
      [](TestTensor &result) {
        auto data = result.getDataCopy<float>();
        std::vector<float> expected{3, 5, 7, 9};
        logging::debug("data: {}", data);
        logging::debug("expected: {}", expected);
        BOOST_CHECK(data == expected);
      },
      inputs,
      outputs);

  // Check false branch
  inputs.back().setData<char>({0});
  runner.checkResult(
      [](TestTensor &result) {
        auto data = result.getDataCopy<float>();
        std::vector<float> expected{3, 5, 7, 9};
        logging::debug("data: {}", data);
        logging::debug("expected: {}", expected);
        BOOST_CHECK(data == expected);
      },
      inputs,
      outputs);
}

BOOST_AUTO_TEST_CASE(LogicalIf_scopes1) {
  TensorInfo info{"FLOAT", std::vector<int64_t>{2, 2}};
  TensorInfo infoBool{"BOOL", std::vector<int64_t>{}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  TestRunner runner;
  runner.enableInPlace = false;

  runner.buildModel([&inputs, &outputs, info, infoBool](Builder &builder) {
    auto aiOnnx       = builder.aiOnnxOpset9();
    auto in0          = builder.addInputTensor(info);
    auto in1          = builder.addInputTensor(info);
    auto in_condition = builder.addInputTensor(infoBool);

    auto then_branch = [&info, in0, in1](Builder &parent_builder) {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromHigherScope(in0);
      builder.addInputTensorFromHigherScope(in1);
      auto out = aiOnnx.add({in0, in1});
      out      = aiOnnx.add({out, out});
      builder.addOutputTensor(out);
      return io::getModelFromString(builder.getModelProto()).graph();
    }(builder);

    auto else_branch = [&info, in0, in1](Builder &parent_builder) {
      Builder &builder = parent_builder.createSubgraphBuilder();
      auto aiOnnx      = builder.aiOnnxOpset9();
      builder.addInputTensorFromHigherScope(in0);
      builder.addInputTensorFromHigherScope(in1);
      auto out = aiOnnx.add({in0, in1});
      builder.addOutputTensor(out);
      return io::getModelFromString(builder.getModelProto()).graph();
    }(builder);

    auto out =
        aiOnnx.logical_if({in_condition}, 1, else_branch, then_branch)[0];
    builder.addOutputTensor(out);

    inputs.push_back(
        TestTensor::create<float>(in0, {1, 2, 3, 4}, info.shape()));
    inputs.push_back(
        TestTensor::create<float>(in1, {2, 3, 4, 5}, info.shape()));
    inputs.push_back(TestTensor::create<bool>(in_condition, infoBool.shape()));
    outputs.push_back(TestTensor::create<float>(out, info.shape()));

    return out;
  });

  // Check true branch
  inputs.back().setData<char>({1});
  runner.checkResult(
      [](TestTensor &result) {
        auto data = result.getDataCopy<float>();
        std::vector<float> expected{6, 10, 14, 18};
        logging::debug("data: {}", data);
        logging::debug("expected: {}", expected);
        BOOST_CHECK(data == expected);
      },
      inputs,
      outputs);

  // Check false branch
  inputs.back().setData<char>({0});
  runner.checkResult(
      [](TestTensor &result) {
        auto data = result.getDataCopy<float>();
        std::vector<float> expected{3, 5, 7, 9};
        logging::debug("data: {}", data);
        logging::debug("expected: {}", expected);
        BOOST_CHECK(data == expected);
      },
      inputs,
      outputs);
}

// The origional model resembles the below python code:
//
// in0 = np.array([1, 2, 3, 4])
// in1 = np.array([2, 3, 4, 5])
//
// t0 = in0 + in1
// t1 = in0 + t0
//
// if condition1:
//     if condition2:
//         out = in0 + t1
//     else:
//         out = in1 + t1
// else:
//     out = in0 - t0
//
// After calling extend scopes, `t1` should be in the scope of the first if:
//
// in0 = np.array([1, 2, 3, 4])
// in1 = np.array([2, 3, 4, 5])
//
// t0 = in0 + in1
//
// if condition1:
//     t1 = in0 + t0
//     if condition2:
//         out = in0 + t1
//     else:
//         out = in1 + t1
// else:
//     out = in0 - t0
BOOST_AUTO_TEST_CASE(LogicalIf_scopes2) {
  TensorInfo info{"FLOAT", std::vector<int64_t>{2, 2}};
  TensorInfo infoBool{"BOOL", std::vector<int64_t>{}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  TestRunner runner;
  runner.enableInPlace = false;

  TensorId if_out;

  auto create_add_model = [](Builder &parent_builder,
                             TensorInfo &lhs_info,
                             std::string &lhs,
                             TensorInfo &rhs_info,
                             std::string &rhs) {
    auto &builder = parent_builder.createSubgraphBuilder();
    auto aiOnnx   = builder.aiOnnxOpset9();
    builder.addInputTensorFromHigherScope(lhs);
    builder.addInputTensorFromHigherScope(rhs);
    auto out = aiOnnx.add({lhs, rhs});
    builder.addOutputTensor(out);
    return io::getModelFromString(builder.getModelProto()).graph();
  };

  auto create_sub_model = [](Builder &parent_builder,
                             TensorInfo &lhs_info,
                             std::string &lhs,
                             TensorInfo &rhs_info,
                             std::string &rhs) {
    auto &builder = parent_builder.createSubgraphBuilder();
    auto aiOnnx   = builder.aiOnnxOpset9();
    builder.addInputTensorFromHigherScope(lhs);
    builder.addInputTensorFromHigherScope(rhs);
    auto out = aiOnnx.sub({lhs, rhs});
    builder.addOutputTensor(out);
    return io::getModelFromString(builder.getModelProto()).graph();
  };

  TensorId t0, t1;

  runner.buildModel([&](Builder &builder) {
    auto aiOnnx = builder.aiOnnxOpset9();
    auto in0    = builder.addInputTensor(info);
    auto in1    = builder.addInputTensor(info);
    auto in2    = builder.addInputTensor(infoBool);
    auto in3    = builder.addInputTensor(infoBool);

    t0 = aiOnnx.add({in0, in1});
    t1 = aiOnnx.add({in0, t0});

    auto then_branch = [&]() {
      auto &sub_builder = builder.createSubgraphBuilder();
      auto aiOnnx       = sub_builder.aiOnnxOpset9();
      sub_builder.addInputTensorFromHigherScope(in0);
      sub_builder.addInputTensorFromHigherScope(in1);
      sub_builder.addInputTensorFromHigherScope(in3);
      sub_builder.addInputTensorFromHigherScope(t1);

      auto then_branch = create_add_model(sub_builder, info, in0, info, t1);
      auto else_branch = create_add_model(sub_builder, info, in1, info, t1);

      auto out = aiOnnx.logical_if({in3}, 1, else_branch, then_branch)[0];

      sub_builder.addOutputTensor(out);
      return io::getModelFromString(sub_builder.getModelProto()).graph();
    }();

    auto else_branch = create_sub_model(builder, info, in0, info, t0);

    if_out =
        aiOnnx.logical_if({in2}, 1, else_branch, then_branch, "TestIfOp")[0];
    builder.addOutputTensor(if_out);

    inputs.push_back(
        TestTensor::create<float>(in0, {1, 2, 3, 4}, info.shape()));
    inputs.push_back(
        TestTensor::create<float>(in1, {2, 3, 4, 5}, info.shape()));
    inputs.push_back(TestTensor::create<bool>(in2, infoBool.shape()));
    inputs.push_back(TestTensor::create<bool>(in3, infoBool.shape()));
    outputs.push_back(TestTensor::create<float>(if_out, info.shape()));

    return if_out;
  });

  runner.checkIr([&](Ir &ir) {
    auto t0_tensor = ir.getTensors().get(t0);
    auto t1_tensor = ir.getTensors().get(t1);

    auto t0_producer = t0_tensor->getProducer();
    auto t1_producer = t1_tensor->getProducer();

    BOOST_CHECK(t0_producer->getScope().depth() == 0);
    BOOST_CHECK(t1_producer->getScope().depth() == 1);
  });
}

// The origional model resembles the below python code:
//
// in0 = np.array([1, 2, 3, 4])
//
// t0 = in0 + in0
//
// if condition1:
//     t1 = t0 + t0
//     if condition2:
//         out = t1 + t1
//     else:
//         out = t0 - t0
// else:
//     out = in0 - in0
//
// After calling extend scopes, `t1` should be in the scope of the first if:
//
// in0 = np.array([1, 2, 3, 4])
//
// if condition1:
//     t0 = in0 + in0
//     if condition2:
//         t1 = t0 + t0
//         out = t1 + t1
//     else:
//         out = t0 - t0
// else:
//     out = in0 - in0
//
BOOST_AUTO_TEST_CASE(LogicalIf_scopes3) {
  TensorInfo info{"FLOAT", std::vector<int64_t>{2, 2}};
  TensorInfo infoBool{"BOOL", std::vector<int64_t>{}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  TestRunner runner;
  runner.enableInPlace = false;

  TensorId if_out;

  TensorId in0;

  runner.buildModel([&](Builder &builder) {
    auto aiOnnx = builder.aiOnnxOpset9();
    auto in0    = builder.addInputTensor(info);
    auto c0     = builder.addInputTensor(infoBool);
    auto c1     = builder.addInputTensor(infoBool);

    auto t0 = aiOnnx.add({in0, in0});

    auto then_branch = [&]() {
      auto &sub_builder = builder.createSubgraphBuilder();
      auto aiOnnx       = sub_builder.aiOnnxOpset9();
      sub_builder.addInputTensorFromHigherScope(t0);
      sub_builder.addInputTensorFromHigherScope(c1);

      auto t1 = aiOnnx.add({t0, t0});

      auto then_branch = [&]() {
        auto &sub_sub_builder = sub_builder.createSubgraphBuilder();
        auto aiOnnx           = sub_sub_builder.aiOnnxOpset9();
        sub_sub_builder.addInputTensorFromHigherScope(t1);
        auto out = aiOnnx.add({t1, t1});
        sub_sub_builder.addOutputTensor(out);
        return io::getModelFromString(sub_sub_builder.getModelProto()).graph();
      }();

      auto else_branch = [&]() {
        auto &sub_sub_builder = sub_builder.createSubgraphBuilder();
        auto aiOnnx           = sub_sub_builder.aiOnnxOpset9();
        sub_sub_builder.addInputTensorFromHigherScope(t0);
        auto out = aiOnnx.sub({t0, t0});
        sub_sub_builder.addOutputTensor(out);
        return io::getModelFromString(sub_sub_builder.getModelProto()).graph();
      }();

      auto out = aiOnnx.logical_if({c1}, 1, else_branch, then_branch)[0];

      sub_builder.addOutputTensor(out);
      return io::getModelFromString(sub_builder.getModelProto()).graph();
    }();

    auto else_branch = [&]() {
      auto &sub_builder = builder.createSubgraphBuilder();
      auto aiOnnx       = sub_builder.aiOnnxOpset9();
      sub_builder.addInputTensorFromHigherScope(in0);
      auto out = aiOnnx.sub({in0, in0});
      sub_builder.addOutputTensor(out);
      return io::getModelFromString(sub_builder.getModelProto()).graph();
    }();

    if_out = aiOnnx.logical_if({c0}, 1, else_branch, then_branch)[0];
    builder.addOutputTensor(if_out);

    inputs.push_back(
        TestTensor::create<float>(in0, {1, 2, 3, 4}, info.shape()));
    inputs.push_back(TestTensor::create<bool>(c0, infoBool.shape()));
    inputs.push_back(TestTensor::create<bool>(c1, infoBool.shape()));
    outputs.push_back(TestTensor::create<float>(if_out, info.shape()));

    return if_out;
  });

  runner.checkIr([&](Ir &ir) {
    // There should be 3 AddOps
    std::vector<Op *> add_ops;
    for (auto &id_op : ir.getOps()) {
      auto op = id_op.second.get();
      if (op->isConvertibleTo<AddOp>()) {
        add_ops.push_back(op);
      }
    }
    BOOST_CHECK(add_ops.size() == 3);

    // Two AddOps should have depth 2
    // One AddOp should have depth 1
    int total_depth = 0;
    for (auto op : add_ops) {
      auto depth = op->getScope().depth();
      total_depth += depth;
      BOOST_CHECK(depth == 1 || depth == 2);
    };
    BOOST_CHECK(total_depth = 5);
  });
}
