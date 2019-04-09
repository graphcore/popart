#define BOOST_TEST_MODULE Add0InplaceTest

#include <boost/test/unit_test.hpp>
#include <inplace_util.hpp>
#include <poponnx/op/add.hpp>
#include <poponnx/tensors.hpp>

using namespace poponnx;

// Basic case where arg0 and arg1 have matching shape
// (2x2) + (2x2)
BOOST_AUTO_TEST_CASE(Inplace_add0) {
  TensorInfo info{"FLOAT", std::vector<int64_t>{2, 2}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  auto buildModel = [&](Builder &builder) {
    auto aiOnnx = builder.aiOnnxOpset9();
    auto in0    = builder.addInputTensor(info);
    auto in1    = builder.addInputTensor(info);
    auto i0     = aiOnnx.identity({in0});
    auto i1     = aiOnnx.identity({in1});
    auto inAdd  = aiOnnx.add({i0, i1});
    auto out    = aiOnnx.identity({inAdd});
    builder.addOutputTensor(out);

    inputs.push_back(
        TestTensor::create<float>(in0, {1, 2, 3, 4}, info.shape()));
    inputs.push_back(
        TestTensor::create<float>(in1, {2, 3, 4, 5}, info.shape()));
    outputs.push_back(TestTensor::create<float>(out, info.shape()));

    return out;
  };

  auto checkIr = [](Ir &ir) {
    BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Add).size() == 0);
    auto lhs_op_count =
        ir.opsOfType(Onnx::CustomOperators::AddLhsInplace).size();
    auto rhs_op_count =
        ir.opsOfType(Onnx::CustomOperators::AddRhsInplace).size();
    BOOST_CHECK(lhs_op_count == 1 || rhs_op_count == 1);
  };

  auto checkResult = [](TestTensor &result) {
    auto data = result.getDataCopy<float>();
    std::vector<float> expected{3, 5, 7, 9};
    BOOST_CHECK(data == expected);
  };

  TestRunner runner;
  runner.enableInPlace = true;
  runner.buildModel(buildModel);
  runner.checkIr(checkIr);
  runner.checkResult(checkResult, inputs, outputs);
}

// Arg0 is larger than arg1
// (2x2) + (1x2)
BOOST_AUTO_TEST_CASE(Inplace_add1) {
  TensorInfo info0{"FLOAT", std::vector<int64_t>{2, 2}};
  TensorInfo info1{"FLOAT", std::vector<int64_t>{1, 2}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  auto buildModel = [&](Builder &builder) {
    auto aiOnnx = builder.aiOnnxOpset9();
    auto in0    = builder.addInputTensor(info0);
    auto in1    = builder.addInputTensor(info1);
    auto i0     = aiOnnx.identity({in0});
    auto i1     = aiOnnx.identity({in1});
    auto inAdd  = aiOnnx.add({i0, i1});
    auto out    = aiOnnx.identity({inAdd});
    builder.addOutputTensor(out);

    inputs.push_back(
        TestTensor::create<float>(in0, {1, 2, 3, 4}, info0.shape()));
    inputs.push_back(TestTensor::create<float>(in1, {2, 3}, info1.shape()));
    outputs.push_back(
        TestTensor::create<float>(out, npOut(info0.shape(), info1.shape())));

    return out;
  };

  auto checkIr = [](Ir &ir) {
    BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Add).size() == 0);
    BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::AddLhsInplace).size() == 1);
  };

  auto checkResult = [](TestTensor &result) {
    auto data = result.getDataCopy<float>();
    std::vector<float> expected{3, 5, 5, 7};
    BOOST_CHECK(data == expected);
  };

  TestRunner runner;
  runner.enableInPlace = true;
  runner.buildModel(buildModel);
  runner.checkIr(checkIr);
  runner.checkResult(checkResult, inputs, outputs);
}

// Arg1 is larger than arg0
// (1x2) + (2x2)
BOOST_AUTO_TEST_CASE(Inplace_add2) {
  TensorInfo info0{"FLOAT", std::vector<int64_t>{1, 2}};
  TensorInfo info1{"FLOAT", std::vector<int64_t>{2, 2}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  auto buildModel = [&](Builder &builder) {
    auto aiOnnx = builder.aiOnnxOpset9();
    auto in0    = builder.addInputTensor(info0);
    auto in1    = builder.addInputTensor(info1);
    auto i0     = aiOnnx.identity({in0});
    auto i1     = aiOnnx.identity({in1});
    auto inAdd  = aiOnnx.add({i0, i1});
    auto out    = aiOnnx.identity({inAdd});
    builder.addOutputTensor(out);

    inputs.push_back(TestTensor::create<float>(in0, {2, 3}, info0.shape()));
    inputs.push_back(
        TestTensor::create<float>(in1, {1, 2, 3, 4}, info1.shape()));
    outputs.push_back(
        TestTensor::create<float>(out, npOut(info0.shape(), info1.shape())));

    return out;
  };

  auto checkIr = [](Ir &ir) {
    BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Add).size() == 0);
    BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::AddRhsInplace).size() == 1);
  };

  auto checkResult = [](TestTensor &result) {
    auto data = result.getDataCopy<float>();
    std::vector<float> expected{3, 5, 5, 7};
    BOOST_CHECK(data == expected);
  };

  TestRunner runner;
  runner.enableInPlace = true;
  runner.buildModel(buildModel);
  runner.checkIr(checkIr);
  runner.checkResult(checkResult, inputs, outputs);
}

// Arg0 and arg1 are of different ranks
// (2x2) + (2)
BOOST_AUTO_TEST_CASE(Inplace_add3) {
  TensorInfo info0{"FLOAT", std::vector<int64_t>{2, 2}};
  TensorInfo info1{"FLOAT", std::vector<int64_t>{2}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  auto buildModel = [&](Builder &builder) {
    auto aiOnnx = builder.aiOnnxOpset9();
    auto in0    = builder.addInputTensor(info0);
    auto in1    = builder.addInputTensor(info1);
    auto i0     = aiOnnx.identity({in0});
    auto i1     = aiOnnx.identity({in1});
    auto inAdd  = aiOnnx.add({i0, i1});
    auto out    = aiOnnx.identity({inAdd});
    builder.addOutputTensor(out);

    inputs.push_back(
        TestTensor::create<float>(in0, {1, 2, 3, 4}, info0.shape()));
    inputs.push_back(TestTensor::create<float>(in1, {2, 3}, info1.shape()));
    outputs.push_back(TestTensor::create<float>(
        out, npOut(info0.shape(), npOut(info0.shape(), info1.shape()))));

    return out;
  };

  auto checkIr = [](Ir &ir) {
    BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Add).size() == 0);
    BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::AddLhsInplace).size() == 1);
  };

  auto checkResult = [](TestTensor &result) {
    auto data = result.getDataCopy<float>();
    std::vector<float> expected{3, 5, 5, 7};
    BOOST_CHECK(data == expected);
  };

  TestRunner runner;
  runner.enableInPlace = true;
  runner.buildModel(buildModel);
  runner.checkIr(checkIr);
  runner.checkResult(checkResult, inputs, outputs);
}

// Arg0 and arg1 are of different ranks
// (2) + (2x2)
BOOST_AUTO_TEST_CASE(Inplace_add4) {
  TensorInfo info0{"FLOAT", std::vector<int64_t>{2}};
  TensorInfo info1{"FLOAT", std::vector<int64_t>{2, 2}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  auto buildModel = [&](Builder &builder) {
    auto aiOnnx = builder.aiOnnxOpset9();
    auto in0    = builder.addInputTensor(info0);
    auto in1    = builder.addInputTensor(info1);
    auto i0     = aiOnnx.identity({in0});
    auto i1     = aiOnnx.identity({in1});
    auto inAdd  = aiOnnx.add({i0, i1});
    auto out    = aiOnnx.identity({inAdd});
    builder.addOutputTensor(out);

    inputs.push_back(TestTensor::create<float>(in0, {1, 2}, info0.shape()));
    inputs.push_back(
        TestTensor::create<float>(in1, {1, 2, 3, 4}, info1.shape()));
    outputs.push_back(TestTensor::create<float>(
        out, npOut(info0.shape(), npOut(info0.shape(), info1.shape()))));

    return out;
  };

  auto checkIr = [](Ir &ir) {
    BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Add).size() == 0);
    BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::AddRhsInplace).size() == 1);
  };

  auto checkResult = [](TestTensor &result) {
    auto data = result.getDataCopy<float>();
    std::vector<float> expected{2, 4, 4, 6};
    logging::debug("data: {}", data);
    BOOST_CHECK(data == expected);
  };

  TestRunner runner;
  runner.enableInPlace = true;
  runner.buildModel(buildModel);
  runner.checkIr(checkIr);
  runner.checkResult(checkResult, inputs, outputs);
}

// Checking AddOp fwdRegMap
BOOST_AUTO_TEST_CASE(Add_fwdRegMap0) {
  TensorInfo info0{"FLOAT", std::vector<int64_t>{4, 2, 4}};
  TensorInfo info1{"FLOAT", std::vector<int64_t>{1, 4}};
  std::string addOut;

  auto buildModel = [&](Builder &builder) {
    auto aiOnnx = builder.aiOnnxOpset9();
    auto in0    = builder.addInputTensor(info0);
    auto in1    = builder.addInputTensor(info1);
    addOut      = aiOnnx.add({in0, in1});
    builder.addOutputTensor(addOut);

    return addOut;
  };

  auto checkIr = [&](Ir &ir) {
    auto tensor = ir.getTensors().get(addOut);
    auto addOp  = tensor->getProducer();

    auto arg0Map = addOp->fwdRegMap(AddOp::getArg0InIndex());
    auto arg1Map = addOp->fwdRegMap(AddOp::getArg1InIndex());

    // dim 0 and 1 broadcast
    auto r                = arg1Map({{0, 0}, {1, 2}});
    view::Region expected = {{0, 0, 0}, {4, 2, 2}};
    BOOST_CHECK(r == expected);
  };

  TestRunner runner;
  runner.enableInPlace = false;
  runner.buildModel(buildModel);
  runner.checkIr(checkIr);
}

// Checking AddOp fwdRegMap
BOOST_AUTO_TEST_CASE(Add_bwdRegMap0) {
  TensorInfo info0{"FLOAT", std::vector<int64_t>{5, 7}};
  TensorInfo info1{"FLOAT", std::vector<int64_t>{5, 1}};
  std::string addOut;

  auto buildModel = [&](Builder &builder) {
    auto aiOnnx = builder.aiOnnxOpset9();
    auto in0    = builder.addInputTensor(info0);
    auto in1    = builder.addInputTensor(info1);
    addOut      = aiOnnx.add({in0, in1});
    builder.addOutputTensor(addOut);

    return addOut;
  };

  auto checkIr = [&](Ir &ir) {
    auto tensor = ir.getTensors().get(addOut);
    auto addOp  = tensor->getProducer();

    auto arg1Map = addOp->bwdRegMap(AddOp::getArg1InIndex());

    // dim 0 and 1 broadcast
    auto r                = arg1Map({{1, 4}, {3, 6}});
    view::Region expected = {{1, 0}, {3, 1}};
    BOOST_CHECK(r == expected);
  };

  TestRunner runner;
  runner.enableInPlace = false;
  runner.buildModel(buildModel);
  runner.checkIr(checkIr);
}
