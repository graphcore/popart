#define BOOST_TEST_MODULE NoGradOpTest

#include <boost/test/unit_test.hpp>

#include <memory>

#include <test_runner.hpp>

#include <popart/builder.hpp>
#include <popart/devicemanager.hpp>
#include <popart/logging.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/onnxutil.hpp>
#include <popart/op.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/opmanager.hpp>
#include <popart/optimizer.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/session.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

#include <popops/ElementWise.hpp>

using namespace popart;

namespace CustomOperators {
const OperatorIdentifier DontTrain = {"com.acme", "DontTrain", 1};
} // namespace CustomOperators

// An IdentityOp that doesn't return any grad ops.
class DontTrainOp : public Op {
public:
  DontTrainOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
      : Op(_opid, settings_) {}

  void setup() final { outInfo(0) = inInfo(0); }

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<DontTrainOp>(*this);
  }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

static OpCreator<DontTrainOp> donttrainOpCreator(CustomOperators::DontTrain);

class DontTrainOpx : public popx::Opx {
public:
  DontTrainOpx(Op *op, popx::Devicex *devicex) : popx::Opx(op, devicex) {
    verifyOp<DontTrainOp>(op, CustomOperators::DontTrain);
  }

  void grow(poplar::program::Sequence &prog) const final {
    insert(outId(0), cloneNcopy(prog, getInTensor(0)));
  }
};

static popx::OpxCreator<DontTrainOpx>
    donttrainOpxCreator(CustomOperators::DontTrain);

// a = conv(in, w0)
// b = conv(in, w1)
// c = donttrain(b)
// d = add(a, c)
BOOST_AUTO_TEST_CASE(Basic0) {
  auto nChans         = 1;
  auto batchSize      = 1;
  auto convHeightWith = 2;
  auto outHeightWidth = 2;

  TensorInfo inputInfo{
      "FLOAT",
      std::vector<int64_t>{batchSize, nChans, convHeightWith, convHeightWith}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  std::vector<float> convWeights0(nChans * nChans, 0);
  for (int i = 0; i < convWeights0.size(); i++) {
    convWeights0[i] = 0.25f;
  }
  std::vector<float> convWeights1 = convWeights0;

  TensorInfo convWeightInfo{"FLOAT",
                            std::vector<int64_t>{nChans, nChans, 1, 1}};

  ConstVoidData convData0{convWeights0.data(), convWeightInfo};
  ConstVoidData convData1{convWeights1.data(), convWeightInfo};

  TensorId conv1WeightsInput;

  TestRunner runner;
  runner.isTraining = true;

  runner.buildModel([&](auto &builder) {
    auto aiOnnx = builder.aiOnnxOpset9();

    auto input             = builder.addInputTensor(inputInfo);
    auto conv0WeightsInput = builder.addInitializedInputTensor(convData0);
    conv1WeightsInput      = builder.addInitializedInputTensor(convData1);

    auto ident0 = aiOnnx.identity({input});
    auto conv0  = aiOnnx.conv(
        {ident0, conv0WeightsInput}, {1, 1}, 1, {}, {0, 0, 0, 0}, {1, 1});
    auto conv1 = aiOnnx.conv(
        {ident0, conv1WeightsInput}, {1, 1}, 1, {}, {0, 0, 0, 0}, {1, 1});
    auto donttrain0 =
        builder.customOp(CustomOperators::DontTrain, 1, {conv0}, 1, {}).at(0);
    auto add0   = aiOnnx.add({donttrain0, conv1});
    auto output = aiOnnx.identity({add0});

    inputs.push_back(
        TestTensor::create<float>(input, {1, 2, 3, 4}, inputInfo.shape()));
    outputs.push_back(TestTensor::create<float>(output, inputInfo.shape()));

    runner.losses.push_back(
        new L1Loss(output, "l1LossVal", 0.1, ReductionType::SUM));

    return output;
  });

  runner.checkIr([&](Ir &ir) {
    // There should only be one SgdVarUpdateOp
    auto varUpdates = ir.opsOfType(Onnx::CustomOperators::SgdVarUpdate);
    BOOST_CHECK_EQUAL(varUpdates.size(), 1);
    // And it should be to update conv1WeightsInput
    BOOST_CHECK_EQUAL(
        varUpdates.at(0)->inId(VarUpdateOp::getVarToUpdateInIndex()),
        conv1WeightsInput);
  });

  runner.checkResult(
      [&](TestTensor &result) {
        auto data = result.getDataCopy<float>();
        logging::debug("result: {}", data);
      },
      inputs,
      outputs);
}

// a = conv(in, w)
// b = donttrain(a)
// c = add(a, b)
BOOST_AUTO_TEST_CASE(Basic1) {
  auto nChans         = 1;
  auto batchSize      = 1;
  auto convHeightWith = 2;
  auto outHeightWidth = 2;

  TensorInfo inputInfo{
      "FLOAT",
      std::vector<int64_t>{batchSize, nChans, convHeightWith, convHeightWith}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  std::vector<float> convWeights(nChans * nChans, 0);
  for (int i = 0; i < convWeights.size(); i++) {
    convWeights[i] = 0.25f;
  }

  TensorInfo convWeightInfo{"FLOAT",
                            std::vector<int64_t>{nChans, nChans, 1, 1}};

  ConstVoidData convData{convWeights.data(), convWeightInfo};

  TestRunner runner;
  runner.isTraining = true;

  TensorId convWeightsInput;

  runner.buildModel([&](auto &builder) {
    auto aiOnnx = builder.aiOnnxOpset9();

    auto input       = builder.addInputTensor(inputInfo);
    convWeightsInput = builder.addInitializedInputTensor(convData);

    auto ident0 = aiOnnx.identity({input});
    auto conv0  = aiOnnx.conv(
        {ident0, convWeightsInput}, {1, 1}, 1, {}, {0, 0, 0, 0}, {1, 1});
    auto donttrain0 =
        builder.customOp(CustomOperators::DontTrain, 1, {conv0}, 1, {}).at(0);
    auto add0   = aiOnnx.add({donttrain0, conv0});
    auto output = aiOnnx.identity({add0});

    inputs.push_back(
        TestTensor::create<float>(input, {1, 2, 3, 4}, inputInfo.shape()));
    outputs.push_back(TestTensor::create<float>(output, inputInfo.shape()));

    runner.losses.push_back(
        new L1Loss(output, "l1LossVal", 0.1, ReductionType::SUM));

    return output;
  });

  runner.checkIr([&](Ir &ir) {
    // There should only be a SgdVarUpdateOp
    auto varUpdates = ir.opsOfType(Onnx::CustomOperators::SgdVarUpdate);
    BOOST_CHECK_EQUAL(varUpdates.size(), 1);
    // And it should be to update conv1WeightsInput
    BOOST_CHECK_EQUAL(
        varUpdates.at(0)->inId(VarUpdateOp::getVarToUpdateInIndex()),
        convWeightsInput);
  });

  runner.checkResult(
      [&](TestTensor &result) {
        auto data = result.getDataCopy<float>();
        logging::debug("result: {}", data);
      },
      inputs,
      outputs);
}

// w0, w1 = split(w)
// a      = conv(in, w0)
// b      = donttrain(conv(in, w1))
// c      = add(a, b)
BOOST_AUTO_TEST_CASE(Basic2) {
  auto nChans         = 1;
  auto batchSize      = 1;
  auto convHeightWith = 2;
  auto outHeightWidth = 2;

  TensorInfo inputInfo{
      "FLOAT",
      std::vector<int64_t>{batchSize, nChans, convHeightWith, convHeightWith}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;

  TensorInfo convWeightInfo{"FLOAT",
                            std::vector<int64_t>{nChans * 2, nChans, 1, 1}};

  std::vector<float> convWeights(convWeightInfo.nelms(), 0);
  for (int i = 0; i < convWeights.size(); i++) {
    convWeights[i] = 0.25f;
  }

  ConstVoidData convData{convWeights.data(), convWeightInfo};

  TestRunner runner;
  runner.isTraining = true;

  TensorId convWeightsInput;

  runner.buildModel([&](auto &builder) {
    auto aiOnnx = builder.aiOnnxOpset9();

    auto input       = builder.addInputTensor(inputInfo);
    convWeightsInput = builder.addInitializedInputTensor(convData);

    auto splits = aiOnnx.split({convWeightsInput}, 2, 0, {nChans, nChans});
    auto cw0    = splits.at(0);
    auto cw1    = splits.at(1);

    auto conv0 = aiOnnx.conv({input, cw0}, {1, 1}, 1, {}, {0, 0, 0, 0}, {1, 1});

    auto conv1 = aiOnnx.conv({input, cw1}, {1, 1}, 1, {}, {0, 0, 0, 0}, {1, 1});
    auto donttrain0 =
        builder.customOp(CustomOperators::DontTrain, 1, {conv1}, 1, {}).at(0);

    auto output = aiOnnx.add({conv0, donttrain0});

    inputs.push_back(
        TestTensor::create<float>(input, {1, 2, 3, 4}, inputInfo.shape()));
    outputs.push_back(TestTensor::create<float>(output, inputInfo.shape()));

    runner.losses.push_back(
        new L1Loss(output, "l1LossVal", 0.1, ReductionType::SUM));

    return output;
  });

  runner.checkIr([&](Ir &ir) {
    // There should be no as a grad edge was not provided for every input of
    // splits grad op.
    auto varUpdates = ir.opsOfType(Onnx::CustomOperators::SgdVarUpdate);
    BOOST_CHECK_EQUAL(varUpdates.size(), 0);
  });

  runner.checkResult(
      [&](TestTensor &result) {
        auto data = result.getDataCopy<float>();
        logging::debug("result: {}", data);
      },
      inputs,
      outputs);
}
