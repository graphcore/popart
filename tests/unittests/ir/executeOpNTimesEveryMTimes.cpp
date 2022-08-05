// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_Ir_executeOpNTimesEveryMTimesTests

#include <algorithm>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <testdevice.hpp>
#include <utility>
#include <vector>
#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/add.hpp>
#include <popart/op/histogram.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/sin.hpp>
#include <popart/op/sinh.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/transforms/automaticlossscaling.hpp>

#include "popart/alias/aliasmodel.hpp"
#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/stepio.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensors.hpp"

namespace popart {
class IArray;
} // namespace popart

using namespace popart;

namespace {

template <typename Ex>
std::function<bool(const Ex &)>
checkErrorMsgHasPrefixFn(const std::string &prefix) {
  return [=](const Ex &ex) -> bool {
    return boost::algorithm::starts_with(ex.what(), prefix);
  };
}

} // namespace

BOOST_AUTO_TEST_CASE(TestFrequency1input1output2Ops) {
  // in -> sinhOp -> sinOp -> out
  auto ir  = std::make_unique<Ir>();
  Graph &g = ir->getMainGraph();

  const TensorId t0   = "t0";
  const TensorId tout = "tout";
  std::vector<float> t0Data(1, 1);
  const TensorInfo t0_ti{DataType::FLOAT, Shape{}};
  const TensorInfo tout_ti{DataType::FLOAT, Shape{}};

  g.getTensors().addVarInit(t0, t0_ti, t0Data.data());

  Op::Settings gSettings(g, "op", {});
  Op *sinhOp = g.createConnectedOp<SinhInplaceOp>(
      {{SinhInplaceOp::getInIndex(), t0}},
      {{SinhInplaceOp::getOutIndex(), ir->createIntermediateTensorId(t0)}},
      gSettings.copy("SinhInplaceOp"));

  Op::Settings gSettingsSin(g, "sin", {});
  Op *sinOp = g.createConnectedOp<SinOp>(
      {{SinOp::getInIndex(), sinhOp->outTensor(0)->id}},
      {{SinOp::getOutIndex(), tout}},
      Onnx::Operators::Sin_7,
      gSettingsSin.copy("SinOp"));
  sinOp->pruneable = false;

  std::map<InIndex, OutIndex> identityInputToOutputIndiciesMapping{{0, 0}};
  std::map<OutIndex, float> outputIndiciesAndValues;
  constexpr int bps = 18;
  ir->setDataFlow(DataFlow{bps, {{tout, AnchorReturnType("All")}}});
  AliasModel aliasModel;
  sinhOp = AutomaticLossScale::executeOpNTimesEveryMTimes(
      sinhOp,
      2,
      6,
      identityInputToOutputIndiciesMapping,
      outputIndiciesAndValues,
      aliasModel);

  ir->updateVertices();
  ir->setIsPrepared();

  const auto session = InferenceSession::createFromIr(
      std::move(ir), createTestDevice(TEST_TARGET));
  session->prepareDevice();

  const TensorInfo toutND_ti{DataType::FLOAT, Shape{bps * tout_ti.nelms()}};
  std::vector<float> outHost(bps * tout_ti.nelms());
  NDArrayWrapper<float> outWrapper(outHost.data(), toutND_ti);
  std::map<TensorId, IArray &> anchors = {{tout, outWrapper}};

  StepIO stepio({}, anchors);
  session->weightsFromHost();
  session->run(stepio);

  // n=2, m=6
  float v1 = 0.922767401;
  float v2 = 0.841470957;
  std::vector<float> expected{
      v1, v1, v2, v2, v2, v2, v1, v1, v2, v2, v2, v2, v1, v1, v2, v2, v2, v2};

  BOOST_TEST(expected.size() == outHost.size());
  BOOST_TEST(expected == outHost);
}

BOOST_AUTO_TEST_CASE(TestExecuteOpNTimesEveryMTimesShapes) {

  auto test = [](const std::map<InIndex, OutIndex>
                     &identityInputToOutputIndiciesMapping,
                 const std::map<OutIndex, float> &outputIndiciesAndValues,
                 float t0Value,
                 float tcValue,
                 Shape testShape,
                 const std::vector<float> &expected) {
    // t0    tc
    //  \   /
    //   Add
    //    |
    // IdentityOp
    //    |

    auto ir  = std::make_unique<Ir>();
    Graph &g = ir->getMainGraph();

    const TensorId t0   = "t0";
    const TensorId tc   = "tc";
    const TensorId tout = "tout";
    const TensorInfo t0_ti{DataType::FLOAT, testShape};
    const TensorInfo tc_ti{DataType::FLOAT, testShape};
    const TensorInfo tout_ti{DataType::FLOAT, testShape};
    std::vector<float> t0Data(t0_ti.nelms(), t0Value);

    g.getTensors().addVarInit(t0, t0_ti, t0Data.data());
    std::vector<float> constData(t0_ti.nelms(), tcValue);
    g.getTensors().addConstInit(tc, tc_ti, constData.data());

    Op::Settings gSettings(g, "op", {});
    Op *addOp = g.createConnectedOp<AddOp>(
        {{AddOp::getArg0InIndex(), t0}, {AddOp::getArg1InIndex(), tc}},
        {{AddOp::getOutIndex(), ir->createIntermediateTensorId(t0)}},
        Onnx::Operators::Add_7,
        gSettings.copy("Add"));

    Op::Settings gSettingsIdentityOp(g, "IdentityOp", {});
    Op *identityOp = g.createConnectedOp<IdentityOp>(
        {{IdentityOp::getInIndex(), addOp->outTensor(0)->id}},
        {{IdentityOp::getOutIndex(), tout}},
        Onnx::Operators::Identity_1,
        gSettingsIdentityOp.copy("IdentityOp"));

    identityOp->pruneable = false;

    constexpr int bps = 18;
    ir->setDataFlow(DataFlow{bps, {{tout, AnchorReturnType("All")}}});
    AliasModel aliasModel;
    addOp = AutomaticLossScale::executeOpNTimesEveryMTimes(
        addOp,
        2,
        6,
        identityInputToOutputIndiciesMapping,
        outputIndiciesAndValues,
        aliasModel);

    ir->updateVertices();
    ir->setIsPrepared();

    const auto session = InferenceSession::createFromIr(
        std::move(ir), createTestDevice(TEST_TARGET));
    session->prepareDevice();

    const TensorInfo toutND_ti{DataType::FLOAT, Shape{bps * t0_ti.nelms()}};
    std::vector<float> outHost(bps * t0_ti.nelms());
    NDArrayWrapper<float> outWrapper(outHost.data(), toutND_ti);
    std::map<TensorId, IArray &> anchors = {{tout, outWrapper}};

    StepIO stepio({}, anchors);
    session->weightsFromHost();
    session->run(stepio);

    // n=2, m=6
    BOOST_TEST(expected.size() == outHost.size());
    BOOST_TEST(expected == outHost);
  };

  // Check t0 is connected by nop to output of Add op in no compute subgraph.
  std::map<InIndex, OutIndex> identityInToOutIndicies1{{0, 0}};
  std::map<OutIndex, float> outputIndiciesAndValues1;
  float t0Value                = 1;
  float tcValue                = 5;
  std::vector<float> expected1 = {
      6, 6, 1, 1, 1, 1, 6, 6, 1, 1, 1, 1, 6, 6, 1, 1, 1, 1};
  test(identityInToOutIndicies1,
       outputIndiciesAndValues1,
       t0Value,
       tcValue,
       Shape{},
       expected1);

  // Check tc is connected by nop to output of Add op in no compute subgraph.
  std::map<InIndex, OutIndex> identityInToOutIndicies2{{1, 0}};
  std::map<OutIndex, float> outputIndiciesAndValues2;
  std::vector<float> expected2 = {
      6, 6, 5, 5, 5, 5, 6, 6, 5, 5, 5, 5, 6, 6, 5, 5, 5, 5};
  test(identityInToOutIndicies2,
       outputIndiciesAndValues2,
       t0Value,
       tcValue,
       Shape{},
       expected2);

  // Check outputIndiciesAndValues for empty subgraph.
  std::map<InIndex, OutIndex> identityInToOutIndiciesTest3;
  float outputValues = 11;
  std::map<OutIndex, float> outputIndiciesAndValues3{{0, outputValues}};
  std::vector<float> expected3 = {
      6, 6, 11, 11, 11, 11, 6, 6, 11, 11, 11, 11, 6, 6, 11, 11, 11, 11};
  test(identityInToOutIndiciesTest3,
       outputIndiciesAndValues3,
       t0Value,
       tcValue,
       Shape{},
       expected3);

  // Check outputIndiciesAndValues for empty subgraph and non scalar Shape.
  std::map<InIndex, OutIndex> identityInToOutIndicies4;
  float outputValues4 = 11;
  std::map<OutIndex, float> outputIndiciesAndValues4{{0, outputValues4}};
  std::vector<float> expected4 = {
      6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  11, 11, 11, 11, 11, 11,
      11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
      6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  11, 11, 11, 11, 11, 11,
      11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
      6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  11, 11, 11, 11, 11, 11,
      11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11};
  test(identityInToOutIndicies4,
       outputIndiciesAndValues4,
       t0Value,
       tcValue,
       Shape{2, 3},
       expected4);

  // Check t0 is connected by nop to output of Add op in no compute subgraph.
  std::map<InIndex, OutIndex> identityInToOutIndicies5{{0, 0}};
  std::map<OutIndex, float> outputIndiciesAndValues5;
  std::vector<float> expected5 = {
      6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 6, 6,
      6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  test(identityInToOutIndicies5,
       outputIndiciesAndValues5,
       t0Value,
       tcValue,
       Shape{2, 3},
       expected5);
}

BOOST_AUTO_TEST_CASE(
    TestExecuteOpNTimesEveryMTimes1input1outputDifferentTypes) {
  //    ts
  //    |
  // Histogram
  //    |
  // IdentityOp
  //    |

  auto ir  = std::make_unique<Ir>();
  Graph &g = ir->getMainGraph();

  const TensorId ts   = "ts";
  const TensorId tout = "tout";
  const TensorInfo ts_ti{DataType::FLOAT, Shape{2, 2}};
  const TensorInfo tout_ti{DataType::UINT32, Shape{2}};

  g.getTensors().addStream(ts, ts_ti);

  Op::Settings gSettings(g, "op", {});
  std::vector<float> levels = {0.5};
  bool absoluteOfInput      = true;

  Op *histogramOp = g.createConnectedOp<HistogramOp>(
      {{HistogramOp::getInIndex(), ts}},
      {{HistogramOp::getOutIndex(), ir->createIntermediateTensorId(ts)}},
      Onnx::CustomOperators::Histogram,
      levels,
      absoluteOfInput,
      gSettings.copy("HistogramOp"));

  Op::Settings gSettingsIdentityOp(g, "IdentityOp", {});
  Op *identityOp = g.createConnectedOp<IdentityOp>(
      {{IdentityOp::getInIndex(), histogramOp->outTensor(0)->id}},
      {{IdentityOp::getOutIndex(), tout}},
      Onnx::Operators::Identity_1,
      gSettingsIdentityOp.copy("IdentityOp"));

  identityOp->pruneable = false;

  std::map<InIndex, OutIndex> identityInputToOutputIndiciesMapping;
  std::map<OutIndex, float> outputIndiciesAndValues{{0, 11}};

  AliasModel aliasModel;
  histogramOp = AutomaticLossScale::executeOpNTimesEveryMTimes(
      histogramOp,
      2,
      6,
      identityInputToOutputIndiciesMapping,
      outputIndiciesAndValues,
      aliasModel);

  ir->updateVertices();
  ir->setIsPrepared();
  constexpr int bps = 20;
  ir->setDataFlow(DataFlow{bps, {{tout, AnchorReturnType("All")}}});

  const auto session = InferenceSession::createFromIr(
      std::move(ir), createTestDevice(TEST_TARGET));
  session->prepareDevice();

  const TensorInfo toutND_ti{DataType::UINT32, Shape{bps * tout_ti.nelms()}};
  std::vector<uint32_t> outHost(bps * tout_ti.nelms());
  NDArrayWrapper<uint32_t> outWrapper(outHost.data(), toutND_ti);
  std::map<TensorId, IArray &> anchors = {{tout, outWrapper}};

  std::vector<float> ts_init(bps * ts_ti.nelms());
  for (int i = 0; i < bps * ts_ti.nelms(); i++) {
    ts_init[i] = 2.0 * i;
  }

  const TensorInfo tsWraper_ti{DataType::FLOAT, Shape{bps, 2, 2}};
  popart::NDArrayWrapper<float> ts_wrapper(ts_init.data(), tsWraper_ti);
  std::map<popart::TensorId, popart::IArray &> inputs = {{ts, ts_wrapper}};

  StepIO stepio(inputs, anchors);
  session->weightsFromHost();
  session->run(stepio);

  // n=2, m=6
  std::vector<uint32_t> expected{1,  3,  0,  4,  11, 11, 11, 11, 11, 11,
                                 11, 11, 0,  4,  0,  4,  11, 11, 11, 11,
                                 11, 11, 11, 11, 0,  4,  0,  4,  11, 11,
                                 11, 11, 11, 11, 11, 11, 0,  4,  0,  4};

  BOOST_TEST(expected.size() == outHost.size());
  BOOST_TEST(expected == outHost);
}

template <typename T1>
void testType(
    int n,
    int m,
    const std::map<InIndex, OutIndex> &identityInputToOutputIndiciesMapping,
    const std::map<OutIndex, float> &outputIndiciesAndValues,
    T1 t0Value,
    T1 tcValue,
    const Shape &testShape,
    const std::vector<T1> &expected,
    DataType T2,
    bool enableGradientAccumulation,
    int bps,
    int accumulationFactor = 1) {

  // t0    tc
  //  \   /
  //   Add
  //    |
  // IdentityOp
  //    |

  auto ir  = std::make_unique<Ir>();
  Graph &g = ir->getMainGraph();

  SessionOptions &sessionOptions = ir->getSessionOptions();
  if (enableGradientAccumulation) {
    sessionOptions.enableGradientAccumulation = true;
    sessionOptions.accumulationFactor         = accumulationFactor;
  }

  const TensorId t0   = "t0";
  const TensorId tc   = "tc";
  const TensorId tout = "tout";
  TensorInfo t0_ti;
  t0_ti = TensorInfo(T2, testShape);
  const TensorInfo tc_ti{T2, testShape};
  const TensorInfo tout_ti{T2, testShape};
  std::vector<T1> t0Data(t0_ti.nelms(), t0Value);

  g.getTensors().addVarInit(t0, t0_ti, t0Data.data());
  std::vector<T1> constData(t0_ti.nelms(), tcValue);
  g.getTensors().addConstInit(tc, tc_ti, constData.data());

  Op::Settings gSettings(g, "op", {});
  Op *addOp = g.createConnectedOp<AddOp>(
      {{AddOp::getArg0InIndex(), t0}, {AddOp::getArg1InIndex(), tc}},
      {{AddOp::getOutIndex(), ir->createIntermediateTensorId(t0)}},
      Onnx::Operators::Add_7,
      gSettings.copy("Add"));

  Op::Settings gSettingsIdentityOp(g, "IdentityOp", {});
  Op *identityOp = g.createConnectedOp<IdentityOp>(
      {{IdentityOp::getInIndex(), addOp->outTensor(0)->id}},
      {{IdentityOp::getOutIndex(), tout}},
      Onnx::Operators::Identity_1,
      gSettingsIdentityOp.copy("IdentityOp"));

  identityOp->pruneable = false;

  ir->setDataFlow(DataFlow{bps, {{tout, AnchorReturnType("All")}}});

  AliasModel aliasModel;
  addOp = AutomaticLossScale::executeOpNTimesEveryMTimes(
      addOp,
      n,
      m,
      identityInputToOutputIndiciesMapping,
      outputIndiciesAndValues,
      aliasModel);

  ir->updateVertices();
  ir->setIsPrepared();

  const auto session = InferenceSession::createFromIr(
      std::move(ir), createTestDevice(TEST_TARGET));

  session->prepareDevice();

  const TensorInfo toutND_ti{T2, Shape{bps * t0_ti.nelms()}};
  std::vector<T1> outHost(bps * t0_ti.nelms());
  NDArrayWrapper<T1> outWrapper(outHost.data(), toutND_ti);
  std::map<TensorId, IArray &> anchors = {{tout, outWrapper}};

  StepIO stepio({}, anchors);
  session->weightsFromHost();
  session->run(stepio);

  BOOST_TEST(expected.size() == outHost.size());
  BOOST_TEST(expected == outHost);
}

BOOST_AUTO_TEST_CASE(TestExecuteOpNTimesEveryMTimesTypes) {

  std::map<InIndex, OutIndex> identityInToOutIndicies1{{0, 0}};
  std::map<OutIndex, float> outputIndiciesAndValues1;
  float t0Value1               = 1;
  float tcValue1               = 5;
  std::vector<float> expected1 = {
      6, 6, 1, 1, 1, 1, 6, 6, 1, 1, 1, 1, 6, 6, 1, 1, 1, 1};
  testType<float>(2,
                  6,
                  identityInToOutIndicies1,
                  outputIndiciesAndValues1,
                  t0Value1,
                  tcValue1,
                  Shape{},
                  expected1,
                  DataType::FLOAT,
                  false,
                  18);

  std::map<InIndex, OutIndex> identityInToOutIndicies2{{0, 0}};
  std::map<OutIndex, float> outputIndiciesAndValues2;
  int32_t t0Value2               = 1;
  int32_t tcValue2               = 5;
  std::vector<int32_t> expected2 = {
      6, 6, 1, 1, 1, 1, 6, 6, 1, 1, 1, 1, 6, 6, 1, 1, 1, 1};
  testType<int32_t>(2,
                    6,
                    identityInToOutIndicies2,
                    outputIndiciesAndValues2,
                    t0Value2,
                    tcValue2,
                    Shape{},
                    expected2,
                    DataType::INT32,
                    false,
                    18);

  std::map<InIndex, OutIndex> identityInToOutIndicies3;
  int32_t t0Value3    = 1;
  int32_t tcValue3    = 5;
  float outputValues3 = 11;
  std::map<OutIndex, float> outputIndiciesAndValues3{{0, outputValues3}};
  std::vector<int32_t> expected3 = {
      6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  11, 11, 11, 11, 11, 11,
      11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
      6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  11, 11, 11, 11, 11, 11,
      11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
      6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  11, 11, 11, 11, 11, 11,
      11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11};
  testType<int32_t>(2,
                    6,
                    identityInToOutIndicies3,
                    outputIndiciesAndValues3,
                    t0Value3,
                    tcValue3,
                    Shape{2, 3},
                    expected3,
                    DataType::INT32,
                    false,
                    18);

  // Check when identityInputToOutputIndiciesMapping and outputIndiciesAndValues
  // contain the same index.
  std::map<InIndex, OutIndex> identityInToOutIndicies4{{0, 0}};
  std::map<OutIndex, float> outputIndiciesAndValues4{{0, 11}};
  float t0Value4               = 1;
  float tcValue4               = 5;
  std::vector<float> expected4 = {
      6, 6, 1, 1, 1, 1, 6, 6, 1, 1, 1, 1, 6, 6, 1, 1, 1, 1};

  const auto checkErrorFn4 = checkErrorMsgHasPrefixFn<error>(
      "[AutomaticLossScale transform][[executeOpNTimesEveryMTimes] "
      "Incorrect frequency modifier settings for 100 (ai.onnx.Add:7) operator "
      "Identity output and default output can not have the same index 0.");

  BOOST_REQUIRE_EXCEPTION(testType<float>(2,
                                          6,
                                          identityInToOutIndicies4,
                                          outputIndiciesAndValues4,
                                          t0Value4,
                                          tcValue4,
                                          Shape{},
                                          expected4,
                                          DataType::FLOAT,
                                          false,
                                          18),
                          error,
                          checkErrorFn4);

  // Check when identityInputToOutputIndiciesMapping contain an invalid index.
  std::map<InIndex, OutIndex> identityInToOutIndicies5{{3, 0}};
  std::map<OutIndex, float> outputIndiciesAndValues5;
  float t0Value5               = 1;
  float tcValue5               = 5;
  std::vector<float> expected5 = {
      6, 6, 1, 1, 1, 1, 6, 6, 1, 1, 1, 1, 6, 6, 1, 1, 1, 1};

  const auto checkErrorFn5 = checkErrorMsgHasPrefixFn<error>(
      "[AutomaticLossScale transform][[executeOpNTimesEveryMTimes] "
      "identityInputToOutputIndiciesMapping has invalid input index 3 "
      "for operator 100 (ai.onnx.Add:7).");

  BOOST_REQUIRE_EXCEPTION(testType<float>(2,
                                          6,
                                          identityInToOutIndicies5,
                                          outputIndiciesAndValues5,
                                          t0Value5,
                                          tcValue5,
                                          Shape{},
                                          expected5,
                                          DataType::FLOAT,
                                          false,
                                          18),
                          error,
                          checkErrorFn5);

  // Check when identityInputToOutputIndiciesMapping contain an invalid index.
  std::map<InIndex, OutIndex> identityInToOutIndicies6{{1, 3}};
  std::map<OutIndex, float> outputIndiciesAndValues6;
  float t0Value6               = 1;
  float tcValue6               = 5;
  std::vector<float> expected6 = {
      6, 6, 1, 1, 1, 1, 6, 6, 1, 1, 1, 1, 6, 6, 1, 1, 1, 1};

  const auto checkErrorFn6 = checkErrorMsgHasPrefixFn<error>(
      "[AutomaticLossScale transform][[executeOpNTimesEveryMTimes] "
      "identityInputToOutputIndiciesMapping has invalid output index 3 "
      "for operator 100 (ai.onnx.Add:7).");

  BOOST_REQUIRE_EXCEPTION(testType<float>(2,
                                          6,
                                          identityInToOutIndicies6,
                                          outputIndiciesAndValues6,
                                          t0Value6,
                                          tcValue6,
                                          Shape{},
                                          expected6,
                                          DataType::FLOAT,
                                          false,
                                          18),
                          error,
                          checkErrorFn6);

  // Check when outputIndiciesAndValues contain an invalid index.
  std::map<InIndex, OutIndex> identityInToOutIndicies7;
  std::map<OutIndex, float> outputIndiciesAndValues7{{3, 11}};
  float t0Value7               = 1;
  float tcValue7               = 5;
  std::vector<float> expected7 = {
      6, 6, 1, 1, 1, 1, 6, 6, 1, 1, 1, 1, 6, 6, 1, 1, 1, 1};

  const auto checkErrorFn7 = checkErrorMsgHasPrefixFn<error>(
      "[AutomaticLossScale transform][[executeOpNTimesEveryMTimes] "
      "outputIndiciesAndValues has invalid output index 3 "
      "for operator 100 (ai.onnx.Add:7).");

  BOOST_REQUIRE_EXCEPTION(testType<float>(2,
                                          6,
                                          identityInToOutIndicies7,
                                          outputIndiciesAndValues7,
                                          t0Value7,
                                          tcValue7,
                                          Shape{},
                                          expected7,
                                          DataType::FLOAT,
                                          false,
                                          18),
                          error,
                          checkErrorFn7);

  // Check M.
  std::map<InIndex, OutIndex> identityInToOutIndicies8{{0, 0}};
  std::map<OutIndex, float> outputIndiciesAndValues8;
  float t0Value8               = 1;
  float tcValue8               = 5;
  std::vector<float> expected8 = {
      6, 6, 1, 1, 1, 1, 6, 6, 1, 1, 1, 1, 6, 6, 1, 1, 1, 1};

  const auto checkErrorFn8 = checkErrorMsgHasPrefixFn<error>(
      "[AutomaticLossScale transform][[executeOpNTimesEveryMTimes]. "
      "Argument M of executeOpNTimesEveryMTimes has inconsistent value 6. "
      "Operation 100 (ai.onnx.Add:7) is in the Normal execution context and "
      "gradient accumulation is enabled hence M should be a factor "
      "or multiple of gradient accumulation factor 10.");

  BOOST_REQUIRE_EXCEPTION(testType<float>(4,
                                          6,
                                          identityInToOutIndicies8,
                                          outputIndiciesAndValues8,
                                          t0Value8,
                                          tcValue8,
                                          Shape{},
                                          expected8,
                                          DataType::FLOAT,
                                          true,
                                          18,
                                          10),
                          error,
                          checkErrorFn8);
}
