// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE TiedGatherTests
#include <boost/test/unit_test.hpp>

#include <typeindex>

#include <popart/patterns/tiedgatherpattern.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/gather.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/transpose.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(TestPatternNamesContainsTiedGather) {
  BOOST_REQUIRE_NO_THROW(PatternNames::getName<TiedGatherPattern>());
  BOOST_REQUIRE_NO_THROW(PatternNames::getName<TiedGatherAccumulatePattern>());
}

BOOST_AUTO_TEST_CASE(TestPatternsEnabledDisabledApiWorks) {
  Patterns ps;

  // Off by default.

  BOOST_REQUIRE(!ps.isTiedGatherEnabled());
  BOOST_REQUIRE(!ps.isTiedGatherAccumulateEnabled());

  // Calling enable works correctly.

  ps.enableTiedGather(true);
  ps.enableTiedGatherAccumulate(true);

  BOOST_REQUIRE(ps.isTiedGatherEnabled());
  BOOST_REQUIRE(ps.isTiedGatherAccumulateEnabled());

  // Calling disable through other api works correctly.

  ps.enablePattern(std::type_index(typeid(TiedGatherPattern)), false);
  ps.enablePattern("TiedGatherAccumulate", false);

  BOOST_REQUIRE(!ps.isPatternEnabled("TiedGather"));
  BOOST_REQUIRE(!ps.isPatternEnabled(
      std::type_index(typeid(TiedGatherAccumulatePattern))));
}

namespace {

class TiedGatherTestCase {
public:
  TiedGatherTestCase(bool transposedW = false) {
    ir_.setExecutionMode(Ir::ExecutionMode::Inference);
    Graph &g = getGraph();

    Shape wShape = {4, 20};
    if (transposedW)
      std::reverse(wShape.begin(), wShape.end());
    TensorInfo wInfo{DataType::FLOAT, wShape};
    wData.resize(wInfo.nelms());
    g.getTensors().addVarInit(w, wInfo, static_cast<void *>(wData.data()));

    TensorInfo d0Info{DataType::INT32, {8}};
    d0Data.resize(d0Info.nelms());
    g.getTensors().addVarInit(d0, d0Info, static_cast<void *>(d0Data.data()));

    TensorInfo d1Info{DataType::FLOAT, {6, 20}};
    d1Data.resize(d1Info.nelms());
    g.getTensors().addVarInit(d1, d1Info, static_cast<void *>(d1Data.data()));
  }

  const TensorId w  = "w";
  const TensorId d0 = "d0";
  const TensorId d1 = "d1";

  Graph &getGraph() { return ir_.getMainGraph(); }

  void matches(Op *op, bool expected) {
    BOOST_CHECK(ps.matches(op) == expected);
  }

private:
  Ir ir_;
  TiedGatherPattern ps;
  std::vector<float> wData;
  std::vector<int32_t> d0Data;
  std::vector<float> d1Data;
};

std::pair<TensorId, Op *>
gather(Graph &g, TensorId data, TensorId indices, int64_t axis) {
  auto out = g.getIr().createIntermediateTensorId("o");
  return {out,
          g.createConnectedOp<GatherOp>({{GatherOp::dataInIndex(), data},
                                         {GatherOp::indicesInIndex(), indices}},
                                        {{GatherOp::outIndex(), out}},
                                        Onnx::AiOnnx::OpSet9::Gather,
                                        axis,
                                        Op::Settings(g, "gather"),
                                        nonstd::nullopt)};
}
std::pair<TensorId, Op *> matmul(Graph &g, TensorId lhs, TensorId rhs) {
  auto out = g.getIr().createIntermediateTensorId("o");
  return {out,
          g.createConnectedOp<MatMulOp>({{MatMulOp::getLhsInIndex(), lhs},
                                         {MatMulOp::getRhsInIndex(), rhs}},
                                        {{MatMulOp::getOutIndex(), out}},
                                        Onnx::AiOnnx::OpSet9::MatMul,
                                        Op::Settings(g, "matmul"),
                                        nonstd::nullopt,
                                        MatMulOp::SerialiseSettings(),
                                        OptionalDataType())};
}
std::pair<TensorId, Op *> transpose(Graph &g, TensorId in) {
  auto out = g.getIr().createIntermediateTensorId("o");
  return {out,
          g.createConnectedOp<TransposeOp>({{TransposeOp::getInIndex(), in}},
                                           {{TransposeOp::getOutIndex(), out}},
                                           Onnx::AiOnnx::OpSet9::Transpose,
                                           std::vector<int64_t>{1, 0},
                                           Op::Settings(g, "transpose"))};
}
} // namespace

BOOST_AUTO_TEST_CASE(TestPatternMatchTranposeMatMul) {
  TiedGatherTestCase tc(true);
  auto &g       = tc.getGraph();
  auto wt       = transpose(g, tc.w).first;
  auto tidAndOp = gather(g, tc.w, tc.d0, 0);
  auto gOut     = tidAndOp.first;
  auto gOp      = tidAndOp.second;
  matmul(g, gOut, wt);
  tc.matches(gOp, true);
}

BOOST_AUTO_TEST_CASE(TestPatternMatchTransposeGather) {
  TiedGatherTestCase tc(false);
  auto &g       = tc.getGraph();
  auto wt       = transpose(g, tc.w).first;
  auto tidAndOp = gather(g, wt, tc.d0, 0);
  auto gOut     = tidAndOp.first;
  auto gOp      = tidAndOp.second;
  matmul(g, gOut, tc.w);
  tc.matches(gOp, true);
}

BOOST_AUTO_TEST_CASE(TestPatternMisMatchAxis1) {
  TiedGatherTestCase tc(false);
  auto &g  = tc.getGraph();
  auto wt  = transpose(g, tc.w).first;
  auto gOp = gather(g, tc.w, tc.d0, 1).second;
  matmul(g, tc.d1, wt);
  tc.matches(gOp, false);
}

BOOST_AUTO_TEST_CASE(TestPatternMisMatchNoTranspose) {
  TiedGatherTestCase tc(true);
  auto &g  = tc.getGraph();
  auto gOp = gather(g, tc.w, tc.d0, 0).second;
  matmul(g, tc.d1, tc.w);
  tc.matches(gOp, false);
}

BOOST_AUTO_TEST_CASE(TestPatternMisMatchDoubleTranspose) {
  TiedGatherTestCase tc(false);
  auto &g  = tc.getGraph();
  auto wt  = transpose(g, tc.w).first;
  auto gOp = gather(g, wt, tc.d0, 0).second;
  matmul(g, tc.d1, wt);
  tc.matches(gOp, false);
}