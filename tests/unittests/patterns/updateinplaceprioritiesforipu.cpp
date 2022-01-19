// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE TestPatternsUpdateInplacePrioritiesForIpu
#include <boost/test/unit_test.hpp>

#include <popart/patterns/updateinplaceprioritiesforipu.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/add.hpp>
#include <popart/op/conv.hpp>
#include <popart/op/dropout.hpp>
#include <popart/op/fmod.hpp>
#include <popart/op/groupnorm.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/reshape.hpp>
#include <popart/operators.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensors.hpp>

#include <typeindex>

using namespace popart;

BOOST_AUTO_TEST_CASE(TestPatternNamesContainsUpdateInplacePrioritiesForIpu) {
  BOOST_REQUIRE_NO_THROW(
      PatternNames::getName<UpdateInplacePrioritiesForIpu>());
}

BOOST_AUTO_TEST_CASE(TestPatternsEnableDisableApi) {
  Patterns ps;

  // Off by default.
  BOOST_REQUIRE(!ps.isUpdateInplacePrioritiesForIpuEnabled());

  // Calling enable works correctly.
  ps.enablePattern("UpdateInplacePrioritiesForIpu", true);
  BOOST_REQUIRE(ps.isUpdateInplacePrioritiesForIpuEnabled());

  // Calling disable (through another api) works correctly.
  ps.enableUpdateInplacePrioritiesForIpu(false);
  BOOST_REQUIRE(!ps.isUpdateInplacePrioritiesForIpuEnabled());
}

BOOST_AUTO_TEST_CASE(TestDoesNothingToNonAddOp) {
  Ir ir;
  Graph &g = ir.getMainGraph();

  /*
    a, b = const
    d = var
    c = matmul(a, b)
    e = fmod(c, d)

    Apply pattern to fmod, check priorities unchanged.
   */

  const TensorId a = "a";
  const TensorId b = "b";
  const TensorId c = "c";
  const TensorId d = "d";
  const TensorId e = "e";

  const TensorInfo ti{DataType::FLOAT, Shape{2, 2}};
  const std::vector<float> someData(ti.nelms());

  g.getTensors().addConstInit(a, ti, someData.data(), "a");
  g.getTensors().addConstInit(b, ti, someData.data(), "b");
  g.getTensors().addVarInit(d, ti, someData.data(), "d");

  g.createConnectedOp<MatMulOp>(
      {{MatMulOp::getLhsInIndex(), a}, {MatMulOp::getRhsInIndex(), b}},
      {{MatMulOp::getOutIndex(), c}},
      Onnx::Operators::MatMul_9,
      Op::Settings{g, "MatMul"},
      nonstd::nullopt,
      MatMulOp::SerialiseSettings{},
      OptionalDataType{},
      MatMulPartialsType::FLOAT);

  auto fmod = g.createConnectedOp<FmodOp>(
      {{FmodOp::getArg0InIndex(), c}, {FmodOp::getArg1InIndex(), d}},
      {{FmodOp::getOutIndex(), e}},
      Onnx::CustomOperators::Fmod,
      Op::Settings{g, "Fmod"});

  // Check priorities of variants are unchanged from defaults after applying
  // pattern to fmod.

  const std::vector<std::tuple<OperatorIdentifier, float>> ps =
      fmod->inplacePriorityDefault();

  UpdateInplacePrioritiesForIpu pat;
  pat.apply(fmod);

  const auto tol = boost::test_tools::tolerance(1e-20);

  for (const auto &opid_p : ps) {
    const auto &variant_opid    = std::get<0>(opid_p);
    const auto default_priority = std::get<1>(opid_p);

    const auto new_priority = fmod->getInplacePriority(variant_opid);

    BOOST_TEST(default_priority == new_priority, tol);
  }
}

BOOST_AUTO_TEST_CASE(TestIncreasesPriorityOfEveryBranchWithConvOrMatMul) {
  Ir ir;
  Graph &g = ir.getMainGraph();

  /*
    d = const
    e = var
    f = conv(d, e)
    i = identity(f)

    a, b = const
    c = matmul(a, b)
    k = groupnorm(c)
    l = reshape(k)
    m = dropout(l)

    h = add(j, m)

    Apply pattern to add, check priorities unchanged.
   */

  const TensorId a = "a";
  const TensorId b = "b";
  const TensorId c = "c";
  const TensorId d = "d";
  const TensorId e = "e";
  const TensorId f = "f";
  const TensorId h = "h";
  const TensorId i = "i";
  const TensorId j = "j";
  const TensorId k = "k";
  const TensorId l = "l";
  const TensorId m = "m";

  // Batch size and number channels for tensor we will create.
  constexpr int64_t B     = 8;
  constexpr int64_t C     = 4;
  constexpr int64_t Nx    = 4;
  constexpr int64_t Ny    = 4;
  constexpr int64_t group = 1;
  const TensorInfo convDataInInfo{DataType::FLOAT, Shape{B, C, Nx, Ny}};
  const TensorInfo convWeightsInInfo{DataType::FLOAT,
                                     Shape{C, C / group, 1, 1}};

  const std::vector<float> convData(convDataInInfo.nelms());

  g.getTensors().addConstInit(d, convDataInInfo, convData.data(), "d");
  g.getTensors().addVarInit(e, convWeightsInInfo, convData.data(), "e");

  g.createConnectedOp<ConvOp>(
      {{ConvOp::getDataInIndex(), d}, {ConvOp::getWeightsInIndex(), e}},
      {{ConvOp::getOutIndex(), f}},
      Onnx::Operators::Conv_11,
      Op::Settings{g, "Conv"},
      std::vector<int64_t>{}, // strides
      std::vector<int64_t>{}, // pads
      std::vector<int64_t>{}, // dilations
      group,
      AutoPad::VALID,
      MultiConvOptions{{}, {}});

  g.createConnectedOp<IdentityOp>({{IdentityOp::getInIndex(), f}},
                                  {{IdentityOp::getOutIndex(), i}},
                                  Onnx::Operators::Identity_1,
                                  Op::Settings{g, "Identity"});

  // Say conv out shape is S. We create tensors a and b with shapes (S, 1) and
  // (1,), then matmul to get back a tensor of shape S.
  const auto convOutShape = g.getTensors().get(f)->info.shape();
  auto matMulLhsShape     = convOutShape;
  matMulLhsShape.push_back(1);
  auto matMulRhsShape = Shape{1};

  const TensorInfo matMulLhsInfo{DataType::FLOAT, matMulLhsShape};
  const TensorInfo matMulRhsInfo{DataType::FLOAT, matMulRhsShape};
  const std::vector<float> someData(matMulLhsInfo.nelms());

  g.getTensors().addConstInit(a, matMulLhsInfo, someData.data(), "a");
  g.getTensors().addConstInit(b, matMulRhsInfo, someData.data(), "b");

  g.createConnectedOp<MatMulOp>(
      {{MatMulOp::getLhsInIndex(), a}, {MatMulOp::getRhsInIndex(), b}},
      {{MatMulOp::getOutIndex(), c}},
      Onnx::Operators::MatMul_9,
      Op::Settings{g, "MatMul"},
      nonstd::nullopt,
      MatMulOp::SerialiseSettings{},
      OptionalDataType{},
      MatMulPartialsType::FLOAT);

  // Groupnorm c (preserves tensor shape)
  constexpr int64_t numGroups = 1;
  constexpr float epsilon     = 0.05;

  // GroupNorm parameter Shapes are num channels, which is the second dimension.
  const TensorInfo groupNormParamInfo{DataType::FLOAT, Shape{convOutShape[1]}};
  const std::vector<float> groupNormParamData(groupNormParamInfo.nelms());
  const TensorId groupNormScale = "gnScale";
  const TensorId groupNormBias  = "gnBias";

  g.getTensors().addVarInit(
      groupNormScale, groupNormParamInfo, groupNormParamData.data(), "gnScale");
  g.getTensors().addVarInit(
      groupNormBias, groupNormParamInfo, groupNormParamData.data(), "gnBias");

  const TensorId groupNormMean      = "gnMean";
  const TensorId groupNormInvStdDev = "gnInvStdDev";

  g.createConnectedOp<GroupNormOp>(
      {{GroupNormOp::getXInIndex(), c},
       {GroupNormOp::getScaleInIndex(), groupNormScale},
       {GroupNormOp::getBInIndex(), groupNormBias}},
      {{GroupNormOp::getYOutIndex(), k},
       {GroupNormOp::getMeanOutIndex(), groupNormMean},
       {GroupNormOp::getInvStdDevOutIndex(), groupNormInvStdDev}},
      Onnx::CustomOperators::GroupNormalization_1,
      numGroups,
      epsilon,
      Op::Settings{g, "GroupNorm"});

  // Reshape (B, C, Nx, Ny) to (1, B, C, Nx, Ny)
  auto newShape = g.getTensors().get(k)->info.shape();
  newShape.insert(newShape.begin(), 1);

  g.createConnectedOp<ReshapeOp>({{ReshapeOp::getInIndex(), k}},
                                 {{ReshapeOp::getOutIndex(), l}},
                                 Onnx::Operators::Reshape_5,
                                 newShape,
                                 Op::Settings{g, "Reshape"});

  constexpr float dropRatio = 0.2f;
  g.createConnectedOp<DropoutOp>({{DropoutOp::getInIndex(), l}},
                                 {{DropoutOp::getOutIndex(), m}},
                                 Onnx::Operators::Dropout_10,
                                 dropRatio,
                                 Op::Settings{g, "Dropout"});

  // Add output of MatMul branch with output of Conv branch.
  auto add = g.createConnectedOp<AddOp>(
      {{AddOp::getArg0InIndex(), i}, {AddOp::getArg1InIndex(), m}},
      {{AddOp::getOutIndex(), h}},
      Onnx::Operators::Add_7,
      Op::Settings{g, "Add"});

  const std::vector<std::tuple<OperatorIdentifier, float>> ps =
      add->inplacePriorityDefault();

  // Test priorities of variants are higher than their defaults after applying
  // pattern to add.

  UpdateInplacePrioritiesForIpu pat;
  pat.apply(add);

  for (const auto &opid_p : ps) {
    const auto &variant_opid    = std::get<0>(opid_p);
    const auto default_priority = std::get<1>(opid_p);

    const auto new_priority = add->getInplacePriority(variant_opid);

    BOOST_TEST(default_priority < new_priority);
  }
}
