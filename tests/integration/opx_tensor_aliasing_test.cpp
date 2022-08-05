// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE OpxTensorAliasingTest

#include <algorithm>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <filereader.hpp>
#include <map>
#include <memory>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <string>
#include <testdevice.hpp>
#include <vector>
#include <poplar/exceptions.hpp>
#include <popops/ElementWise.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/popopx.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>

#include "popart/error.hpp"
#include "popart/names.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/stepio.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/vendored/any.hpp"
#include "random_util.hpp"

namespace popart {
class IArray;
namespace popx {
class Devicex;
} // namespace popx
} // namespace popart

using namespace popart;

namespace CustomOperators {
const popart::OperatorIdentifier OpxTensorAliasingTest = {
    "ai.graphcore",
    "OpxTensorAliasingTestOp",
    1};
} // namespace CustomOperators

class OpxTensorAliasingTestOp : public Op {
public:
  OpxTensorAliasingTestOp(const OperatorIdentifier &_opid,
                          const Op::Settings &settings)
      : Op(_opid, settings) {}

  static InIndex inIndex() { return 0; }
  static InIndex outIndex() { return 0; }

  void setup() final { outInfo(outIndex()) = inInfo(inIndex()); }

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<OpxTensorAliasingTestOp>(*this);
  }

  virtual float getSubgraphValue() const override {
    return getLowSubgraphValue();
  }
};

class OpxTensorAliasingTestOpx : public popx::PopOpx {
public:
  OpxTensorAliasingTestOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::PopOpx(op, devicex) {
    verifyOp<OpxTensorAliasingTestOp>(op,
                                      CustomOperators::OpxTensorAliasingTest);
  }

  void grow(snap::program::Sequence &prog) const final {
    auto inTensor =
        getInTensor(OpxTensorAliasingTestOp::inIndex()).getPoplarTensor();
    popops::mulInPlace(graph().getPoplarGraph(),
                       inTensor,
                       inTensor,
                       prog.getPoplarSequence(),
                       debugContext("mulInPlace"));
    setOutTensor(OpxTensorAliasingTestOp::outIndex(),
                 snap::Tensor{inTensor, graph()});
  }
};

static popart::popx::OpxCreator<OpxTensorAliasingTestOpx>
    OpxTensorAliasingTestOpxCreator(CustomOperators::OpxTensorAliasingTest);

static popart::OpDefinition OpxTensorAliasingTestOpDef({});

static popart::OpCreator<OpxTensorAliasingTestOp>
    OpxTensorAliasingTestOpCreator(
        popart::OpDefinitions({{CustomOperators::OpxTensorAliasingTest,
                                OpxTensorAliasingTestOpDef}}),
        [](const OpCreatorInfo &info) {
          return std::unique_ptr<OpxTensorAliasingTestOp>(
              new OpxTensorAliasingTestOp(info.opid, info.settings));
        },
        true);

// Tests if the OpxTensorAliasingOp triggers the opxAliasChecking error by
// aliasing the input to the output without providing correct aliasing
// information to the IR
BOOST_AUTO_TEST_CASE(OpxTensorAliasingTest_0) {
  auto opts              = SessionOptions();
  opts.opxAliasChecking  = true;
  opts.opxModifyChecking = false;

  int seed = 1337;
  DefaultRandomEngine eng(seed);
  UniformRealDistribution<float> fdis(-4.f, 4.f);

  auto bder        = Builder::create();
  auto aiOnnx      = bder->aiOnnxOpset9();
  auto aiGraphcore = bder->aiGraphcoreOpset1();

  // Tensor A of shape 5 x 5
  TensorInfo A_info{"FLOAT", std::vector<int64_t>{5, 5}};
  auto B_info = A_info;
  std::vector<float> v_A_init(A_info.nelms());
  for (auto &val : v_A_init) {
    val = fdis(eng);
  }
  TensorId A_id =
      bder->addInitializedInputTensor({v_A_init.data(), A_info}, "A");

  TensorId B_id = bder->customOp(
      CustomOperators::OpxTensorAliasingTest, 1, {A_id}, 1, {}, "TestOp")[0];

  bder->addOutputTensor(B_id);

  auto proto         = bder->getModelProto();
  auto modelProto    = io::getModelFromString(proto);
  auto art           = AnchorReturnType("All");
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{B_id, art}});
  auto device        = createTestDevice(TEST_TARGET);

  try {
    auto session = popart::InferenceSession::createFromOnnxModel(
        proto,
        dataFlow,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));

    std::vector<float> raw_B_out(B_info.nelms());
    popart::NDArrayWrapper<float> B_wrapper(raw_B_out.data(), B_info.shape());
    std::map<popart::TensorId, popart::IArray &> anchors = {
        {B_id, B_wrapper},
    };

    session->prepareDevice();
    popart::StepIO stepio({}, anchors);
    session->run(stepio);

    for (size_t i = 0; i < B_info.nelms(); ++i) {
      BOOST_CHECK_CLOSE(raw_B_out[i], v_A_init[i] * v_A_init[i], 1e-4f);
    }
  } catch (const popart::error &e) {
    BOOST_CHECK(std::string(e.what()).find(
                    "claims input 0 -> output 0 do not contain aliases, but "
                    "the Poplar tensors disagree.") != std::string::npos);
  }
}

// Tests if the OpxTensorAliasingOp triggers the opxModifyChecking error by
// modifying the input without providing correct modification information
// to the IR
BOOST_AUTO_TEST_CASE(OpxTensorAliasingTest_1) {
  auto opts              = SessionOptions();
  opts.opxAliasChecking  = false;
  opts.opxModifyChecking = true;

  int seed = 1337;
  DefaultRandomEngine eng(seed);
  UniformRealDistribution<float> fdis(-4.f, 4.f);

  auto bder        = Builder::create();
  auto aiOnnx      = bder->aiOnnxOpset9();
  auto aiGraphcore = bder->aiGraphcoreOpset1();

  // Tensor A of shape 5 x 5
  TensorInfo A_info{"FLOAT", std::vector<int64_t>{5, 5}};
  auto B_info = A_info;
  std::vector<float> v_A_init(A_info.nelms());
  for (auto &val : v_A_init) {
    val = fdis(eng);
  }
  TensorId A_id =
      bder->addInitializedInputTensor({v_A_init.data(), A_info}, "A");

  TensorId B_id = bder->customOp(
      CustomOperators::OpxTensorAliasingTest, 1, {A_id}, 1, {}, "TestOp")[0];

  bder->addOutputTensor(B_id);

  auto proto         = bder->getModelProto();
  auto modelProto    = io::getModelFromString(proto);
  auto art           = AnchorReturnType("All");
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{B_id, art}});
  auto device        = createTestDevice(TEST_TARGET);

  try {
    auto session = popart::InferenceSession::createFromOnnxModel(
        proto,
        dataFlow,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));

    std::vector<float> raw_B_out(B_info.nelms());
    popart::NDArrayWrapper<float> B_wrapper(raw_B_out.data(), B_info.shape());
    std::map<popart::TensorId, popart::IArray &> anchors = {
        {B_id, B_wrapper},
    };

    session->prepareDevice();
    popart::StepIO stepio({}, anchors);
    session->run(stepio);

    for (size_t i = 0; i < B_info.nelms(); ++i) {
      BOOST_CHECK_CLOSE(raw_B_out[i], v_A_init[i] * v_A_init[i], 1e-4f);
    }
  } catch (const poplar::poplar_error &e) {
    BOOST_CHECK(std::string(e.what()).find("User defined exception") !=
                std::string::npos);
  }
}
