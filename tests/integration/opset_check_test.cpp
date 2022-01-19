// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE OpsetCheckTest

#include <boost/test/unit_test.hpp>

#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/operators.hpp>
#include <popart/opmanager.hpp>
#include <popart/session.hpp>

using namespace popart;

class HardmaxOp : public Op {
public:
  HardmaxOp(const OperatorIdentifier &opid_, const Op::Settings &settings_)
      : Op(opid_, settings_) {}

  std::unique_ptr<Op> clone() const override { throw error("unimplemented"); }
  float getSubgraphValue() const override { throw error("unimplemented"); }
  void setup() override { throw error("setup is unimplemented"); }
};

static OpDefinition hardmaxOpDef({OpDefinition::Inputs({}),
                                  OpDefinition::Outputs({}),
                                  OpDefinition::Attributes({})});

static OpCreator<HardmaxOp> hardmaxOpCreator(OpDefinitions({
    {Onnx::Operators::Hardmax_1, hardmaxOpDef},

}));

// This test works because Hardmax is not implemented in any version at the
// moment. If Hardmax is ever implemented, this op will break.
//
// This test checks that the strict onnx opset version checking works, and also
// checks that the session option 'strictOpVersions' works.
BOOST_AUTO_TEST_CASE(OpsetCheck) {
  auto test = [](bool strictOpsetVersions) {
    auto builder     = Builder::create();
    auto aiGraphcore = builder->aiGraphcoreOpset1();

    TensorInfo inputInfo{"FLOAT", std::vector<int64_t>{2}};

    auto input = builder->addInputTensor(inputInfo);
    std::vector<TensorId> outputs;
    if (strictOpsetVersions) {
      outputs =
          builder->customOp(Onnx::Operators::Hardmax_1, 11, {input}, 1, {});
    } else {
      outputs =
          builder->customOp(Onnx::Operators::Hardmax_1, 9, {input}, 1, {});
    }

    auto l1 = aiGraphcore.l1loss({outputs.at(0)}, 0.1f, ReductionType::Sum);

    auto proto = builder->getModelProto();

    auto dataFlow = DataFlow(1, {{outputs.at(0), AnchorReturnType("All")}});

    auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

    SessionOptions opts   = SessionOptions();
    opts.strictOpVersions = strictOpsetVersions;

    // Create the session
    BOOST_CHECK_EXCEPTION(
        InferenceSession::createFromOnnxModel(
            proto, dataFlow, cpuDevice, {}, opts),
        error,
        [&](error const &err) {
          std::string what    = err.what();
          auto expected_error = [&]() -> std::string {
            if (strictOpsetVersions) {
              return "For an opset 11 Model, the ONNX spec stipulates that a "
                     "Hardmax op must be version 11. The highest version we "
                     "have implemented less than or equal to 11 is 1, so "
                     "bailing. This check may be disabled by setting "
                     "popart::SessionOptions::strictOpVersions to false.";
            } else {
              return "setup is unimplemented";
            }
          };

          return what == expected_error();
        });
  };

  test(true);
  test(false);
}
