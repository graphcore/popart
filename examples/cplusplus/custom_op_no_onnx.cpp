// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

//
// This example demonstrates how to provide builder shape inference for a custom
// operator without requiring onnx.
//

#include <memory>
#include <popart/builder.hpp>
#include <popart/devicemanager.hpp>
#include <popart/logging.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/l1.hpp>
#include <popart/opmanager.hpp>
#include <popart/optimizer.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/session.hpp>
#include <popart/shapeinference.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

#include <popops/ElementWise.hpp>

namespace CustomOperators {
const popart::OperatorIdentifier Cube = {"com.acme", "Cube", 1};
} // namespace CustomOperators

namespace {
// for C++11 compatibility, we don't use std::make_unique
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namespace

class CubeOp : public popart::Op {
public:
  CubeOp(const popart::OperatorIdentifier &_opid,
         const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_) {}

  virtual void setup() { outInfo(0) = inInfo(0); }

  std::unique_ptr<Op> clone() const final { return make_unique<CubeOp>(*this); }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

static popart::OpCreator<CubeOp> cubeOpCreator({{CustomOperators::Cube, {}}});

static popart::RegisterShapeInferenceFunction
    cubeOpShapeInference(CustomOperators::Cube,
                         [](auto &ctx) { ctx.outInfo(0) = ctx.inInfo(0); });

auto main(int argc, char **argv) -> int {
  auto builder     = popart::Builder::create();
  auto aiGraphcore = builder->aiGraphcoreOpset1();

  popart::TensorInfo inputInfo{"FLOAT", std::vector<int64_t>{2}};

  auto input = builder->addInputTensor(inputInfo);

  auto outputs = builder->customOp(CustomOperators::Cube, 1, {input}, 1, {});
  auto outputShape = builder->getTensorShape(outputs[0]);
  popart::logging::debug("outputShape: {}", outputShape);

  auto x      = builder->aiOnnxOpset9().mul({outputs[0], outputs[0]});
  auto xShape = builder->getTensorShape(x);
  popart::logging::debug("x shape: {}", xShape);

  auto proto = builder->getModelProto();

  auto dataFlow =
      popart::DataFlow(1, {{outputs[0], popart::AnchorReturnType("All")}});

  auto cpuDevice =
      popart::DeviceManager::createDeviceManager().createCpuDevice();

  auto patterns =
      popart::Patterns(popart::PatternsLevel::Minimal).enablePreUniRepl(true);
  auto session = popart::InferenceSession::createFromOnnxModel(
      proto, dataFlow, cpuDevice, popart::InputShapeInfo(), {}, patterns);
}
