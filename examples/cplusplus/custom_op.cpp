
//
// This example demonstrates how to create a custom operator for onnx, in this
// case an op that will take a tensor and cube all the elements
//
//
// ISSUE : the version can currently only be 9. Need to support onnx version
// information

#include <poponnx/builder.hpp>
#include <poponnx/devicemanager.hpp>
#include <poponnx/logging.hpp>
#include <poponnx/ndarraywrapper.hpp>
#include <poponnx/op.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/patterns/pattern.hpp>
#include <poponnx/popx/opx.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/session.hpp>
#include <poponnx/tensordata.hpp>
#include <poponnx/tensorinfo.hpp>

#include <popops/ElementWise.hpp>

namespace Onnx {
namespace CustomOperators {
const poponnx::OperatorIdentifier Cube = {"com.acme", "Cube", 1};
}
namespace CustomGradOperators {
const poponnx::OperatorIdentifier CubeGrad = {"com.acme", "CubeGrad", 1};
}
} // namespace Onnx

class CubeGradOp : public poponnx::Op {
public:
  CubeGradOp(const poponnx::Op &fwdOp)
      : poponnx::Op(Onnx::CustomGradOperators::CubeGrad, fwdOp.getSettings()) {}

  virtual void setup() { outInfo(0) = inInfo(0); }

  virtual const std::vector<poponnx::GradInOutMapper> &gradInputInfo() const {
    static const std::vector<poponnx::GradInOutMapper> inInfo = {
        {0, 0, poponnx::GradOpInType::GRADOUT},
        {1, 0, poponnx::GradOpInType::OUT}};
    return inInfo;
  }

  const std::map<int, int> &gradOutToNonGradIn() const {
    static const std::map<int, int> outInfo = {{0, 0}};
    return outInfo;
  }
};

class CubeOp : public poponnx::Op {
public:
  CubeOp(const poponnx::OperatorIdentifier &_opid,
         const poponnx::Op::Settings &settings_)
      : poponnx::Op(_opid, settings_) {}

  virtual void setup() { outInfo(0) = inInfo(0); }

  std::vector<std::unique_ptr<poponnx::Op>> getGradOps() {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(new CubeGradOp(*this));
    return upops;
  }
};

static poponnx::OpCreator<CubeOp> cubeOpCreator(Onnx::CustomOperators::Cube);

class CubeOpx : public poponnx::popx::Opx {
public:
  CubeOpx(poponnx::Op *op, poponnx::popx::Devicex *devicex)
      : poponnx::popx::Opx(op, devicex) {
    verifyOp<CubeOp>(op, Onnx::CustomOperators::Cube);
  }
  void grow(poplar::program::Sequence &prog) const final {
    // Cube the input
    insert(outId(0),
           popops::map(graph(),
                       popops::expr::Mul(popops::expr::Mul(popops::expr::_1,
                                                           popops::expr::_1),
                                         popops::expr::_1),
                       {get(inId(0))},
                       prog,
                       idStr()));
  }
};

class CubeGradOpx : public poponnx::popx::Opx {
public:
  CubeGradOpx(poponnx::Op *op, poponnx::popx::Devicex *devicex)
      : poponnx::popx::Opx(op, devicex) {
    verifyOp<CubeGradOp>(op, Onnx::CustomGradOperators::CubeGrad);
  }
  void grow(poplar::program::Sequence &prog) const final {

    insert(
        outId(0),
        popops::map(graph(),
                    popops::expr::Mul(
                        popops::expr::Const(3),
                        popops::expr::Mul(popops::expr::Mul(popops::expr::_1,
                                                            popops::expr::_1),
                                          popops::expr::_2)),
                    {get(inId(0)), get(inId(1))}, // FwdOut, GradOut
                    prog,
                    idStr()));
  }
};

static poponnx::popx::OpxCreator<CubeOpx>
    cubeOpxCreator(Onnx::CustomOperators::Cube);
static poponnx::popx::OpxCreator<CubeGradOpx>
    cubeGradOpxCreator(Onnx::CustomGradOperators::CubeGrad);

auto main(int argc, char **argv) -> int {

  // TODO : parse input arguments so we can test on different targets cpu vs hw
  (void)argc;
  (void)argv;

  auto builder = poponnx::Builder::create();

  poponnx::TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto input = builder->addInputTensor(shape);

  auto outputs =
      builder->customOp(Onnx::CustomOperators::Cube, 1, {input}, 1, {});

  builder->addOutputTensor(outputs[0]);

  auto proto = builder->getModelProto();

  auto dataFlow =
      poponnx::DataFlow(1,
                        {{outputs[0], poponnx::AnchorReturnType("ALL")},
                         {poponnx::reservedGradientPrefix() + input,
                          poponnx::AnchorReturnType("ALL")}});
  auto optimizer = poponnx::ConstSGD(0.01f);
  std::vector<poponnx::Loss *> losses{
      new poponnx::L1Loss(outputs[0], "l1LossVal", 0.1f)};

  auto cpuDevice =
      poponnx::DeviceManager::createDeviceManager().createCpuDevice();

  // Create the session
  auto session = poponnx::TrainingSession::createFromOnnxModel(
      proto,
      dataFlow,
      losses,
      optimizer,
      cpuDevice,
      poponnx::InputShapeInfo(),
      {},
      poponnx::Patterns({poponnx::PreAliasPatternType::PREUNIREPL}));

  // prepare the anchors
  float rawOutputData[2] = {0, 0};
  poponnx::NDArrayWrapper<float> outData(rawOutputData, {2});

  float rawWeightData[2] = {0, 0};
  poponnx::NDArrayWrapper<float> outWeights(rawWeightData, {2});
  std::map<poponnx::TensorId, poponnx::IArray &> anchors = {
      {outputs[0], outData},
      {poponnx::reservedGradientPrefix() + input, outWeights},
  };

  session->prepareDevice();

  // prepare the inputs
  float rawInputData[2] = {2.0f, 4.0f};
  poponnx::NDArrayWrapper<float> inData(rawInputData, {2});
  std::map<poponnx::TensorId, poponnx::IArray &> inputs = {{input, inData}};

  poponnx::StepIO stepio(inputs, anchors);

  session->weightsFromHost();
  session->optimizerFromHost();
  session->run(stepio);

  poponnx::logging::ir::err("input : {}", inData);
  poponnx::logging::ir::err("output : {}", outData);
  poponnx::logging::ir::err("dInput : {}", outWeights);
}
