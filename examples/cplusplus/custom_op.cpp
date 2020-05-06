// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

//
// This example demonstrates how to create a custom operator for onnx, in this
// case an op that will take a tensor and cube all the elements
//
//
// ISSUE : the version can currently only be 9. Need to support onnx version
// information

#include <memory>
#include <popart/builder.hpp>
#include <popart/devicemanager.hpp>
#include <popart/logging.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op.hpp>
#include <popart/op/l1.hpp>
#include <popart/opmanager.hpp>
#include <popart/optimizer.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/session.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

#include <popops/ElementWise.hpp>

namespace Onnx {
namespace CustomOperators {
const popart::OperatorIdentifier Cube = {"com.acme", "Cube", 1};
} // namespace CustomOperators
namespace CustomGradOperators {
const popart::OperatorIdentifier CubeGrad = {"com.acme", "CubeGrad", 1};
} // namespace CustomGradOperators
} // namespace Onnx

// For training with a custom Op, four classes need to be implemented,
// one for each of:
// {forward, gradient} x {Op, Opx}.
//
// If only inference is required, then two classes need to be implemented:
// {forward} x {Op, Opx}.
//
// The Op is a poplar/hardware agnostic description of the computation.
// the Opx is the poplar implementation of the Op.
//
// We do training in this example, so the four classes implemented are:
//
class CubeOp;
class CubeGradOp;
class CubeOpx;
class CubeGradOpx;

namespace {
// for C++11 compatibility, we don't use std::make_unique
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namespace

// The gradient Op
class CubeGradOp : public popart::Op {
public:
  CubeGradOp(const popart::Op &fwdOp)
      : popart::Op(Onnx::CustomGradOperators::CubeGrad, fwdOp.getSettings()) {}

  std::unique_ptr<Op> clone() const final {
    return make_unique<CubeGradOp>(*this);
  }

  // same comment as for CubeOp, for running shape/type inference "statically"
  virtual void setup() { outInfo(0) = inInfo(0); }

  // function describing the inputs and output(s) of CubeGradOp
  // The Gradient Op which we are implementing (CubeGradOp) has 2 inputs.
  // The input at index 0 is:
  // the gradient of the 0'th output Tensor of the CubeOp.
  // The input at index 1 is :
  // the 0'th output Tensor of the CubeOp.
  // Supposing the CubeOp has input Tensor T0 and output Tensor T1,
  //
  //   input at index 0 (T0)
  //          |
  //        CubeOp
  //          |
  //   output at index 0 (T1)
  //
  // Then the picture described by the map below looks like,
  //
  //
  //    input at index 0 (gradient of T1)
  //         |   input at index 1 (T1)
  //         |     |
  //         |     |
  //        CubeGradOp
  //            |
  //            |
  //   output at index 0 (gradient of T0)
  //
  virtual const std::vector<popart::GradInOutMapper> &gradInputInfo() const {
    static const std::vector<popart::GradInOutMapper> inInfo = {
        {0, 0, popart::GradOpInType::GradOut},
        {1, 0, popart::GradOpInType::Out}};
    return inInfo;
  }

  // The Grad Op only has one output, at index 0. The output at index 0
  // is the gradient of the input at index 0 of the CubeOp
  const std::map<int, int> &gradOutToNonGradIn() const {
    static const std::map<int, int> outInfo = {{0, 0}};
    return outInfo;
  }

  // an estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

// The forward Op
class CubeOp : public popart::Op {
public:
  CubeOp(const popart::OperatorIdentifier &_opid,
         const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_) {}

  // The output popart Tensor has the same inputInfo and numerical type
  // (i.e. the same TensorInfo) as the input Tensor. This function is
  // required for inputInfo/type inference
  //
  virtual void setup() { outInfo(0) = inInfo(0); }

  std::unique_ptr<Op> clone() const final { return make_unique<CubeOp>(*this); }

  // There is only one Gradient Op for CubeOp, a CubeGradOp
  // It is possible to have multiple Gradient Ops
  // (Conv has 2 in popart, one for weights and one for activations)
  //
  std::vector<std::unique_ptr<popart::Op>> getGradOps() {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(new CubeGradOp(*this));
    return upops;
  }

  // an estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

// describe the inputs and outputs that are supported by the operation

static popart::OpDefinition::DataTypes T = {popart::DataType::FLOAT16,
                                            popart::DataType::FLOAT};

static popart::OpDefinition
    cubeOpDef({popart::OpDefinition::Inputs({{"input", T}}),
               popart::OpDefinition::Outputs({{"output", T}}),
               popart::OpDefinition::Attributes({})});

static popart::OpCreator<CubeOp>
    cubeOpCreator({{Onnx::CustomOperators::Cube, cubeOpDef}});

// forward Opx (poplar implementation of the forward Op)
class CubeOpx : public popart::popx::Opx {
public:
  CubeOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    // not strictly necessary, we check that op is castable to a CubeOp *.
    verifyOp<CubeOp>(op, Onnx::CustomOperators::Cube);
  }

  void grow(poplar::program::Sequence &prog) const final {
    // Cube the input. We create a poplar::Tensor of name outId(0)
    insert(outId(0),
           popops::map(graph(),
                       popops::expr::Mul(popops::expr::Mul(popops::expr::_1,
                                                           popops::expr::_1),
                                         popops::expr::_1),
                       {get(inId(0))},
                       prog,
                       debugPrefix()));
  }
};

class CubeGradOpx : public popart::popx::Opx {
public:
  CubeGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<CubeGradOp>(op, Onnx::CustomGradOperators::CubeGrad);
  }

  // Create the gradient poplar::Tensor, which is
  // 3 * input_to_cube**2 * gradient_of_cube_output
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
                    debugPrefix()));
  }
};

static popart::popx::OpxCreator<CubeOpx>
    cubeOpxCreator(Onnx::CustomOperators::Cube);
static popart::popx::OpxCreator<CubeGradOpx>
    cubeGradOpxCreator(Onnx::CustomGradOperators::CubeGrad);

auto main(int argc, char **argv) -> int {

  // TODO : parse input arguments so we can test on different targets cpu vs hw
  (void)argc;
  (void)argv;

  // step 1 : generate an ONNX inference Model which uses Cube.
  // The simple mode will be : input->Cube->output
  //
  auto builder = popart::Builder::create();

  // The input Tensor will be of type FLOAT, and will
  // be a rank-1 tensor with 2 elements
  popart::TensorInfo inputInfo{"FLOAT", std::vector<int64_t>{2}};

  auto input = builder->addInputTensor(inputInfo);

  auto outputs =
      builder->customOp(Onnx::CustomOperators::Cube, 1, {input}, 1, {});

  builder->addOutputTensor(outputs[0]);

  auto proto = builder->getModelProto();

  // step 2 : add additional information for training, currently not part of
  // the ONNX specification:
  // 2.1 an Optimiser.
  auto optimizer = popart::ConstSGD(0.01f);

  // 2.2 Loss(es).
  // 2.2.1 l1 loss : 0.1 * |output|_1
  std::unique_ptr<popart::L1Loss> l1Loss(new popart::L1Loss(
      outputs[0], "l1LossVal", 0.1f, popart::ReductionType::Sum));
  std::vector<popart::Loss *> losses{l1Loss.get()};

  // 2.3 Data streaming.
  // We will stream
  // 1) the output tensor back to host every iteration
  // 2) the gradient of input tensor back to host every iteration
  auto dataFlow =
      popart::DataFlow(1, // this is the number of batches per step. It does not
                          // have an equivalent in other standard frameworks
                          // like Tensorflow. It is the number of batches to
                          // process when session->run(.) is called. (see below)
                       {{outputs[0], popart::AnchorReturnType("All")},
                        {popart::reservedGradientPrefix() + input,
                         popart::AnchorReturnType("All")}});

  auto cpuDevice =
      popart::DeviceManager::createDeviceManager().createCpuDevice();

  // Create the session
  auto session = popart::TrainingSession::createFromOnnxModel(
      proto,
      dataFlow,
      losses,
      optimizer,
      cpuDevice,
      popart::InputShapeInfo(),
      {},
      popart::Patterns({popart::PreAliasPatternType::PreUniRepl}));

  // prepare the anchors buffers. The anchors are what were specified in 2.3
  // for data streaming: the tensors which will be returned from the device
  // to the host. We specified 2 such tensors in 2.3,
  // 1) the output tensor (i.e. the output of the forward pass)
  float rawOutputData[2] = {0, 0};
  popart::NDArrayWrapper<float> outData(rawOutputData, {2});

  // 2) and the gradient of input tensor
  float rawGradInputData[2] = {0, 0};
  popart::NDArrayWrapper<float> gradInData(rawGradInputData, {2});
  std::map<popart::TensorId, popart::IArray &> anchors = {
      {outputs[0], outData},
      {popart::reservedGradientPrefix() + input, gradInData},
  };

  session->prepareDevice();

  // prepare the input tensor for this example
  float rawInputData[2] = {2.0f, 4.0f};
  popart::NDArrayWrapper<float> inData(rawInputData, {2});
  std::map<popart::TensorId, popart::IArray &> inputs = {{input, inData}};

  popart::StepIO stepio(inputs, anchors);

  session->weightsFromHost();

  session->run(stepio);

  popart::logging::ir::err("input : {}", inData);
  popart::logging::ir::err("output : {}", outData);
  popart::logging::ir::err("dInput : {}", gradInData);
}
