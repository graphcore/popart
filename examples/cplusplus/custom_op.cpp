
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
#include <poponnx/tensornames.hpp>

#include <popops/ElementWise.hpp>

namespace Onnx {
namespace CustomOperators {
const poponnx::OperatorIdentifier Cube = {"com.acme", "Cube", 1};
} // namespace CustomOperators
namespace CustomGradOperators {
const poponnx::OperatorIdentifier CubeGrad = {"com.acme", "CubeGrad", 1};
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

// The forward Op
class CubeOp : public poponnx::Op {
public:
  CubeOp(const poponnx::OperatorIdentifier &_opid,
         const poponnx::Op::Settings &settings_)
      : poponnx::Op(_opid, settings_) {}

  // The output poponnx Tensor has the same inputInfo and numerical type
  // (i.e. the same TensorInfo) as the input Tensor. This function is
  // required for inputInfo/type inference
  //
  virtual void setup() { outInfo(0) = inInfo(0); }

  // There is only one Gradient Op for CubeOp, a CubeGradOp
  // It is possible to have multiple Gradient Ops
  // (Conv has 2 in poponnx, one for weights and one for activations)
  //
  std::vector<std::unique_ptr<poponnx::Op>> getGradOps() {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(new CubeGradOp(*this));
    return upops;
  }
};

// The gradient Op
class CubeGradOp : public poponnx::Op {
public:
  CubeGradOp(const poponnx::Op &fwdOp)
      : poponnx::Op(Onnx::CustomGradOperators::CubeGrad, fwdOp.getSettings()) {}

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
  virtual const std::vector<poponnx::GradInOutMapper> &gradInputInfo() const {
    static const std::vector<poponnx::GradInOutMapper> inInfo = {
        {0, 0, poponnx::GradOpInType::GRADOUT},
        {1, 0, poponnx::GradOpInType::OUT}};
    return inInfo;
  }

  // The Grad Op only has one output, at index 0. The output at index 0
  // is the gradient of the input at index 0 of the CubeOp
  const std::map<int, int> &gradOutToNonGradIn() const {
    static const std::map<int, int> outInfo = {{0, 0}};
    return outInfo;
  }
};

static poponnx::OpCreator<CubeOp> cubeOpCreator(Onnx::CustomOperators::Cube);

// forward Opx (poplar implementation of the forward Op)
class CubeOpx : public poponnx::popx::Opx {
public:
  CubeOpx(poponnx::Op *op, poponnx::popx::Devicex *devicex)
      : poponnx::popx::Opx(op, devicex) {
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
                       idStr()));
  }
};

class CubeGradOpx : public poponnx::popx::Opx {
public:
  CubeGradOpx(poponnx::Op *op, poponnx::popx::Devicex *devicex)
      : poponnx::popx::Opx(op, devicex) {
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

  // step 1 : generate an ONNX inference Model which uses Cube.
  // The simple mode will be : input->Cube->output
  //
  auto builder = poponnx::Builder::create();

  // The input Tensor will be of type FLOAT, and will
  // be a rank-1 tensor with 2 elements
  poponnx::TensorInfo inputInfo{"FLOAT", std::vector<int64_t>{2}};

  auto input = builder->addInputTensor(inputInfo);

  auto outputs =
      builder->customOp(Onnx::CustomOperators::Cube, 1, {input}, 1, {});

  builder->addOutputTensor(outputs[0]);

  auto proto = builder->getModelProto();

  // step 2 : add additional information for training, currently not part of
  // the ONNX specification:
  // 2.1 an Optimiser.
  auto optimizer = poponnx::ConstSGD(0.01f);

  // 2.2 Loss(es).
  // 2.2.1 l1 loss : 0.1 * |output|_1
  std::unique_ptr<L1Loss> l1Loss(
      new popoonnx::L1Loss(outputs[0], "l1LossVal", 0.1f));
  std::vector<poponnx::Loss *> losses{l1Loss.get()};

  // 2.3 Data streaming.
  // We will stream
  // 1) the output tensor back to host every iteration
  // 2) the gradient of input tensor back to host every iteration
  auto dataFlow = poponnx::DataFlow(
      1, // this is the number of batches per step. It does not have an
         // equivalent in other standard frameworks like Tensorflow. It is the
         // number of batches to process when session->run(.) is called.
         // (see below)
      {{outputs[0], poponnx::AnchorReturnType("ALL")},
       {poponnx::reservedGradientPrefix() + input,
        poponnx::AnchorReturnType("ALL")}});

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

  // prepare the anchors buffers. The anchors are what were specified in 2.3
  // for data streaming: the tensors which will be returned from the device
  // to the host. We specified 2 such tensors in 2.3,
  // 1) the output tensor (i.e. the output of the forward pass)
  float rawOutputData[2] = {0, 0};
  poponnx::NDArrayWrapper<float> outData(rawOutputData, {2});

  // 2) and the gradient of input tensor
  float rawGradInputData[2] = {0, 0};
  poponnx::NDArrayWrapper<float> gradInData(rawGradInputData, {2});
  std::map<poponnx::TensorId, poponnx::IArray &> anchors = {
      {outputs[0], outData},
      {poponnx::reservedGradientPrefix() + input, gradInData},
  };

  session->prepareDevice();

  // prepare the input tensor for this example
  float rawInputData[2] = {2.0f, 4.0f};
  poponnx::NDArrayWrapper<float> inData(rawInputData, {2});
  std::map<poponnx::TensorId, poponnx::IArray &> inputs = {{input, inData}};

  poponnx::StepIO stepio(inputs, anchors);

  session->weightsFromHost();
  session->optimizerFromHost();
  session->run(stepio);

  poponnx::logging::ir::err("input : {}", inData);
  poponnx::logging::ir::err("output : {}", outData);
  poponnx::logging::ir::err("dInput : {}", gradInData);
}
