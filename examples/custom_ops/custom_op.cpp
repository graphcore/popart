
//
// This example demonstrates how to create a custom operator for onnx, in this
// case an op that will take a tensor and cube all the elements.
//
// This example if compiled into a .so shared object library, will be usable in
// python.
// To compile, cd into the custom_ops directory, then run:
// `g++  -fPIC custom_op.cpp -shared -lpopart -o custom_op.so`
// This will create an .so file that can be used in the python file. Make sure
// any virtualenvs and poplar are activated then run `python custom_op.py` and
// you should see a printout of some inputs, weights and outputs.

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

namespace {
// for C++11 compatibility, we don't use std::make_unique
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namespace

// Use extern to avoid mangled names when importing to python
extern "C" {

namespace Onnx {
namespace CustomOperators {
const popart::OperatorIdentifier Cube = {"ai.acme", "Cube", 1};
} // namespace CustomOperators
namespace CustomGradOperators {
const popart::OperatorIdentifier CubeGrad = {"ai.acme", "CubeGrad", 1};
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
        {0, 0, popart::GradOpInType::GRADOUT},
        {1, 0, popart::GradOpInType::OUT}};
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

static popart::OpCreator<CubeOp> cubeOpCreator(Onnx::CustomOperators::Cube);

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
}