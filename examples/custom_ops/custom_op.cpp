// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

//
// This example demonstrates how to create a custom operator for onnx, in this
// case an op that will take a tensor and cube all the elements.
//
// This example if compiled into a .so shared object library, will be usable in
// python.
// To compile, cd into the custom_ops directory, then run:
// `cmake .` and `make`
// This will create an .so file that can be used in the python file. Make sure
// any virtualenvs and poplar are activated then run `python custom_op.py` and
// you should see a printout of some inputs, weights and outputs.

#include <memory>
#include <popart/builder.hpp>
#include <popart/devicemanager.hpp>

#include <popart/ir.hpp>

#include <popart/logging.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op.hpp>
#include <popart/op/l1.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/optimizer.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/session.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

#include <poprand/RandomGen.hpp>

#include <popops/ElementWise.hpp>

#include <popart/names.hpp>
#include <popart/opidentifier.hpp>

#include <onnx/defs/schema.h>
#include <onnx/defs/shape_inference.h>

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
const popart::OperatorIdentifier Cube = {"com.acme", "Cube", 1};
} // namespace CustomOperators
namespace CustomGradOperators {
const static popart::OperatorIdentifier CubeGrad = {"com.acme", "CubeGrad", 1};
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
// This is defined first as the CubeOp::getGradOps requires it.
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
  //  input at index 0 (T0)
  //         |
  //       CubeOp
  //         |
  //  output at index 0 (T1)
  //
  // Then the picture described by the map below looks like,
  //
  //
  //   input at index 0 (gradient of T1)
  //        |   input at index 1 (T1)
  //        |     |
  //        |     |
  //       CubeGradOp
  //           |
  //           |
  //  output at index 0 (gradient of T0)
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

  // Configure the output popart Tensor
  void setup() final { outInfo(0) = inInfo(0); }

  std::unique_ptr<Op> clone() const final { return make_unique<CubeOp>(*this); }

  // There is only one Gradient Op for CubeOp, a CubeGradOp
  std::vector<std::unique_ptr<popart::Op>> getGradOps() final {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(new CubeGradOp(*this));
    return upops;
  }

  // An estimate of how valuable sub-graph matching will be
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

// The forward Opx (poplar implementation of the forward Op)

class CubeOpx : public popart::popx::Opx {

public:
  CubeOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {

    // Not strictly necessary, we check that op is castable to a CubeOp *.
    verifyOp<CubeOp>(op, Onnx::CustomOperators::Cube);
  }

  void grow(poplar::program::Sequence &prog) const {

    auto &op = getOp<CubeOp>();

    auto input = getInTensor(0);

    auto output = popops::map(
        graph(),
        popops::expr::Mul(popops::expr::Mul(popops::expr::_1, popops::expr::_1),
                          popops::expr::_1),
        {getInTensor(0)},
        prog,
        debugContext());

    setOutTensor(0, output);
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
                    {getInTensor(0), getInTensor(1)}, // FwdOut, GradOut
                    prog,
                    debugContext()));
  }
};

static popart::popx::OpxCreator<CubeOpx>
    cubeOpxCreator(Onnx::CustomOperators::Cube);
static popart::popx::OpxCreator<CubeGradOpx>
    cubeGradOpxCreator(Onnx::CustomGradOperators::CubeGrad);
}
namespace ONNX_NAMESPACE {

void CubeShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

static const char CubeDoc[] = "Cube cubes (x^3) each element of the tensor.";

ONNX_OPERATOR_SET_SCHEMA_EX(
    Cube,
    comAcme,
    "com.acme",
    1,
    false,
    OpSchema()
        .SetDoc(CubeDoc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(int32)", "tensor(float16)"},
            "Constrain input and output types to signed numeric tensors.")
        .TypeAndShapeInferenceFunction(CubeShapeInference));

static bool registerOps() {
  auto &d = ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance();
  d.AddDomainToVersion("com.acme", 1, 1);

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(comAcme, 1, Cube)>());

  return true;
}

static bool ret = registerOps();

} // namespace ONNX_NAMESPACE
