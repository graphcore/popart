// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

//
// This example demonstrates how to create a custom operator for PopART, in this
// case a Leaky ReLU op that returns `x` for any element `x >= 0` and `x *
// alpha` for any element `x < 0`, where `alpha` is provided as a scalar
// attribute to the operator.

#include <memory>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

#include <onnx/defs/schema.h>
#include <onnx/defs/shape_inference.h>

namespace {

// for C++11 compatibility, we don't use make_unique
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

} // namespace

//! [OpId begin]
namespace CustomOperators {
const popart::OperatorIdentifier LeakyReluId = {"custom.ops", "LeakyRelu", 1};
} // namespace CustomOperators
namespace CustomGradOperators {
const popart::OperatorIdentifier LeakyReluGradId = {"custom.ops",
                                                    "LeakyReluGrad",
                                                    1};
} // namespace CustomGradOperators
//! [OpId end]

class LeakyReluOp;
class LeakyReluOpx;
class LeakyReluGradOpx;

//! [GradOp begin]
class LeakyReluGradOp : public popart::Op {
public:
  LeakyReluGradOp(const LeakyReluOp &fwdOp);

  std::unique_ptr<popart::Op> clone() const final {
    return make_unique<LeakyReluGradOp>(*this);
  }
  void setup() final { outInfo(0) = inInfo(0); };

  const std::vector<popart::GradInOutMapper> &gradInputInfo() const override;

  // The Grad Op has 1 output, which is the gradient of the only input
  const std::map<int, int> &gradOutToNonGradIn() const override;

  bool requiresRandomSeed() const override { return false; }

  // an estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  float getAlpha() const { return alpha; }

  // Implementation defined below
  void appendAttributes(popart::OpSerialiserBase &os) const override;

  // Implementation defined below
  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override;

private:
  float alpha;
};
//! [GradOp end]

//! [Op begin]
class LeakyReluOp : public popart::Op {
public:
  LeakyReluOp(const popart::OperatorIdentifier &_opid,
              float _alpha,
              const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_), alpha(_alpha) {}

  std::unique_ptr<Op> clone() const final {
    return make_unique<LeakyReluOp>(*this);
  }

  void setup() final { outInfo(0) = inInfo(0); }

  void appendAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendAttributes(os);
    os.appendAttribute("alpha", getAlpha());
  }

  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendOutlineAttributes(os);
    os.appendAttribute("alpha", getAlpha());
  }

  std::vector<std::unique_ptr<popart::Op>> getGradOps() override {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(new LeakyReluGradOp(*this));
    return upops;
  }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  bool requiresRandomSeed() const override { return false; }

  // Attributes
  float getAlpha() const { return alpha; }

private:
  float alpha;
};
//! [Op end]

//! [OpCreator begin]
namespace {
using popart::DataType;
using popart::OpDefinition;

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition
    leakyReluOpDef({OpDefinition::Inputs({{"input", T}}),
                    OpDefinition::Outputs({{"output", T}}),
                    OpDefinition::Attributes({{"alpha", {"*"}}})});

static popart::OpCreator<LeakyReluOp> leakyReluOpCreator(
    popart::OpDefinitions({{CustomOperators::LeakyReluId, leakyReluOpDef}}),
    [](const popart::OpCreatorInfo &info) {
      // default alpha is 10**(-2)
      float alpha = info.attributes.getAttribute<popart::Attributes::Float>(
          "alpha", 1e-2f);
      return make_unique<LeakyReluOp>(info.opid, alpha, info.settings);
    },
    true);
} // namespace
//! [OpCreator end]

namespace pe = popops::expr;

//! [Opx begin]
class LeakyReluOpx : public popart::popx::Opx {
public:
  LeakyReluOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<LeakyReluOp>(op, {CustomOperators::LeakyReluId});
  }

  void grow(poplar::program::Sequence &prog) const final {

    auto op = getOp<LeakyReluOp>();

    poplar::Tensor input = getInTensor(0);

    float alpha = op.getAlpha();

    // x < 0.0f ? alpha * x : x
    auto expression = pe::Select(pe::Mul(pe::Const(alpha), pe::_1),
                                 pe::_1,
                                 pe::Lt(pe::_1, pe::Const(0.0f)));

    popops::mapInPlace(graph(),
                       expression,
                       {input},
                       prog,
                       debugContext("LeakyRelu"),
                       poplar::OptionFlags());

    setOutTensor(0, input);
  }
};
//! [Opx end]

//! [GradOpx begin]
class LeakyReluGradOpx : public popart::popx::Opx {
public:
  LeakyReluGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<LeakyReluGradOp>(op, {CustomGradOperators::LeakyReluGradId});
  }

  void grow(poplar::program::Sequence &prog) const final {

    auto op = getOp<LeakyReluGradOp>();

    poplar::Tensor grad  = getInTensor(0);
    poplar::Tensor input = getInTensor(1);

    float alpha = op.getAlpha();

    // (grad * (x < 0.0f ? alpha : 1))
    pe::Mul expression = pe::Mul(pe::Select(pe::Const(alpha),
                                            pe::Const(1.0f),
                                            pe::Lt(pe::_2, pe::Const(0.0f))),
                                 pe::_1);

    auto output = popops::map(graph(),
                              expression,
                              {grad, input},
                              prog,
                              debugContext("LeakyReluGrad"),
                              poplar::OptionFlags());

    setOutTensor(0, output);
  }
};
//! [GradOpx end]

LeakyReluGradOp::LeakyReluGradOp(const LeakyReluOp &fwdOp)
    : popart::Op(CustomGradOperators::LeakyReluGradId, fwdOp.settings),
      alpha(fwdOp.getAlpha()) {}

const std::vector<popart::GradInOutMapper> &
LeakyReluGradOp::gradInputInfo() const {
  static const std::vector<popart::GradInOutMapper> inInfo = {
      {0, 0, popart::GradOpInType::GradOut}, {1, 0, popart::GradOpInType::In}};
  return inInfo;
}

// The Grad Op has 1 output, which is the gradient of the only input
const std::map<int, int> &LeakyReluGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, 0}};
  return outInfo;
}

void LeakyReluGradOp::appendAttributes(popart::OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("alpha", getAlpha());
}

void LeakyReluGradOp::appendOutlineAttributes(
    popart::OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("alpha", getAlpha());
}

//! [OpxCreator begin]
static popart::popx::OpxCreator<LeakyReluOpx>
    LeakyReluOpxCreator({CustomOperators::LeakyReluId});
static popart::popx::OpxCreator<LeakyReluGradOpx>
    LeakyReluGradOpxCreator({CustomGradOperators::LeakyReluGradId});
//! [OpxCreator end]

//! [Onnx begin]
namespace ONNX_NAMESPACE {

void LeakyReluShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

static const char LeakyReluDoc[] =
    "See https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html .";

ONNX_OPERATOR_SET_SCHEMA_EX(
    LeakyRelu,
    comAcme,
    "custom.ops",
    1,
    false,
    OpSchema()
        .SetDoc(LeakyReluDoc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(int32)", "tensor(float16)"},
            "Constrain input and output types to signed numeric tensors.")
        .TypeAndShapeInferenceFunction(LeakyReluShapeInference));

static bool registerOps() {
  auto &d = ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance();
  d.AddDomainToVersion("custom.ops", 1, 1);

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          comAcme, 1, LeakyRelu)>());

  return true;
}

static bool ret = registerOps();

} // namespace ONNX_NAMESPACE

//! [Onnx end]
