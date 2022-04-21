// cppimport
// NOTE: the cppimport comment is necessary for dynamic compilation when loading
// the python module!
// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// This example demonstrates how to create a custom operator for PopART and
// PopXL, in this case a Leaky ReLU op that returns `x` for any element `x >= 0`
// and `x * alpha` for any element `x < 0`, where `alpha` is provided as a
// scalar attribute to the operator.

#include <popops/ElementWise.hpp>
#include <popart/op/custom/parameterizedop.hpp>
#include <popart/op/custom/parameterizedopbinder.hpp>

namespace py = pybind11;

namespace popart {
/**
 * @brief Leaky relu custom op parameters.
 *
 * This structure is encapsulating the parameters/attributes logic, such that it
 * can be shared between FWD and GRAD op implementations.
 */
struct LeakyReluParams {
  float alpha = 1e-2;
  /**
   * @brief Append custom op parameters to op serialiser.
   *
   * @param os The serialised op to add the attributes to.
   */
  void appendAttributes(popart::OpSerialiserBase &os) const {
    os.appendAttribute("alpha", this->alpha);
  }

  /**
   * @brief Build from PopART attributes. Using default values if no info
   * provided.
   *
   * @param attributes The attributes to use to build the parameters with.
   * \return LeakyReluParams The object encapsulating the leaky relu parameters.
   */
  static LeakyReluParams
  makeFromAttributes(const popart::Attributes &attributes) {
    auto params  = LeakyReluParams();
    params.alpha = attributes.getAttribute<popart::Attributes::Float>(
        "alpha", params.alpha);
    return params;
  }
};

/**
 * @brief Leaky Relu grad op.
 */
class LeakyReluGradOp
    : public ParameterizedOp<LeakyReluGradOp, LeakyReluParams> {
public:
  using ParameterizedOp<LeakyReluGradOp, LeakyReluParams>::ParameterizedOp;

  static popart::OperatorIdentifier defaultOperatorId() {
    return popart::OperatorIdentifier{
        "custom.ops", "LeakyReluGrad", 1, {1, 1}, 1};
  }
  void setup() final { outInfo(0) = inInfo(0); };

  const std::vector<popart::GradInOutMapper> &gradInputInfo() const {
    static const std::vector<popart::GradInOutMapper> inInfo = {
        {0, 0, popart::GradOpInType::GradOut},
        {1, 0, popart::GradOpInType::In}};
    return inInfo;
  }
  const std::map<int, int> &gradOutToNonGradIn() const {
    // The Grad Op has 1 output, which is the gradient of the only input
    static const std::map<int, int> outInfo = {{0, 0}};
    return outInfo;
  }
};

/**
 * @brief Leaky Relu op.
 */
class LeakyReluOp : public ParameterizedOp<LeakyReluOp, LeakyReluParams> {
public:
  using ParameterizedOp<LeakyReluOp, LeakyReluParams>::ParameterizedOp;

  static popart::OperatorIdentifier defaultOperatorId() {
    return popart::OperatorIdentifier{"custom.ops", "LeakyRelu", 1, {1, 1}, 1};
  }
  void setup() final { outInfo(0) = inInfo(0); }

  std::vector<std::unique_ptr<popart::Op>> getGradOps() {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(new LeakyReluGradOp(*this));
    return upops;
  }
};

namespace CustomOperators {
const popart::OperatorIdentifier LeakyReluId = LeakyReluOp::defaultOperatorId();
} // namespace CustomOperators
namespace CustomGradOperators {
const popart::OperatorIdentifier LeakyReluGradId =
    LeakyReluGradOp::defaultOperatorId();
} // namespace CustomGradOperators

namespace {
// Registering in PopART.

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition
    leakyReluOpDef({OpDefinition::Inputs({{"input", T}}),
                    OpDefinition::Outputs({{"output", T}}),
                    OpDefinition::Attributes({{"alpha", {"*"}}})});

static popart::OpCreator<LeakyReluOp> leakyReluOpCreator(
    popart::OpDefinitions({{CustomOperators::LeakyReluId, leakyReluOpDef}}),
    &LeakyReluOp::createOpFromCreatorInfo,
    true);
} // namespace

namespace pe = popops::expr;

// Leaky Relu ops implementation.
class LeakyReluOpx : public popart::popx::Opx {
public:
  LeakyReluOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<LeakyReluOp>(op, {CustomOperators::LeakyReluId});
  }

  void grow(poplar::program::Sequence &prog) const final {
    auto op              = getOp<LeakyReluOp>();
    poplar::Tensor input = getInTensor(0);
    const float alpha    = op.params().alpha;
    // x < 0.0f ? alpha * x : x
    // pe::_1 here are placeholders for the argument for the expression.
    // \sa popops::expr
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

class LeakyReluGradOpx : public popart::popx::Opx {
public:
  LeakyReluGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<LeakyReluGradOp>(op, {CustomGradOperators::LeakyReluGradId});
  }

  void grow(poplar::program::Sequence &prog) const final {
    auto op              = getOp<LeakyReluGradOp>();
    poplar::Tensor grad  = getInTensor(0);
    poplar::Tensor input = getInTensor(1);

    const float alpha = op.params().alpha;
    // (grad * (x < 0.0f ? alpha : 1))
    pe::Mul expression = pe::Mul(pe::Select(pe::Const(alpha),
                                            pe::Const(1.0f),
                                            pe::Lt(pe::_2, pe::Const(0.0f))),
                                 pe::_1);
    auto output        = popops::map(graph(),
                              expression,
                              {grad, input},
                              prog,
                              debugContext("LeakyReluGrad"),
                              poplar::OptionFlags());
    setOutTensor(0, output);
  }
};

// Necessary for registering in PopART
static popart::popx::OpxCreator<LeakyReluOpx>
    LeakyReluOpxCreator({CustomOperators::LeakyReluId});
static popart::popx::OpxCreator<LeakyReluGradOpx>
    LeakyReluGradOpxCreator({CustomGradOperators::LeakyReluGradId});

/**
 * @brief Pybind11 custom op module declaration.
 * NOTE: make sure the name of the module correspond to the source filename!
 */
// cppcheck-suppress syntaxError
PYBIND11_MODULE(leaky_relu_op_impl, m) {
  // Bindings the parameters of the op: constructor + fields.
  py::class_<LeakyReluParams>(m, "LeakyReluParams")
      .def(py::init<float>(), py::arg("alpha") = 0.01)
      .def_readwrite("alpha", &LeakyReluParams::alpha);

  // Helper function to make the custom op bindings automatically (once the
  // params are pybinded).
  _internal::ir::op::makeParameterizedOpBindings<LeakyReluOp>(m, "LeakyRelu");
}
} // namespace popart

// clang-format off
// cppimport configuration for compiling the pybind11 module.
/*
<%
cfg['extra_compile_args'] = ['-std=c++14', '-fPIC', '-shared', '-O3', '-DONNX_NAMESPACE=onnx']
cfg['libraries'] = ['popart']
setup_pybind11(cfg)
%>
*/
