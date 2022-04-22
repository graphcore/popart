// cppimport
// NOTE: the cppimport comment is necessary for dynamic compilation when loading
// the python module!
// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// This example demonstrates how to create a custom operator for PopART and
// PopXL, in this case a Leaky ReLU op that returns `x` for any element `x >= 0`
// and `x * alpha` for any element `x < 0`, where `alpha` is provided as a
// scalar attribute to the operator.

#include <algorithm>
#include <initializer_list>
#include <map>
#include <memory>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <string>
#include <vector>
#include <poplar/OptionFlags.hpp>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popart/op/custom/parameterizedop.hpp>
#include <popart/op/custom/parameterizedopbinder.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/opmanager.hpp"
#include "popart/tensorinfo.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace py = pybind11;

namespace popart {
class LeakyReluOp;
namespace popx {
class Devicex;
} // namespace popx

/**
 * @brief Struct to encapsulate Leaky ReLU parameters.
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
 * @brief Leaky ReLU gradient op.
 */
class LeakyReluGradOp
    : public ParameterizedOp<LeakyReluGradOp, LeakyReluParams> {
public:
  // Inherit constructor from ParameterizedOp<LeakyReluGradOp, LeakyReluParams>.
  using ParameterizedOp<LeakyReluGradOp, LeakyReluParams>::ParameterizedOp;

  /**
   * Create an operator identifier for LeakyReluOp.
   */
  static popart::OperatorIdentifier defaultOperatorId() {
    return popart::OperatorIdentifier{
        "custom.ops", "LeakyReluGrad", 1, {1, 1}, 1};
  }

  /**
   * @brief Determine the shapes and type of output tensors.
   */
  void setup() override { outInfo(0) = inInfo(0); };

  const std::vector<popart::GradInOutMapper> &gradInputInfo() const override {
    static const std::vector<popart::GradInOutMapper> inInfo = {
        {0, 0, popart::GradOpInType::GradOut},
        {1, 0, popart::GradOpInType::In}};
    return inInfo;
  }

  /**
   * @brief Return mapping to associated the outputs indices of LeakyReluGradOp
   * with inputs of the LeakyReluOp. \return const std::map<int, int>&
   */
  const std::map<int, int> &gradOutToNonGradIn() const override {
    // The Grad Op has 1 output, which is the gradient of the only input
    static const std::map<int, int> outInfo = {{0, 0}};
    return outInfo;
  }
};

/**
 * @brief Leaky ReLU op.
 */
class LeakyReluOp : public ParameterizedOp<LeakyReluOp, LeakyReluParams> {
public:
  // Inherit constructor from ParameterizedOp<LeakyReluGradOp, LeakyReluParams>.
  using ParameterizedOp<LeakyReluOp, LeakyReluParams>::ParameterizedOp;

  /**
   * Create an operator identifier for LeakyReluOp.
   */
  static popart::OperatorIdentifier defaultOperatorId() {
    return popart::OperatorIdentifier{"custom.ops", "LeakyRelu", 1, {1, 1}, 1};
  }

  /**
   * @brief Determine the shapes and type of output tensors.
   */
  void setup() override { outInfo(0) = inInfo(0); }

  /**
   * @brief Construct associated gradient operations.
   * \return std::vector<std::unique_ptr<popart::Op>>
   */
  std::vector<std::unique_ptr<popart::Op>> getGradOps() override {
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

/**
 * Leaky ReLU implementation.
 */
class LeakyReluOpx : public popart::popx::Opx {
public:
  LeakyReluOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<LeakyReluOp>(op, {CustomOperators::LeakyReluId});
  }

  /**
   * @brief Add the Poplar code for this operation to a Poplar sequence.
   * \param prog The Poplar sequence to add the operation to.
   */
  void grow(poplar::program::Sequence &prog) const override {
    auto op              = getOp<LeakyReluOp>();
    poplar::Tensor input = getInTensor(0);
    const float alpha    = op.params().alpha;
    // x < 0.0f ? alpha * x : x
    // pe::_1 here is a placeholder for the argument of the expression.
    auto expression = pe::Select(pe::Mul(pe::Const(alpha), pe::_1),
                                 pe::_1,
                                 pe::Lt(pe::_1, pe::Const(0.0f)));
    auto output     = popops::map(graph(),
                              expression,
                              {input},
                              prog,
                              debugContext("LeakyRelu"),
                              poplar::OptionFlags());
    setOutTensor(0, output);
  }
};

/**
 * Leaky ReLU gradient operation implementation.
 */
class LeakyReluGradOpx : public popart::popx::Opx {
public:
  LeakyReluGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<LeakyReluGradOp>(op, {CustomGradOperators::LeakyReluGradId});
  }

  /**
   * @brief Add the Poplar code for this operation to a Poplar sequence.
   * \param prog The Poplar sequence to add the operation to.
   */
  void grow(poplar::program::Sequence &prog) const override {
    auto op              = getOp<LeakyReluGradOp>();
    poplar::Tensor grad  = getInTensor(0);
    poplar::Tensor input = getInTensor(1);

    const float alpha = op.params().alpha;
    // (grad * (x < 0.0f ? alpha : 1))
    // pe::_1 and pe::_2 are placeholders for the arguments of the expression.
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
  popart::ir::op::makeParameterizedOpBindings<LeakyReluOp>(m, "LeakyRelu");
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
