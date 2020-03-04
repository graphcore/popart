#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/dynamic/dynamicupdate.hpp>
#include <popart/op/dynamic/dynamiczero.hpp>
#include <popart/op/identity.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>

namespace popart {

DynamicZeroOp::DynamicZeroOp(const OperatorIdentifier &_opid,
                             std::vector<int64_t> axes_,
                             std::vector<int64_t> sizes_,
                             bool noOverlap_,
                             const Op::Settings &settings_,
                             TensorInfo updateInInfo_)
    : DynamicBinaryBaseOp(_opid,
                          axes_,
                          sizes_,
                          noOverlap_,
                          settings_,
                          updateInInfo_) {}

std::unique_ptr<Op> DynamicZeroOp::clone() const {
  return std::make_unique<DynamicZeroOp>(*this);
}

std::vector<std::unique_ptr<Op>> DynamicZeroOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<DynamicZeroGradOp>(
      Onnx::CustomGradOperators::DynamicZeroGrad,
      getAxes(),
      getSizes(),
      noOverlap,
      settings));
  return upops;
}

std::unique_ptr<Op>
DynamicZeroOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::DynamicZeroInplace) {
    return std::make_unique<DynamicZeroInplaceOp>(*this);
  }

  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

std::vector<std::tuple<OperatorIdentifier, float>>
DynamicZeroOp::inplacePriorityDefault() const {
  return {{Onnx::CustomOperators::DynamicZeroInplace, 10.f}};
}

DynamicZeroInplaceOp::DynamicZeroInplaceOp(const DynamicZeroOp &dynamicZeroOp)
    : DynamicBinaryBaseInplaceOp(Onnx::CustomOperators::DynamicZeroInplace,
                                 dynamicZeroOp.getAxes(),
                                 dynamicZeroOp.getSizes(),
                                 dynamicZeroOp.isNotOverlapping(),
                                 dynamicZeroOp.getSettings(),
                                 dynamicZeroOp.getUpdateTensorInfo()) {}

DynamicZeroInplaceOp::DynamicZeroInplaceOp(const OperatorIdentifier &_opid,
                                           std::vector<int64_t> axes_,
                                           std::vector<int64_t> sizes_,
                                           bool noOverlap_,
                                           const Op::Settings &settings_,
                                           TensorInfo updateInInfo_)
    : DynamicBinaryBaseInplaceOp(_opid,
                                 axes_,
                                 sizes_,
                                 noOverlap_,
                                 settings_,
                                 updateInInfo_) {}

std::unique_ptr<Op> DynamicZeroInplaceOp::clone() const {
  return std::make_unique<DynamicZeroInplaceOp>(*this);
}

DynamicZeroGradOp::DynamicZeroGradOp(const OperatorIdentifier &_opid,
                                     std::vector<int64_t> axes_,
                                     std::vector<int64_t> sizes_,
                                     bool noOverlap_,
                                     const Op::Settings &settings_)
    : DynamicBinaryBaseOp(_opid, axes_, sizes_, noOverlap_, settings_) {}

std::unique_ptr<Op> DynamicZeroGradOp::clone() const {
  return std::make_unique<DynamicZeroGradOp>(*this);
}

std::unique_ptr<Op> DynamicZeroGradOp::getInplaceVariant(
    const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomGradOperators::DynamicZeroGrad) {
    return std::make_unique<DynamicZeroGradOp>(*this);
  }

  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

const std::vector<GradInOutMapper> &DynamicZeroGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getUpdateInIndex(),
       DynamicBinaryBaseOp::getOutIndex(),
       GradOpInType::GRADOUT},
      {getIndexInIndex(),
       DynamicBinaryBaseOp::getIndexInIndex(),
       GradOpInType::IN}};
  return inInfo;
}

const std::map<int, int> &DynamicZeroGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), DynamicBinaryBaseOp::getUpdateInIndex()}};
  return outInfo;
}

// Creators
static OpDefinition::DataTypes U = {DataType::UINT32};

static OpDefinition::DataTypes T = {DataType::FLOAT,
                                    DataType::FLOAT16,
                                    DataType::INT32,
                                    DataType::UINT32};

static OpDefinition
    dynamicZeroOpDef({OpDefinition::Inputs({{"O", U}, {"X", T}}),
                      OpDefinition::Outputs({{"Z", T}}),
                      OpDefinition::Attributes({
                          {"axes", {"*"}},
                          {"size", {"*"}},
                          {"noOverlap", {"*"}},
                      })});

static OpDefinition
    dynamicUpdateOpDef({OpDefinition::Inputs({{"Y", T}, {"O", U}, {"X", T}}),
                        OpDefinition::Outputs({{"Z", T}}),
                        OpDefinition::Attributes({
                            {"axes", {"*"}},
                            {"size", {"*"}},
                            {"noOverlap", {"*"}},
                        })});

static OpCreator<DynamicUpdateOp> dynamicZeroOpCreator(
    OpDefinitions({{Onnx::CustomOperators::DynamicZero_1, dynamicZeroOpDef}}),
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr = {}) -> std::unique_ptr<Op> {
      std::vector<int64_t> axes  = attr.getAttribute<Attributes::Ints>("axes");
      std::vector<int64_t> sizes = attr.getAttribute<Attributes::Ints>("sizes");
      bool noOverlap             = attr.hasAttribute("noOverlap") &&
                       attr.getAttribute<Attributes::Int>("noOverlap");
      return std::unique_ptr<DynamicZeroOp>(
          new DynamicZeroOp(_opid, axes, sizes, noOverlap, settings));
    },
    true);
} // namespace popart