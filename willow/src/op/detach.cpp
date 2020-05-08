#include <memory>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/detach.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>

namespace popart {

DetachOp::DetachOp(const OperatorIdentifier &_opid,
                   const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> DetachOp::clone() const {
  return std::make_unique<DetachOp>(*this);
}

std::unique_ptr<Op>
DetachOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::DetachInplace) {
    return std::make_unique<DetachInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}
std::vector<std::tuple<OperatorIdentifier, float>>
DetachOp::inplacePriorityDefault() const {
  return {{Onnx::CustomOperators::DetachInplace, 10}};
}

DetachInplaceOp::DetachInplaceOp(const DetachOp &detachOp)
    : DetachOp(Onnx::CustomOperators::DetachInplace, detachOp.settings) {}

std::unique_ptr<Op> DetachInplaceOp::clone() const {
  return std::make_unique<DetachInplaceOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT8,
                                    DataType::UINT16,
                                    DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT8,
                                    DataType::INT16,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT,
                                    DataType::BOOL};

static OpDefinition detachOpDef({OpDefinition::Inputs({{"X", T}}),
                                 OpDefinition::Outputs({{"Y", T}}),
                                 OpDefinition::Attributes({})});

static OpCreator<DetachOp> detachOpCreator(
    OpDefinitions({
        {Onnx::CustomOperators::Detach_1, detachOpDef},
    }),
    [](const OperatorIdentifier &opid_,
       const Op::Settings &settings,
       const Attributes &) -> std::unique_ptr<Op> {
      return std::unique_ptr<Op>(new DetachOp(opid_, settings));
    },
    true);

} // namespace
} // namespace popart
