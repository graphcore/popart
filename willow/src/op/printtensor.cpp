#include <memory>
#include <popart/op/identity.hpp>
#include <popart/op/printtensor.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>

namespace popart {

PrintTensorOp::PrintTensorOp(const OperatorIdentifier &opid_,
                             bool printSelf_,
                             bool printGradient_,
                             const Op::Settings &settings_)
    : ElementWiseUnaryOp(opid_, settings_), printSelf(printSelf_),
      printGradient(printGradient_) {}

std::unique_ptr<Op> PrintTensorOp::clone() const {
  return std::make_unique<PrintTensorOp>(*this);
}

void PrintTensorOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  ElementWiseUnaryOp::appendOutlineAttributes(os);
  os.appendAttribute("printSelf", printSelf);
  os.appendAttribute("printGradient", printGradient);
}

std::vector<std::unique_ptr<Op>> PrintTensorOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(
      std::make_unique<PrintTensorOp>(Onnx::CustomOperators::PrintTensor_1,
                                      printGradient,
                                      printGradient,
                                      getSettings()));
  return upops;
}

const std::vector<GradInOutMapper> &PrintTensorOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), PrintTensorOp::getOutIndex(), GradOpInType::GRADOUT}};
  return inInfo;
}

const std::map<int, int> &PrintTensorOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), PrintTensorOp::getInIndex()}};

  return outInfo;
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

static OpDefinition printTensorOpDef({OpDefinition::Inputs({{"X", T}}),
                                      OpDefinition::Outputs({{"Y", T}}),
                                      OpDefinition::Attributes({
                                          {"print_gradient", {"*"}},
                                      })});

static OpCreator<PrintTensorOp> printtensorOpCreator(
    OpDefinitions({
        {Onnx::CustomOperators::PrintTensor_1, printTensorOpDef},
    }),
    [](const OperatorIdentifier &opid_,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      bool printGradient =
          attr.getAttribute<Attributes::Int>("print_gradient", true) != 0;

      return std::unique_ptr<Op>(
          new PrintTensorOp(opid_, true, printGradient, settings));
    },
    true);
} // namespace

} // namespace popart
