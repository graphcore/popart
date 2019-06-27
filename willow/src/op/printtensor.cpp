#include <memory>
#include <poponnx/op/identity.hpp>
#include <poponnx/op/printtensor.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>

namespace poponnx {

PrintTensorOp::PrintTensorOp(const OperatorIdentifier &opid_,
                             bool printSelf_,
                             bool printGradient_,
                             const Op::Settings &settings_)
    : ElementWiseUnaryOp(opid_, settings_), printSelf(printSelf_),
      printGradient(printGradient_) {}

std::unique_ptr<Op> PrintTensorOp::clone() const {
  return std::make_unique<PrintTensorOp>(*this);
}

void PrintTensorOp::appendAttributes(OpSerialiserBase &os) const {
  ElementWiseUnaryOp::appendAttributes(os);
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
static OpCreator<PrintTensorOp> printtensorOpCreator(
    Onnx::CustomOperators::PrintTensor_1,
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

} // namespace poponnx
