// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <popart/op/printtensor.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"

namespace popart {
struct OperatorIdentifier;

PrintTensorOp::PrintTensorOp(const OperatorIdentifier &opid_,
                             bool printSelf_,
                             bool printGradient_,
                             const std::string &title_,
                             const Op::Settings &settings_)
    : ElementWiseUnaryOp(opid_, settings_), printSelf(printSelf_),
      printGradient(printGradient_), title(title_) {}

std::unique_ptr<Op> PrintTensorOp::clone() const {
  return std::make_unique<PrintTensorOp>(*this);
}

void PrintTensorOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  ElementWiseUnaryOp::appendOutlineAttributes(os);
  os.appendAttribute("printSelf", printSelf);
  os.appendAttribute("printGradient", printGradient);
  os.appendAttribute("title", title);
}

std::vector<std::unique_ptr<Op>> PrintTensorOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  auto inputId = inTensor(getInIndex())->id;
  // If title is specifed, grad title is title + "_gradient"
  // otherwise its "Gradient__" + inputTensor
  auto newTitle =
      title.empty() ? reservedGradientPrefix() + inputId : title + "_gradient";
  upops.emplace_back(
      std::make_unique<PrintTensorOp>(Onnx::CustomOperators::PrintTensor_1,
                                      printGradient,
                                      printGradient,
                                      newTitle,
                                      getSettings()));
  return upops;
}

const std::vector<GradInOutMapper> &PrintTensorOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), PrintTensorOp::getOutIndex(), GradOpInType::GradOut}};
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

static OpDefinition printTensorOpDef(
    {OpDefinition::Inputs({{"X", T}}),
     OpDefinition::Outputs({{"Y", T}}),
     OpDefinition::Attributes({{"print_gradient", {"*"}}, {"title", {"*"}}})});

static OpCreator<PrintTensorOp> printtensorOpCreator(
    OpDefinitions({
        {Onnx::CustomOperators::PrintTensor_1, printTensorOpDef},
    }),
    [](const OpCreatorInfo &info) {
      bool printGradient = info.attributes.getAttribute<Attributes::Int>(
                               "print_gradient", true) != 0;
      std::string title =
          info.attributes.getAttribute<Attributes::String>("title", "");

      return std::unique_ptr<Op>(new PrintTensorOp(
          info.opid, true, printGradient, title, info.settings));
    },
    true);
} // namespace

} // namespace popart
