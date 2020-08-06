#include <cmath>

#include <popart/error.hpp>
#include <popart/op/upsample.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

std::string toString(const UpsampleMode &mode) {
  switch (mode) {
  case UpsampleMode::Nearest:
    return "nearest";
  case UpsampleMode::Linear:
    return "linear";
  case UpsampleMode::N:
  default:
    throw error("Bad UpsampleMode '{}'", static_cast<int>(mode));
  }
}

std::ostream &operator<<(std::ostream &os, const UpsampleMode &x) {
  os << toString(x);
  return os;
}

UpsampleOp::UpsampleOp(const OperatorIdentifier &opid_,
                       const Op::Settings &settings_,
                       UpsampleMode mode_,
                       const std::vector<float> &scales_)
    : Op(opid_, settings_), scales(scales_), mode(mode_) {}

std::unique_ptr<Op> UpsampleOp::clone() const {
  return std::make_unique<UpsampleOp>(*this);
}
void UpsampleOp::setup() {
  auto inputShape = inShape(getInIndex());

  // Check the mode
  if (mode != UpsampleMode::Nearest) {
    throw error("Upsample op only supports the mode 'nearest' at this time.");
  }

  // Sanity check scales
  if (scales.size() != inputShape.size()) {
    throw error("There should be at exactly {inputShape.size()} elements in "
                "upsample op "
                "inputs, scales. Scales has {scales.size()} elements.");
  }

  Shape outputShape;
  for (int i = 0; i < inputShape.size(); i++) {
    float dim = std::floor(inputShape[i] * scales.at(i));
    outputShape.push_back(static_cast<int64_t>(dim));
  }

  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(), outputShape};
}

void UpsampleOp::connectInTensor(InIndex inIndex, TensorId tenId) {
  // Ignore all but the first input
  if (inIndex == getInIndex()) {
    Op::connectInTensor(inIndex, tenId);
  }
}

namespace {

UpsampleMode getUpsampleModeFromString(const std::string &mode) {
  static std::map<std::string, UpsampleMode> modeMap = {
      {"nearest", UpsampleMode::Nearest}, {"linear", UpsampleMode::Linear}};
  auto found = modeMap.find(mode);
  if (found != modeMap.end()) {
    return found->second;
  } else {
    throw error("Unrecognised upsample mode {}", mode);
  }
}

static OpDefinition::DataTypes T1 = {DataType::UINT8,
                                     DataType::UINT16,
                                     DataType::UINT32,
                                     DataType::INT8,
                                     DataType::INT16,
                                     DataType::INT32,
                                     DataType::FLOAT16,
                                     DataType::FLOAT};
static OpDefinition::DataTypes T2 = {DataType::FLOAT};

static OpDefinition upsampleOpDef({OpDefinition::Inputs({
                                       {"X", T1},
                                       {"scales", T2},
                                   }),
                                   OpDefinition::Outputs({{"Y", T1}}),
                                   OpDefinition::Attributes({
                                       {"mode", {"nearest"}},
                                   })});

static OpCreator<UpsampleOp> upsample9_OpCreator(
    OpDefinitions({{Onnx::Operators::Upsample_9, upsampleOpDef}}),
    [](const OpCreatorInfo &info) -> std::unique_ptr<Op> {
      int xInputIndex      = 0;
      int scalesInputIndex = 1;
      auto scalesTensor    = info.getInputTensor(scalesInputIndex);
      if (!scalesTensor->hasTensorData()) {
        throw error("Scales tensor has no data");
      }
      auto inputTensor = info.getInputTensor(xInputIndex);
      int nelms        = static_cast<int>(inputTensor->info.shape().size());
      std::vector<float> scales;
      if (scalesTensor->info.dataType() == DataType::FLOAT) {
        scales = scalesTensor->tensorData()->copyDataAs<float>(nelms);
      } else if (scalesTensor->info.dataType() == DataType::FLOAT16) {
        auto temp = scalesTensor->tensorData()->copyDataAs<float16_t>(nelms);
        for (auto i : temp) {
          scales.push_back(i);
        }
      } else {
        throw error("Can not import scales from tensor of type {}",
                    scalesTensor->info.dataType());
      }

      UpsampleMode mode = UpsampleMode::Nearest;
      if (info.attributes.hasAttribute("mode")) {
        std::string modeString =
            info.attributes.getAttribute<Attributes::String>("mode");
        mode = getUpsampleModeFromString(modeString);
      }

      return std::unique_ptr<Op>(
          new UpsampleOp(info.opid, info.settings, mode, scales));
    },
    true);

} // namespace

} // namespace popart
