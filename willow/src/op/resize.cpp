#include <cmath>

#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/op/resize.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

std::string toString(const ResizeMode &mode) {
  switch (mode) {
  case ResizeMode::Nearest:
    return "nearest";
  case ResizeMode::Linear:
    return "linear";
  case ResizeMode::Cubic:
    return "cubic";
  case ResizeMode::N:
  default:
    throw error("Bad ResizeMode '{}'", static_cast<int>(mode));
  }
}

std::ostream &operator<<(std::ostream &os, const ResizeMode &x) {
  os << toString(x);
  return os;
}

ResizeOp::ResizeOp(const OperatorIdentifier &opid_,
                   const Op::Settings &settings_,
                   ResizeMode mode_,
                   const std::vector<float> &scales_)
    : ResizeOp(opid_,
               settings_,
               mode_,
               scales_,
               ResizeNearestMode::RoundPreferFloor,
               ResizeCoordinateTransformationMode::HalfPixel) {}

ResizeOp::ResizeOp(
    const OperatorIdentifier &opid_,
    const Op::Settings &settings_,
    ResizeMode mode_,
    const std::vector<float> &scales_,
    ResizeNearestMode nearestMode_,
    ResizeCoordinateTransformationMode coordinateTransformationMode_)
    : Op(opid_, settings_), scales(scales_), mode(mode_),
      nearestMode(nearestMode_),
      coordinateTransformationMode(coordinateTransformationMode_) {}

std::unique_ptr<Op> ResizeOp::clone() const {
  return std::make_unique<ResizeOp>(*this);
}
std::vector<std::unique_ptr<Op>> ResizeOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<ResizeGradOp>(*this));
  return upops;
}
void ResizeOp::setup() {
  auto inputShape = inShape(getInIndex());

  // Sanity check scales
  if (scales.size() != inputShape.size()) {
    throw error("The number of dimensions of the resize op scales ({}) must "
                "match the number of dimensions of the input ({})",
                scales.size(),
                inputShape.size());
  }

  Shape outputShape;
  for (int i = 0; i < inputShape.size(); i++) {
    float dim = std::floor(inputShape[i] * scales.at(i));
    outputShape.push_back(static_cast<int64_t>(dim));
  }

  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(), outputShape};
}

namespace {

// We cant just inverse the forward pass scales because of the floor operation.
//   floor(3 * 2.5) = 7
// but
//   floor(7 * (1 / 2.5)) = 2, not 3
std::vector<float> gradScales(const ResizeOp &op) {
  auto inShape  = op.inShape(ResizeOp::getInIndex());
  auto outShape = op.outShape(ResizeOp::getOutIndex());
  std::vector<float> result;
  for (int i = 0; i < inShape.size(); i++) {
    result.push_back(static_cast<float>(inShape.at(i)) / outShape.at(i));
  }
  return result;
}

} // namespace

ResizeGradOp::ResizeGradOp(const ResizeOp &op_)
    : ResizeOp(Onnx::GradOperators::ResizeGrad,
               op_.getSettings(),
               op_.getMode(),
               gradScales(op_)),
      fwdScales(op_.getScales()) {}

const std::vector<GradInOutMapper> &ResizeGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), ResizeOp::getOutIndex(), GradOpInType::GradOut}};
  return inInfo;
}

const std::map<int, int> &ResizeGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), ResizeOp::getInIndex()}};
  return outInfo;
}

namespace {

const int resize10ScalesInIndex = 1;
const int resize11ScalesInIndex = 2;
const int resize11SizesInIndex  = 3;

ResizeMode getResizeModeFromString(const std::string &mode) {
  static std::map<std::string, ResizeMode> modeMap = {
      {"nearest", ResizeMode::Nearest},
      {"linear", ResizeMode::Linear},
      {"cubic", ResizeMode::Cubic}};
  auto found = modeMap.find(mode);
  if (found != modeMap.end()) {
    return found->second;
  } else {
    throw error("Unrecognised resize mode {}", mode);
  }
}

ResizeNearestMode getResizeNearestModeFromString(const std::string &mode) {
  static std::map<std::string, ResizeNearestMode> modeMap = {
      {"round_prefer_floor", ResizeNearestMode::RoundPreferFloor},
      {"round_prefer_ceil", ResizeNearestMode::RoundPreferCeil},
      {"floor", ResizeNearestMode::Floor},
      {"ceil", ResizeNearestMode::Ceil},
      {"pytorch", ResizeNearestMode::Pytorch}};
  auto found = modeMap.find(mode);
  if (found != modeMap.end()) {
    return found->second;
  } else {
    throw error("Unrecognised resize nearest mode {}", mode);
  }
}

std::vector<float> inferScalesFromSizes(const OpCreatorInfo &info) {
  auto inShape   = info.getInputTensorInfo(ResizeOp::getInIndex()).shape();
  auto sizesInfo = info.getInputTensorInfo(resize11SizesInIndex);

  auto outShape = info.getInputData<int64_t>(
      resize11SizesInIndex, {DataType::INT64, DataType::INT32});

  if (inShape.size() != outShape.size()) {
    throw error("ResizeOp, length of sizes ({}) does not match the rank of "
                "input shape ({})",
                outShape.size(),
                inShape.size());
  }

  std::vector<float> scales;
  for (int i = 0; i < inShape.size(); i++) {
    float s = static_cast<float>(outShape.at(i)) / inShape.at(i);
    scales.push_back(s);
  }
  return scales;
}

std::vector<float> getScales(const OpCreatorInfo &info) {
  auto readScalesInput = [&info](int scalesInputIndex) -> std::vector<float> {
    return info.getInputData<float>(scalesInputIndex,
                                    {DataType::FLOAT, DataType::FLOAT16});
  };

  if (info.opid == Onnx::Operators::Resize_10) {
    return readScalesInput(resize10ScalesInIndex);
  } else if (info.opid == Onnx::Operators::Resize_11) {
    if (info.hasInputTensor(resize11ScalesInIndex) &&
        info.hasInputTensor(resize11SizesInIndex)) {
      throw error("Resize op has inputs for `sizes` and `scales`. Only one of "
                  "these tensors should be present. If `size` is needed, "
                  "please pass an empty string as the name of `scales` in the "
                  "input list ( resize([X, roi, '', sizes]) ).");
    }

    if (info.hasInputTensor(resize11ScalesInIndex)) {
      return readScalesInput(resize11ScalesInIndex);
    } else if (info.hasInputTensor(resize11SizesInIndex)) {
      return inferScalesFromSizes(info);
    } else {
      throw error("Resize op has no input for `size` or `scales`. One of these "
                  "tensors needs to be present.");
    }
  } else {
    throw internal_error("Don't know how to set `scalesInputIndex` for {}",
                         info.opid);
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
static OpDefinition::DataTypes T2 = {DataType::FLOAT16, DataType::FLOAT};
static OpDefinition::DataTypes TensorFloat = {DataType::FLOAT};

static OpDefinition resize10_OpDef({OpDefinition::Inputs({
                                        {"X", T1},
                                        {"scales", T2},
                                    }),
                                    OpDefinition::Outputs({{"Y", T1}}),
                                    OpDefinition::Attributes({
                                        {"mode", {"nearest"}},
                                    })});

static OpDefinition
    resize11_OpDef({OpDefinition::Inputs({
                        {"X", T1},
                        {"roi", T2},
                        {"scales", TensorFloat},
                        {"sizes", TensorFloat},
                    }),
                    OpDefinition::Outputs({{"Y", T1}}),
                    OpDefinition::Attributes({{"coordinate_transformation_mode",
                                               {"half_pixel"}},
                                              {"cubic_coeff", {"*"}},
                                              {"exclude_outside", {"0"}},
                                              {"extrapalation_value", {"*"}},
                                              {"mode", {"nearest"}},
                                              {"nearest_mode", {"*"}}})});

template <typename T>
void checkAttribute(const OperatorIdentifier &opid,
                    const Attributes &attr,
                    const std::string &key,
                    const std::vector<T> &acceptableValues) {
  if (attr.hasAttribute(key)) {
    auto v = attr.getAttribute<T>(key);
    for (const auto &value : acceptableValues) {
      if (v == value) {
        return;
      }
    }
    throw error("{}: Unsupported value '{}' for attribute '{}'. Acceptable "
                "values are {}",
                opid,
                v,
                key,
                acceptableValues);
  }
}

ResizeNearestMode getNearestMode(const OpCreatorInfo &info) {
  // Setting the default nearest mode to "pytorch" for Resize10 will
  // preserve existing resize behaviour for poptorch.
  std::string defaultNearestMode = "round_prefer_floor";
  if (info.opid == Onnx::Operators::Resize_10) {
    defaultNearestMode = "pytorch";
  }
  std::string nearestModeString =
      info.attributes.getAttribute<Attributes::String>("nearest_mode",
                                                       defaultNearestMode);
  return getResizeNearestModeFromString(nearestModeString);
}

ResizeCoordinateTransformationMode
getResizeCoordinateTransformationModeFromString(const std::string &mode) {
  static std::map<std::string, ResizeCoordinateTransformationMode> modeMap = {
      {"half_pixel", ResizeCoordinateTransformationMode::HalfPixel},
      {"pytorch_half_pixel",
       ResizeCoordinateTransformationMode::PytorchHalfPixel},
      {"align_corners", ResizeCoordinateTransformationMode::AlignCorners},
      {"asymmetric", ResizeCoordinateTransformationMode::Asymmetric},
      {"tf_crop_and_resize",
       ResizeCoordinateTransformationMode::TfCropAndResize}};
  auto found = modeMap.find(mode);
  if (found != modeMap.end()) {
    return found->second;
  } else {
    throw error("Unrecognised resize coordinate transformation mode {}", mode);
  }
}

static OpCreator<ResizeOp> resize_OpCreator(
    OpDefinitions({{Onnx::Operators::Resize_10, resize10_OpDef},
                   {Onnx::Operators::Resize_11, resize11_OpDef}}),
    [](const OpCreatorInfo &info, Graph &graph) -> Op * {
      std::vector<float> scales = getScales(info);

      const Attributes &attr = info.attributes;

      // Check attributes.
      // The attributes 'cubic_coeff_a' and 'extrapolation_value' dont need to
      // be checked, as we do not support the modes they are used in.
      // checkAttribute<Attributes::String>(
      //     info.opid, attr, "coordinate_transformation_mode", {"half_pixel"});
      checkAttribute<Attributes::Int>(info.opid, attr, "exclude_outside", {0});

      std::string modeString =
          attr.getAttribute<Attributes::String>("mode", "nearest");
      ResizeMode mode = getResizeModeFromString(modeString);

      ResizeNearestMode nearestMode = getNearestMode(info);
      std::string ctmString         = attr.getAttribute<Attributes::String>(
          "coordinate_transformation_mode", "half_pixel");
      auto ctm = getResizeCoordinateTransformationModeFromString(ctmString);

      // Create the op in the graph.
      Op *op = graph.createOp<ResizeOp>(Onnx::CustomOperators::Resize,
                                        info.settings,
                                        mode,
                                        scales,
                                        nearestMode,
                                        ctm);

      // Connect only the first input.
      op->connectInTensor(ResizeOp::getInIndex(), info.getInputIds().at(0));
      op->createAndConnectOutTensor(ResizeOp::getOutIndex(),
                                    info.getOutputIds().at(0));

      return op;
    },
    true);

} // namespace

} // namespace popart
