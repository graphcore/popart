#include <popart/ir.hpp>
#include <popart/op/convtranspose.hpp>
#include <popart/opmanager.hpp>

namespace popart {

ConvTransposeOp::ConvTransposeOp(const OperatorIdentifier &_opid,
                                 const Settings &settings_,
                                 std::vector<int64_t> strides_,
                                 std::vector<int64_t> pads_,
                                 std::vector<int64_t> dilations_,
                                 int64_t group_,
                                 const AutoPad &padType_,
                                 std::vector<int64_t> outputPadding_,
                                 std::vector<int64_t> outputShape_,
                                 const MultiConvOptions &convOpts_)
    : Op(_opid, settings_), strides(strides_), pads(pads_),
      dilations(dilations_), group(group_), padType(padType_),
      outputPadding(outputPadding_), outputShape(outputShape_),
      convOpts(convOpts_) {}

std::unique_ptr<Op> ConvTransposeOp::clone() const {
  return std::make_unique<ConvTransposeOp>(*this);
}

void ConvTransposeOp::setup() {
  Shape inputShape            = inShape(ConvTransposeOp::getInIndex());
  Shape kernelShape           = inShape(ConvTransposeOp::getWeightsInIndex());
  bool output_shape_specified = false;

  const auto nSpatialDims = inputShape.size() - 2;
  // Set the default values.
  strides.resize(nSpatialDims, 1);
  outputPadding.resize(nSpatialDims, 0);
  dilations.resize(nSpatialDims, 1);
  pads.resize(nSpatialDims * 2, 0);

  // Check dilations.
  if (!std::all_of(dilations.begin(), dilations.end(), [](int64_t i) {
        return i == 1;
      })) {
    throw error("Non default value for dilations is not supported.");
  }

  Shape outShape{inputShape.at(0), kernelShape.at(1) * group};
  if (outputShape.size() == 0) {
    for (int i = 0; i < strides.size(); i++) {
      int64_t x = strides.at(i) * (inputShape.at(i + 2) - 1) +
                  outputPadding.at(i) + ((kernelShape.at(i + 2) - 1) + 1) -
                  pads.at(i) - pads.at(nSpatialDims + i);
      outShape.push_back(x);
    }
  } else if (outputShape.size() == nSpatialDims) {
    output_shape_specified = true;
    outShape.push_back(outputShape[0]);
    outShape.push_back(outputShape[1]);
  } else {
    outShape = outputShape;
  }

  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(), outShape};

  // Now generate the params
  params.type      = inInfo(getInIndex()).dataType();
  params.batchSize = inShape(getInIndex()).at(0);

  params.numInChannelsPerGroup  = inputShape.at(1) / group;
  params.numOutChannelsPerGroup = kernelShape.at(1);
  params.numGroups              = group;

  inputShape = inShape(getInIndex());
  for (int i = 2; i < inputShape.size(); i++) {
    params.inputShape.push_back(inputShape.at(i));
  }

  kernelShape = inShape(getWeightsInIndex());
  for (int i = 2; i < kernelShape.size(); i++) {
    params.kernelShape.push_back(kernelShape.at(i));
  }

  std::vector<int64_t> zeroes(nSpatialDims, 0);
  std::vector<int64_t> ones(nSpatialDims, 1);
  std::vector<bool> falses(nSpatialDims, false);

  params.inputTransformation.lowerTruncation = zeroes;
  params.inputTransformation.upperTruncation = zeroes;
  params.inputTransformation.dilation        = strides;
  params.inputTransformation.flip            = falses;

  for (int i = 0; i < nSpatialDims; i++) {
    int64_t x = inShape(getWeightsInIndex()).at(i + 2) - 1;
    if (output_shape_specified) {
      // The ONNX doc states that the shape of the output can be explicitly set
      // which will cause pads values to be auto generated
      int64_t total_padding = strides.at(i) * (inputShape[i + 2] - 1) +
                              outputPadding.at(i) + (x * dilations[i] + 1) -
                              outShape[i + 2];

      // The padding argument effectively adds:
      // dilation * (kernel_size - 1) - padding amount of zero padding to both
      // sizes of the input. (source: Pytorch doc)
      if (padType == AutoPad::SAME_UPPER) {
        // When total_padding is an odd number and pad=SAME_UPPER, the extra
        // padding is added at the end
        params.inputTransformation.lowerPadding.push_back(
            x - (total_padding - ((total_padding) / 2)));
        params.inputTransformation.upperPadding.push_back(
            (x - (total_padding) / 2));
      } else if (padType == AutoPad::SAME_LOWER || padType == AutoPad::NOTSET) {
        // Defaulting to "SAME_LOWER" when no auto_pad option is set
        params.inputTransformation.lowerPadding.push_back(
            x - (total_padding - ((total_padding + 1) / 2)));
        params.inputTransformation.upperPadding.push_back(
            (x - (total_padding + 1) / 2));
      } else {
        throw error("`VALID` type for auto_pad cannot be used"
                    " if the `output_shape` is also set");
      }
    }

    else {
      params.inputTransformation.lowerPadding.push_back(x - pads.at(i));
      params.inputTransformation.upperPadding.push_back(
          x + outputPadding.at(i) - pads.at(nSpatialDims + i));
    }
  }

  params.kernelTransformation.lowerTruncation = zeroes;
  params.kernelTransformation.upperTruncation = zeroes;
  params.kernelTransformation.dilation        = ones;
  params.kernelTransformation.lowerPadding    = zeroes;
  params.kernelTransformation.upperPadding    = zeroes;
  params.kernelTransformation.flip            = falses;

  params.outputTransformation.lowerTruncation = zeroes;
  params.outputTransformation.upperTruncation = zeroes;
  params.outputTransformation.stride          = ones;
  params.outputTransformation.lowerPadding    = zeroes;
  params.outputTransformation.upperPadding    = zeroes;
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition convtranspose_OpDef({OpDefinition::Inputs({
                                             {"X", T},
                                             {"W", T},
                                             {"B", T},
                                         }),
                                         OpDefinition::Outputs({{"Y", T}}),
                                         OpDefinition::Attributes({
                                             {"auto_pad", {"NOTSET"}},
                                             {"dilations", {"*"}},
                                             {"group", {"*"}},
                                             {"kernel_shape", {"*"}},
                                             {"output_padding", {"*"}},
                                             {"output_shape", {"*"}},
                                             {"pads", {"*"}},
                                             {"strides", {"*"}},
                                         })});

static OpCreator<ConvTransposeOp> convtranspose_OpCreator(
    OpDefinitions({{Onnx::Operators::ConvTranspose_1, convtranspose_OpDef},
                   {Onnx::Operators::ConvTranspose_11, convtranspose_OpDef}}),
    [](const OpCreatorInfo &info) -> std::unique_ptr<Op> {
      // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      auto &attr = info.attributes;

      auto padType =
          attr.getAttribute<Attributes::String>("auto_pad", "NOTSET");
      auto dilations = attr.getAttribute<Attributes::Ints>("dilations", {});
      auto group     = attr.getAttribute<Attributes::Int>("group", 1);
      // check kernel_shape?
      auto outputPadding =
          attr.getAttribute<Attributes::Ints>("output_padding", {});
      auto outputShape =
          attr.getAttribute<Attributes::Ints>("output_shape", {});
      auto pads    = attr.getAttribute<Attributes::Ints>("pads", {});
      auto strides = attr.getAttribute<Attributes::Ints>("strides", {});

      auto sessOpts =
          info.settings.getIr().getSessionOptions().convolutionOptions;
      auto convOpts = MultiConvOptions(sessOpts, attr);

      return std::make_unique<ConvTransposeOp>(
          info.opid,
          info.settings,
          strides,
          pads,
          dilations,
          group,
          HasReceptiveFieldOp::getAutoPad(padType),
          outputPadding,
          outputShape,
          convOpts);
      // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    },
    true);

} // namespace

} // namespace popart
