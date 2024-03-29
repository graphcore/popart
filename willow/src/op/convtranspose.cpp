// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "popart/util.hpp"
#include "popart/util/float8util.hpp"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <popart/ir.hpp>
#include <popart/op/convtranspose.hpp>
#include <popart/opmanager.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/convbase.hpp"
#include "popart/op/receptive.hpp"
#include "popart/operators.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
struct OperatorIdentifier;

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
    : Op(_opid, settings_), strides(strides_), dilations(dilations_),
      group(group_), padType(padType_), convOpts(convOpts_), pads(pads_),
      outputPadding(outputPadding_), outputShape(outputShape_) {}

std::unique_ptr<Op> ConvTransposeOp::clone() const {
  return std::make_unique<ConvTransposeOp>(*this);
}

void ConvTransposeOp::setup() {
  // validate that either a valid set of pow2 scaled inputs is provided,
  // or all inputs are not related to the pow2 scaled variant of this op.
  validateOpFloat8Inputs(input.get(), getLog2ScaleInIndex(), debugName());

  // The non-optional 'group' argument can always be determined based
  // on input shapes. Check that they match
  if (group < 1) {
    throw error("group attribute in {} must be greater than zero", debugName());
  }

  auto dataInInfo = inInfo(ConvTransposeOp::getInIndex());
  auto weightInfo = inInfo(ConvTransposeOp::getWeightsInIndex());
  if (dataInInfo.dim(1) != weightInfo.dim(0)) {
    throw error("Unexpected number of channels in the input tensor: {}. Given "
                "non-transposed weight "
                "tensor of shape {}, transposed convolution expects input "
                "tensor of size {}"
                " to have {} channels.",
                dataInInfo.dim(1),
                weightInfo.shape(),
                dataInInfo.shape(),
                weightInfo.dim(0));
  }

  const Shape inputShape  = inShape(getInIndex());
  const Shape kernelShape = inShape(getWeightsInIndex());

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

  // The output shape may or may not be specified
  bool output_shape_specified = false;
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

  DataType outType;
  if (isPow2ScaledConvTranspose()) {
    outType = DataType::FLOAT16;
  } else {
    outType = inInfo(getInIndex()).dataType();
  }
  outInfo(getOutIndex()) = {outType, outShape};

  // Now calculate the padding
  std::vector<int64_t> lowerPadding;
  std::vector<int64_t> upperPadding;
  for (int dim = 0; dim < nSpatialDims; dim++) {
    // (kernel size - 1)*dilations
    // (currently dilation always == 1) but this is the correct maths
    int64_t k_minus_one_times_d =
        (inShape(getWeightsInIndex()).at(dim + 2) - 1) * dilations[dim];

    if (output_shape_specified) {
      // The ONNX doc states that the shape of the output can be explicitly set
      // which will cause pads values to be auto generated

      // The padding argument results in:
      // dilation * (kernel_size - 1) - padding amount of zero padding to both
      // sizes of the input. (source: Pytorch doc)
      // This is so that it is the inverse of the equivalent convolution with
      // the same padding.

      // This is the amount of actual padding to actually add the convTranspose
      // input, tsummed over the start and end of the dimension
      int64_t total_padding = strides.at(dim) * (inputShape[dim + 2] - 1) +
                              outputPadding.at(dim) +
                              (k_minus_one_times_d + 1) - outShape[dim + 2];

      // The extra padding will differ by 1 in the case of an odd number
      int64_t extra_padding_big   = k_minus_one_times_d - total_padding / 2;
      int64_t extra_padding_small = extra_padding_big;
      if (total_padding % 2 == 1) {
        extra_padding_small--;
      }

      if (padType ==
          AutoPad::SAME_UPPER) { // When total_padding is an odd number and
                                 // pad=SAME_UPPER, the extra
        // padding is added at the end
        lowerPadding.push_back(extra_padding_small);
        upperPadding.push_back((k_minus_one_times_d - (total_padding) / 2));
      } else if (padType == AutoPad::SAME_LOWER || padType == AutoPad::NOTSET) {
        // Defaulting to "SAME_LOWER" when no auto_pad option is set
        lowerPadding.push_back(k_minus_one_times_d -
                               (total_padding - ((total_padding + 1) / 2)));
        upperPadding.push_back(k_minus_one_times_d - (total_padding + 1) / 2);
      } else {
        throw error("`VALID` type for auto_pad cannot be used"
                    " if the `output_shape` is also set");
      }
    } else {
      // The padding should match pads, but corrected for the effect of the
      // convolution, as it is the padding of the conv which would be the
      // inverse of the conv transpose.
      int64_t extra_padding_lower = k_minus_one_times_d - pads.at(dim);
      int64_t extra_padding_upper = k_minus_one_times_d +
                                    outputPadding.at(dim) -
                                    pads.at(nSpatialDims + dim);

      lowerPadding.push_back(extra_padding_lower);
      upperPadding.push_back(extra_padding_upper);
    }
  }

  setParams(lowerPadding, upperPadding);
}

void ConvTransposeOp::setParams(const std::vector<int64_t> &lowerPadding,
                                const std::vector<int64_t> &upperPadding) {
  params.type      = inInfo(getInIndex()).dataType();
  params.batchSize = inShape(getInIndex()).at(0);

  Shape inputShape             = inShape(getInIndex());
  params.numInChannelsPerGroup = inputShape.at(1) / group;

  Shape kernelShape             = inShape(getWeightsInIndex());
  params.numOutChannelsPerGroup = kernelShape.at(1);
  params.numGroups              = group;

  for (int i = 2; i < inputShape.size(); i++) {
    params.inputShape.push_back(inputShape.at(i));
  }

  for (int i = 2; i < kernelShape.size(); i++) {
    params.kernelShape.push_back(kernelShape.at(i));
  }

  const auto nSpatialDims = inputShape.size() - 2;
  std::vector<int64_t> zeroes(nSpatialDims, 0);
  std::vector<int64_t> ones(nSpatialDims, 1);
  std::vector<bool> falses(nSpatialDims, false);

  params.inputTransformation.dilation = strides;
  params.inputTransformation.flip     = falses;

  for (auto pad : lowerPadding) {
    if (pad >= 0) {
      params.inputTransformation.lowerTruncation.push_back(0);
      params.inputTransformation.lowerPadding.push_back(pad);
    } else {
      params.inputTransformation.lowerTruncation.push_back(-pad);
      params.inputTransformation.lowerPadding.push_back(0);
    }
  }

  for (auto pad : upperPadding) {
    if (pad >= 0) {
      params.inputTransformation.upperTruncation.push_back(0);
      params.inputTransformation.upperPadding.push_back(pad);
    } else {
      params.inputTransformation.upperTruncation.push_back(-pad);
      params.inputTransformation.upperPadding.push_back(0);
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

bool ConvTransposeOp::isPow2ScaledConvTranspose() const {
  return opInputsAreValidPow2ScaledInputs(input.get(), getLog2ScaleInIndex());
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16,
                                    DataType::FLOAT,
                                    DataType::FLOAT8_143,
                                    DataType::FLOAT8_152};

static OpDefinition
    convtranspose_OpDef({OpDefinition::Inputs({
                             {"X", T},
                             {"W", T},
                             {"B", {DataType::FLOAT16, DataType::FLOAT}},
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
