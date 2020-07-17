// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <string>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/matmul.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/util.hpp>

namespace popart {

MatMulBaseOp::MatMulBaseOp(
    const OperatorIdentifier &_opid,
    const Op::Settings &settings_,
    const Phase phase_,
    const nonstd::optional<float> availableMemoryProportion_,
    const SerialiseSettings &serialization_,
    const OptionalDataType outputType_,
    const MatMulPartialsType partialsType_,
    const bool enableFullyConnectedPass_)
    : Op(_opid, settings_), phase(phase_),
      enableFullyConnectedPass(enableFullyConnectedPass_),
      availableMemoryProportion(availableMemoryProportion_),
      serialization(serialization_), outputType(outputType_),
      partialsType(partialsType_) {}

bool MatMulBaseOp::useFullyConnectedPass() const {
  return getIr().getSessionOptions().enableFullyConnectedPass &&
         enableFullyConnectedPass;
}

void MatMulBaseOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("available_memory_proportion",
                     getAvailableMemoryProportion());
  os.appendAttribute("fully_connected_pass",
                     useFullyConnectedPass() ? static_cast<int64_t>(phase)
                                             : -1);
  os.appendAttribute("partialsType", toString(partialsType));
}

void MatMulBaseOp::appendMore(OpSerialiserBase &os) const {
  Op::appendMore(os);
  os.appendAttribute("serialization_mode",
                     static_cast<int64_t>(serialization.mode));
  os.appendAttribute("serialization_factor",
                     static_cast<int64_t>(serialization.factor));
  os.appendAttribute("partialsType", toString(partialsType));
}

MatMulBaseGradOp::MatMulBaseGradOp(const OperatorIdentifier &_opid,
                                   const MatMulOp &fwdOp,
                                   Phase phaseArg)
    : MatMulBaseOp(_opid,
                   fwdOp.getSettings(),
                   phaseArg,
                   fwdOp.getAvailableMemoryProportion(),
                   fwdOp.getSerialiseSettings(),
                   fwdOp.getOutputType(),
                   fwdOp.getPartialsType(),
                   fwdOp.useFullyConnectedPass()),
      fwdOpOutputGrad(fwdOp.outInfo(0)), fwdOpLhsInfo(fwdOp.lhsIn()->info),
      fwdOpRhsInfo(fwdOp.rhsIn()->info), cloneOfCreator(fwdOp.clone()) {}

const MatMulOp *MatMulBaseGradOp::getCloneOfCreator() const {
  return dynamic_cast<const MatMulOp *>(cloneOfCreator.get());
}

MatMulOp::MatMulOp(const OperatorIdentifier &_opid,
                   const Op::Settings &settings_,
                   const nonstd::optional<float> availableMemoryProportion_,
                   const SerialiseSettings &serialization_,
                   const OptionalDataType outputType_,
                   const MatMulPartialsType partialsType_)
    : MatMulBaseOp(_opid,
                   settings_,
                   Phase::Fwd,
                   availableMemoryProportion_,
                   serialization_,
                   outputType_,
                   partialsType_) {}

std::unique_ptr<Op> MatMulOp::clone() const {
  return std::make_unique<MatMulOp>(*this);
}

std::vector<std::unique_ptr<Op>> MatMulOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<MatMulLhsGradOp>(*this));
  upops.emplace_back(std::make_unique<MatMulRhsGradOp>(*this));
  return upops;
}

const Tensor *MatMulOp::lhsIn() const { return inTensor(getLhsInIndex()); }

const Tensor *MatMulOp::rhsIn() const { return inTensor(getRhsInIndex()); }

const Tensor *MatMulOp::out() const { return outTensor(getOutIndex()); }

void MatMulOp::verifyInputShapes(const Shape &lhs, const Shape &rhs) const {
  if (lhs.empty()) {
    throw error("{} doesn't support scalar tensor {} as the lhs input",
                debugName(),
                lhsIn()->str());
  }

  if (rhs.empty()) {
    throw error("{} doesn't support scalar tensor {} as the rhs input",
                debugName(),
                rhsIn()->str());
  }
}

Shape MatMulOp::npMatMulOut(Shape lhs, Shape rhs) {
  verifyInputShapes(lhs, rhs);

  auto originalLhsShape = lhs;
  auto originalRhsShape = rhs;

  const bool lhs_prepend = lhs.size() == 1;
  const bool rhs_append  = rhs.size() == 1;

  // If the first argument is 1-D, it is promoted to a matrix by prepending a 1
  // to its dimensions.
  if (lhs_prepend) {
    lhs.insert(lhs.begin(), 1);
  }

  // If the second argument is 1-D, it is promoted to a matrix by appending a 1
  // to its dimensions
  if (rhs_append) {
    rhs.push_back(1);
  }

  // Add a 1 group dim
  bool lhsGroupDimAppend = false;
  if (lhs.size() == 2) {
    lhs.insert(lhs.begin(), 1);
    lhsGroupDimAppend = true;
  }

  // Add a 1 group dim
  bool rhsGroupDimAppend = false;
  if (rhs.size() == 2) {
    rhs.insert(rhs.begin(), 1);
    rhsGroupDimAppend = true;
  }

  // Save the expanded input shapes - minium of 3D
  lhsShape = lhs;
  rhsShape = rhs;

  Shape result =
      prettyNpOut({lhs.begin(), lhs.end() - 2}, {rhs.begin(), rhs.end() - 2});

  // Save the expanded output shape - minium of 3D
  outShape = result;
  outShape.push_back(lhs[lhs.size() - 2]);
  outShape.push_back(rhs[rhs.size() - 1]);

  // After matrix multiplication the prepended 1 is removed.
  // We implement this by not adding it.
  if (!lhs_prepend) {
    result.push_back(lhs[lhs.size() - 2]);
  }

  // After matrix multiplication the appended 1 is removed.
  // We implement this by not adding it.
  if (!rhs_append) {
    result.push_back(rhs[rhs.size() - 1]);
  }

  // Squeeze off any prepended 1's if both
  // lhs & rhs had prepended a group dimension
  if (lhsGroupDimAppend && rhsGroupDimAppend && result[0] == 1) {
    result.erase(result.begin());
  }

  // Special case of 2 1d inputs
  if (originalLhsShape.size() == 1 && originalRhsShape.size() == 1 &&
      result.size() == 1 && result[0] == 1)
    result.erase(result.begin());

  if (lhs[lhs.size() - 1] != rhs[rhs.size() - 2]) {

    // Remove the group dimension if added to return to the user defined
    // shape
    if (lhsGroupDimAppend)
      lhs.erase(lhs.begin());

    if (rhsGroupDimAppend)
      rhs.erase(rhs.begin());

    throw error("{} contracting dimensions unequal: lhs '{}' {}, rhs '{}' {}",
                debugName(),
                lhsIn()->str(),
                lhs,
                rhsIn()->str(),
                rhs);
  }

  return result;
}

void MatMulOp::setup() {

  if (phase == Phase::Fwd) {
    if (getSerialiseSettings().mode !=
        MatMulBaseOp::SerialiseSettings::Mode::None) {

      // assuming
      // lhs = [group_dims, input_channels, reduce_dim]
      // rhs = [group_dims, reduce_dim, output_channels]

      if (getSerialiseSettings().mode ==
          MatMulBaseOp::SerialiseSettings::Mode::InputChannels) {

        // Get the input channels of the left hand size
        auto inputChannelsDim =
            lhsIn()->info.shape()[lhsIn()->info.shape().size() - 2];

        if (inputChannelsDim % getSerialiseSettings().factor != 0) {
          throw error("Invalid serialisation factor {} for input channels dim "
                      "{}. input_channels dim should be a multple of the "
                      "serialisation factor ",
                      getSerialiseSettings().factor,
                      inputChannelsDim);
        }
      } else if (getSerialiseSettings().mode ==
                 MatMulBaseOp::SerialiseSettings::Mode::ReducingDim) {
        // Get the reducing dim of the left hand tensor
        auto reducingChannelsDim =
            lhsIn()->info.shape()[lhsIn()->info.shape().size() - 1];

        if (reducingChannelsDim % getSerialiseSettings().factor != 0) {
          throw error("Invalid serialisation factor {} for reducing dimension "
                      "{}. reducing_dim dim should be a multple of the "
                      "serialisation factor ",
                      getSerialiseSettings().factor,
                      reducingChannelsDim);
        }
      } else {

        // Get the output channels of the right hand size
        auto outputChannelsDim =
            rhsIn()->info.shape()[rhsIn()->info.shape().size() - 1];

        logging::op::info("{}", rhsIn()->info.shape());
        if (outputChannelsDim % getSerialiseSettings().factor != 0) {
          throw error("Invalid serialisation factor {} for output channels dim "
                      "{}. output_channels dim should be a multple of the "
                      "serialisation factor ",
                      getSerialiseSettings().factor,
                      outputChannelsDim);
        }
      }
    }
  }

  auto type = lhsIn()->info.dataType();
  if (outputType)
    type = *outputType;

  // Define the shape of the output tensor
  outInfo(0) = {type,
                npMatMulOut(lhsIn()->info.shape(), rhsIn()->info.shape())};
}

MatMulLhsGradOp::MatMulLhsGradOp(const MatMulOp &fwdOp)
    : MatMulBaseGradOp(Onnx::GradOperators::MatMulLhsGrad,
                       fwdOp,
                       Phase::BwdLHS) {}

void MatMulLhsGradOp::setup() { outInfo(0) = fwdOpLhsInfo; }

std::unique_ptr<Op> MatMulLhsGradOp::clone() const {
  return std::make_unique<MatMulLhsGradOp>(*this);
}

const std::vector<GradInOutMapper> &MatMulLhsGradOp::gradInputInfo() const {
  // The gradient of the fwd-op is input at index 0.
  // The index at which the rhs tensor is the input to the grad-op
  // is the same as the index at which it the input to the fwd-op
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), MatMulOp::getOutIndex(), GradOpInType::GradOut},
      {getRhsInIndex(), MatMulOp::getRhsInIndex(), GradOpInType::In}};
  return inInfo;
}

const std::map<int, int> &MatMulLhsGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {
      {getOutIndex(), MatMulOp::getLhsInIndex()}};
  return outInfo;
}

Shape MatMulLhsGradOp::getGradInputShape() const {
  return fwdOpOutputGrad.shape();
}

Shape MatMulLhsGradOp::getRhsInputShape() const { return fwdOpRhsInfo.shape(); }

Shape MatMulLhsGradOp::getOutputShape() const { return fwdOpLhsInfo.shape(); }

MatMulRhsGradOp::MatMulRhsGradOp(const MatMulOp &fwdOp)
    : MatMulBaseGradOp(Onnx::GradOperators::MatMulRhsGrad,
                       fwdOp,
                       Phase::BwdRHS) {}

std::unique_ptr<Op> MatMulRhsGradOp::clone() const {
  return std::make_unique<MatMulRhsGradOp>(*this);
}

void MatMulRhsGradOp::setup() { outInfo(0) = fwdOpRhsInfo; }

const std::vector<GradInOutMapper> &MatMulRhsGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), MatMulOp::getOutIndex(), GradOpInType::GradOut},
      {getLhsInIndex(), MatMulOp::getLhsInIndex(), GradOpInType::In}};
  return inInfo;
}

const std::map<int, int> &MatMulRhsGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 1
  static const std::map<int, int> outInfo = {
      {getOutIndex(), MatMulOp::getRhsInIndex()}};
  return outInfo;
}

Shape MatMulRhsGradOp::getGradInputShape() const {
  return fwdOpOutputGrad.shape();
}

Shape MatMulRhsGradOp::getLhsInputShape() const { return fwdOpLhsInfo.shape(); }

Shape MatMulRhsGradOp::getOutputShape() const { return fwdOpRhsInfo.shape(); }

std::string toString(const MatMulPartialsType &pt) {
  switch (pt) {
  case MatMulPartialsType::HALF:
    return "MatMulPartialsType::HALF";
  case MatMulPartialsType::FLOAT:
    return "MatMulPartialsType::FLOAT";
  default:
    throw error("Bad MatMulPartialsType '{}'", static_cast<int>(pt));
  }
}

std::ostream &operator<<(std::ostream &os, const MatMulPartialsType &pt) {
  os << toString(pt);
  return os;
}

namespace {

// Accepts the strings "half", "float" in any kind of letter case.
MatMulPartialsType fromString(const std::string &user_pt) {
  std::string lowered_pt;
  lowered_pt.resize(user_pt.length());

  std::transform(user_pt.begin(), user_pt.end(), lowered_pt.begin(), ::tolower);

  if (lowered_pt == "half") {
    return MatMulPartialsType::HALF;
  } else if (lowered_pt == "float") {
    return MatMulPartialsType::FLOAT;
  } else {
    const auto err_str_tmpl =
        "Unable to get option 'partialsTypeMatMul' from "
        "string '{}'. Possible values are 'float' and 'half' in any letter "
        "case.";
    throw error(err_str_tmpl, user_pt);
  }
}

static OpDefinition::DataTypes T = {DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition
    matmulOpDef({OpDefinition::Inputs({
                     {"A", T},
                     {"B", T},
                 }),
                 OpDefinition::Outputs({{"Y", T}}),
                 OpDefinition::Attributes({
                     {sSerializeMatMulModeAttribute, {"*"}},
                     {sSerializeMatMulFactorAttribute, {"*"}},
                     {sSerializeMatMulPrecisionAttribute, {"*"}},
                 })});

static OpCreator<MatMulOp> matMulOpCreator(
    OpDefinitions({{Onnx::Operators::MatMul_1, matmulOpDef},
                   {Onnx::Operators::MatMul_9, matmulOpDef}}),
    [](const OpCreatorInfo &info) {
      // try set the availMemAttribute from an attribute

      nonstd::optional<float> availableMemoryProportion;
      nonstd::optional<int64_t> serialize;

      MatMulBaseOp::SerialiseSettings serialisation;

      OptionalDataType outputType;

      auto partialsType = MatMulPartialsType::FLOAT;

      if (info.attributes.hasAttribute(sSerializeMatMulModeAttribute)) {

        std::string mode = info.attributes.getAttribute<Attributes::String>(
            sSerializeMatMulModeAttribute, sSerializeMatMulMode_None);
        if (mode == sSerializeMatMulMode_InputChannels) {
          serialisation.mode =
              MatMulBaseOp::SerialiseSettings::Mode::InputChannels;
        } else if (mode == sSerializeMatMulMode_ReducingDim) {
          serialisation.mode =
              MatMulBaseOp::SerialiseSettings::Mode::ReducingDim;
        } else if (mode == sSerializeMatMulMode_OutputChannels) {
          serialisation.mode =
              MatMulBaseOp::SerialiseSettings::Mode::OutputChannels;
        } else if (mode == sSerializeMatMulMode_None) {
          serialisation.mode = MatMulBaseOp::SerialiseSettings::Mode::None;
        } else {
          throw error("Unsupport matmul serialisation mode {}", mode);
        }

        serialisation.factor = info.attributes.getAttribute<Attributes::Int>(
            sSerializeMatMulFactorAttribute);

        serialisation.keep_precision =
            info.attributes.getAttribute<Attributes::Int>(
                sSerializeMatMulPrecisionAttribute);
      }

      if (info.attributes.hasAttribute(sAvailMemAttribute)) {
        availableMemoryProportion =
            info.attributes.getAttribute<Attributes::Float>(sAvailMemAttribute);
      }

      if (info.attributes.hasAttribute(sOutputTypeAttribute)) {
        auto dtype_str = info.attributes.getAttribute<Attributes::String>(
            sOutputTypeAttribute);
        outputType = {dataTypeFromString(dtype_str)};
      }

      // same as in ConvOp's create
      // try set the partials from an attribute
      if (info.attributes.hasAttribute(sPartialsTypeAttribute)) {
        std::string partialsTypeAttr =
            info.attributes.getAttribute<Attributes::String>(
                sPartialsTypeAttribute);
        partialsType = fromString(partialsTypeAttr);
      }
      // otherwise see if partials type was set in the session options
      else {
        const auto &opts = info.settings.getIr().getSessionOptions();
        const std::string globalPartialsTypeStr = opts.partialsTypeMatMuls;
        if (!globalPartialsTypeStr.empty()) {
          partialsType = fromString(globalPartialsTypeStr);
        }
      }

      return std::unique_ptr<Op>(new MatMulOp(info.opid,
                                              info.settings,
                                              availableMemoryProportion,
                                              serialisation,
                                              outputType,
                                              partialsType));
    },
    true);
} // namespace

} // namespace popart
