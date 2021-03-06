// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "popart/tensorinfo.hpp"
#include <memory>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/lstm.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>

namespace popart {

ActivationFunction fromString(const std::string &s) {
  if (s == "Sigmoid") {
    return ActivationFunction::Sigmoid;
  } else if (s == "Relu") {
    return ActivationFunction::Relu;
  } else if (s == "Tanh") {
    return ActivationFunction::Tanh;
  } else if (s == "Gelu") {
    return ActivationFunction::Gelu;
  } else if (s == "Swish") {
    return ActivationFunction::Swish;
  } else if (s == "Softmax") {
    return ActivationFunction::Softmax;
  } else if (s == "SoftmaxStable") {
    return ActivationFunction::SoftmaxStable;
  } else if (s == "SoftmaxScaled") {
    return ActivationFunction::SoftmaxScaled;
  } else {
    return ActivationFunction::Invalid;
  }
}

std::ostream &operator<<(std::ostream &os, const ActivationFunction &af) {
  switch (af) {
  case ActivationFunction::Sigmoid:
    os << "Sigmoid";
    break;
  case ActivationFunction::Relu:
    os << "Relu";
    break;
  case ActivationFunction::Tanh:
    os << "Tanh";
    break;
  case ActivationFunction::Gelu:
    os << "Gelu";
    break;
  case ActivationFunction::Swish:
    os << "Swish";
    break;
  case ActivationFunction::Softmax:
    os << "Softmax";
    break;
  case ActivationFunction::SoftmaxStable:
    os << "SoftmaxStable";
    break;
  case ActivationFunction::SoftmaxScaled:
    os << "SoftmaxScaled";
    break;
  case ActivationFunction::Invalid:
  default:
    os << "Invalid";
    break;
  }
  return os;
}

LSTMOp::LSTMOp(const OperatorIdentifier &_opid,
               nonstd::optional<int64_t> hidden_size,
               ActivationFunction activation,
               ActivationFunction recurrent_activation,
               const Op::Settings &settings_)
    : Op(_opid, settings_), hidden_size_attribute(hidden_size),
      activation(activation), recurrent_activation(recurrent_activation) {
  // TODO : Use the output_sequence attribute in version 1
}

std::unique_ptr<Op> LSTMOp::clone() const {
  return std::make_unique<LSTMOp>(*this);
}

std::vector<std::unique_ptr<Op>> LSTMOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<LSTMGradOp>(*this));
  return upops;
}

bool LSTMOp::isTraining() const {
  return getGraph().getIr().getExecutionMode() == Ir::ExecutionMode::Training;
}

void LSTMOp::trySetOutInfo(OutIndex index, const TensorInfo &info) {
  if (output->hasIndex(index)) {
    outInfo(index) = info;
  }
}

namespace {

int64_t lstmGetNumIntermediates(const SessionOptions &opts) {
  auto found = opts.lstmOptions.find("recomputationMode");
  if (found == opts.lstmOptions.end() || found->second == "none") {
    return 6;
  } else if (found->second == "cellAndTanh") {
    return 4;
  } else if (found->second == "full") {
    // Currently 'full' causes a poplibs error 'Unhandled recomputation type'.
    // I would think that the intermediates output is not required with full
    // recomputation.
    return 0;
  } else {
    throw error("Unrecognised lstm recomputation mode '{}'", found->second);
  }
}

} // namespace

int64_t LSTMOp::getNumIntermediates() const {
  return lstmGetNumIntermediates(getIr().getSessionOptions());
}

void LSTMOp::setup() {
  if (input->hasIndex(getPeepholeInIndex())) {
    throw error("Popart does not support peephole connections");
  }
  int64_t hidden_size = 0;
  if (!hidden_size_attribute) {
    hidden_size = inShape(getRecurrenceInIndex())[2];
  } else if (*hidden_size_attribute != inShape(getRecurrenceInIndex())[2]) {
    throw error("LSTMOp hidden_size attribute, {}, does not match calculated "
                "hidden size, {}.",
                *hidden_size_attribute,
                inShape(getRecurrenceInIndex())[2]);
  } else {
    hidden_size = *hidden_size_attribute;
  }

  auto max_seq_length = getMaxSeqLength();
  auto num_directions = getNumDirections();
  auto batch_size     = getBatchSize();
  auto data_type      = inInfo(getInputInIndex()).data_type();
  auto input_size     = getInputSize();

  if (input->hasIndex(getSequenceLensInIndex())) {
    if (inInfo(getSequenceLensInIndex()).rank() != 1) {
      throw error("Invalid rank for sequence length tensor : {} != 1",
                  inInfo(getSequenceLensInIndex()).rank());

    } else if (inInfo(getSequenceLensInIndex()).shape().size() > 0) {
      if (inInfo(getSequenceLensInIndex()).shape()[0] != batch_size) {
        throw error("Incorect sequence len shape : {} != [{}]",
                    inInfo(getSequenceLensInIndex()).shape(),
                    batch_size);
      }
    }
  }

  Shape y_shape{max_seq_length, num_directions, batch_size, hidden_size};

  trySetOutInfo(getOutputOutIndex(), {data_type, y_shape});

  Shape yhc_shape{num_directions, batch_size, hidden_size};
  trySetOutInfo(getHiddenStateOutIndex(), {data_type, yhc_shape});
  trySetOutInfo(getCellStateOutIndex(), {data_type, yhc_shape});

  createPassThroughOutput("initstateoutput",
                          getInitStateOutputPassThroughIndex(),
                          {data_type, Shape{batch_size, hidden_size}});
  createPassThroughOutput("initstatecellstate",
                          getInitStateCellStatePassThroughIndex(),
                          {data_type, Shape{batch_size, hidden_size}});
  createPassThroughOutput(
      "intermediates",
      getIntermediatesPassThroughIndex(),
      {data_type,
       Shape{max_seq_length, getNumIntermediates(), batch_size, hidden_size}});
  createPassThroughOutput("inputweights",
                          getInputWeightsPassThroughIndex(),
                          {data_type, Shape{4, input_size, hidden_size}});
  createPassThroughOutput("outputweights",
                          getOutputWeightsPassThroughIndex(),
                          {data_type, Shape{4, hidden_size, hidden_size}});
  createPassThroughOutput("biases",
                          getBiasesPassThroughIndex(),
                          {data_type, Shape{4, hidden_size}});
  createPassThroughOutput(
      "input",
      getInputPassThroughIndex(),
      {data_type, Shape{max_seq_length, batch_size, input_size}});
  createPassThroughOutput(
      "output",
      getOutputPassThroughIndex(),
      {data_type, Shape{max_seq_length, batch_size, hidden_size}});
}

void LSTMOp::createPassThroughOutput(const TensorId &new_id,
                                     OutIndex pass_through_index,
                                     const TensorInfo &out_info) {
  auto tensor_id =
      (getScope() / logging::format("lstm({})_{}", id, new_id)).str();
  if (hasOutput(pass_through_index)) {
    disconnectOutTensor(outTensor(pass_through_index));
  }
  if (getGraph().getTensors().contains(tensor_id)) {
    connectOutTensor(pass_through_index, tensor_id);
  } else {
    createAndConnectOutTensor(pass_through_index, tensor_id);
  }
  outInfo(pass_through_index) = out_info;
}

unsigned LSTMOp::getNumChannels() const { return 1; }

int64_t LSTMOp::getMaxSeqLength() const {
  return inShape(getInputInIndex())[0];
}

int64_t LSTMOp::getBatchSize() const { return inShape(getInputInIndex())[1]; }

int64_t LSTMOp::getInputSize() const { return inShape(getInputInIndex())[2]; }

int64_t LSTMOp::getNumDirections() const { return 1; }

int64_t LSTMOp::getHiddenSize() const {
  return inShape(getRecurrenceInIndex())[2];
}

bool LSTMOp::hasBiasInput() const { return input->hasIndex(getBiasInIndex()); }

bool LSTMOp::hasInitialHInput() const {
  return input->hasIndex(getInitialHInIndex());
}

bool LSTMOp::hasInitialCInput() const {
  return input->hasIndex(getInitialCInIndex());
}

bool LSTMOp::hasSeqLenInput() const {
  return input->hasIndex(getSequenceLensInIndex());
}

bool LSTMOp::hasOutput(OutIndex index) const { return output->hasIndex(index); }

std::set<InIndex> LSTMOp::optionalInputs() const {
  return {getInitialHInIndex(),
          getInitialCInIndex(),
          getSequenceLensInIndex(),
          getPeepholeInIndex()};
}

void LSTMOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("hidden_size", hidden_size_attribute);
  os.appendAttribute("activation", activation);
  os.appendAttribute("recurrent_activation", recurrent_activation);
}

int LSTMOp::getInBatchAxis(InIndex index) const {
  if (index == getInputInIndex() || index == getInitialHInIndex() ||
      index == getInitialCInIndex()) {
    return 1;
  }
  return 0;
}

int LSTMOp::getOutBatchAxis(OutIndex index) const {
  if (index == getOutputOutIndex()) {
    return 2;
  } else if (index == getHiddenStateOutIndex() || getCellStateOutIndex()) {
    return 1;
  }
  return 0;
}

view::Regions LSTMOp::aliases(InIndex in, OutIndex out) const {
  if (in == getInputInIndex() && out == getInputPassThroughIndex()) {
    return {view::Region::getFull(inShape(in))};
  } else {
    return {view::Region::getEmpty(inRank(in))};
  }
}

view::RegMap LSTMOp::fwdRegMap(InIndex in, OutIndex out) const {
  auto emptyRegion = view::Region::getEmpty(outRank(out));
  if (in == getInputInIndex() && out == getInputPassThroughIndex()) {
    return [emptyRegion](const view::Region &r) { return view::Regions(1, r); };
  } else {
    return [emptyRegion](const view::Region &r) {
      return view::Regions(1, emptyRegion);
    };
  }
}

view::RegMap LSTMOp::bwdRegMap(InIndex in, OutIndex out) const {
  auto emptyRegion = view::Region::getEmpty(inRank(in));
  if (in == getInputInIndex() && out == getInputPassThroughIndex()) {
    return [emptyRegion](const view::Region &r) { return view::Regions(1, r); };
  } else {
    return [emptyRegion](const view::Region &r) {
      return view::Regions(1, emptyRegion);
    };
  }
}

LSTMGradOp::LSTMGradOp(const LSTMOp &fwd_op)
    : Op(Onnx::GradOperators::LSTMGrad, fwd_op.getSettings()),
      forward_op(fwd_op) {}

std::unique_ptr<Op> LSTMGradOp::clone() const {
  return std::make_unique<LSTMGradOp>(*this);
}

void LSTMGradOp::setup() {
  outInfo(getInputOutIndex()) = forward_op.inInfo(LSTMOp::getInputInIndex());
  outInfo(getWeightsOutIndex()) =
      forward_op.inInfo(LSTMOp::getWeightsInIndex());
  outInfo(getRecurrenceOutIndex()) =
      forward_op.inInfo(LSTMOp::getRecurrenceInIndex());

  if (forward_op.hasBiasInput()) {
    outInfo(getBiasOutIndex()) = forward_op.inInfo(LSTMOp::getBiasInIndex());
  }
  if (forward_op.hasInitialHInput()) {
    outInfo(getInitialHOutIndex()) =
        forward_op.inInfo(LSTMOp::getInitialHInIndex());
  }
  if (forward_op.hasInitialCInput()) {
    outInfo(getInitialCOutIndex()) =
        forward_op.inInfo(LSTMOp::getInitialCInIndex());
  }
}

bool LSTMGradOp::hasCellStateGradInput() const {
  return input->hasIndex(getCellStateOutputGradInIndex());
}

bool LSTMGradOp::hasHiddenStateGradInput() const {
  return input->hasIndex(getHiddenStateOutputGradInIndex());
}

std::set<InIndex> LSTMGradOp::optionalInputs() const {
  return {getCellStateOutputGradInIndex(),
          getHiddenStateOutputGradInIndex(),
          getSequenceLensInIndex()};
}

const std::vector<GradInOutMapper> &LSTMGradOp::gradInputInfo() const {
  static std::vector<GradInOutMapper> inInfo = {
      {getInitStateOutputInIndex(),
       LSTMOp::getInitStateOutputPassThroughIndex(),
       GradOpInType::Out},
      {getInitStateCellStateInIndex(),
       LSTMOp::getInitStateCellStatePassThroughIndex(),
       GradOpInType::Out},
      {getIntermediatesInIndex(),
       LSTMOp::getIntermediatesPassThroughIndex(),
       GradOpInType::Out},
      {getInputWeightsInIndex(),
       LSTMOp::getInputWeightsPassThroughIndex(),
       GradOpInType::Out},
      {getOutputWeightsInIndex(),
       LSTMOp::getOutputWeightsPassThroughIndex(),
       GradOpInType::Out},
      {getBiasesInIndex(),
       LSTMOp::getBiasesPassThroughIndex(),
       GradOpInType::Out},
      {getInputInIndex(),
       LSTMOp::getInputPassThroughIndex(),
       GradOpInType::Out},
      {getOutputInIndex(),
       LSTMOp::getOutputPassThroughIndex(),
       GradOpInType::Out},
      {getCellStateOutputGradInIndex(),
       LSTMOp::getCellStateOutIndex(),
       GradOpInType::GradOut},
      {getHiddenStateOutputGradInIndex(),
       LSTMOp::getHiddenStateOutIndex(),
       GradOpInType::GradOut},
      {getOutputGradInIndex(),
       LSTMOp::getOutputOutIndex(),
       GradOpInType::GradOut}};
  if (getForwardOp().hasSeqLenInput()) {
    inInfo.push_back({getSequenceLensInIndex(),
                      LSTMOp::getSequenceLensInIndex(),
                      GradOpInType::In});
  }
  return inInfo;
}

const std::map<int, int> &LSTMGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getInputOutIndex(), LSTMOp::getInputInIndex()},
      {getWeightsOutIndex(), LSTMOp::getWeightsInIndex()},
      {getRecurrenceOutIndex(), LSTMOp::getRecurrenceInIndex()},
      {getBiasOutIndex(), LSTMOp::getBiasInIndex()},
      {getInitialHOutIndex(), LSTMOp::getInitialHInIndex()},
      {getInitialCOutIndex(), LSTMOp::getInitialCInIndex()}};

  return outInfo;
}

const LSTMOp &LSTMGradOp::getForwardOp() const { return forward_op; }

PopartLSTMOp::PopartLSTMOp(const OperatorIdentifier &opid_,
                           bool outputFullSequence_,
                           const Op::Settings &settings_)
    : PopartLSTMOp(opid_,
                   outputFullSequence_,
                   ActivationFunction::Tanh,
                   ActivationFunction::Sigmoid,
                   settings_) {}

PopartLSTMOp::PopartLSTMOp(const OperatorIdentifier &opid_,
                           bool outputFullSequence_,
                           ActivationFunction activation,
                           ActivationFunction recurrent_activation,
                           const Op::Settings &settings_)
    : Op(opid_, settings_), outputFullSequence(outputFullSequence_),
      activation(activation), recurrent_activation(recurrent_activation) {}

std::unique_ptr<Op> PopartLSTMOp::clone() const {
  return std::make_unique<PopartLSTMOp>(*this);
}

std::vector<std::unique_ptr<Op>> PopartLSTMOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<PopartLSTMGradOp>(*this));
  return upops;
}

int64_t PopartLSTMOp::getNumIntermediates() const {
  return lstmGetNumIntermediates(getIr().getSessionOptions());
}

void PopartLSTMOp::setup() {
  auto verifyShape = [this](
                         InIndex inIndex, Shape refShape, std::string idStr) {
    auto inputShape = inShape(inIndex);
    if (inputShape != refShape) {
      throw error("Bad {} shape {}, should be {}", idStr, inIndex, refShape);
    }
  };

  // Verify the input shapes.
  verifyShape(getInputInIndex(),
              {getMaxSeqLength(), getBatchSize(), getInputSize()},
              "input");
  verifyShape(getWeightsInIndex(),
              {4, getInputSize() + getHiddenSize(), getHiddenSize()},
              "weights");
  if (hasBiasesInput()) {
    verifyShape(getBiasesInIndex(), {4, getHiddenSize()}, "bias");
  }
  if (input->hasIndex(getInitialStateInIndex())) {
    verifyShape(getInitialStateInIndex(),
                {2, getBatchSize(), getHiddenSize()},
                "initialState");
  }

  auto dtype = inInfo(getInputInIndex()).dataType();

  Shape outputShape;
  if (outputFullSequence) {
    outputShape = {getMaxSeqLength(), getBatchSize(), getHiddenSize()};
  } else {
    outputShape = {getBatchSize(), getHiddenSize()};
  }

  outInfo(getOutputOutIndex())    = {dtype, outputShape};
  outInfo(getCellStateOutIndex()) = {dtype, {getBatchSize(), getHiddenSize()}};

  if (getIr().isTraining()) {
    if (output->hasIndex(getIntermediatesOutIndex())) {
      disconnectOutTensor(outTensor(getIntermediatesOutIndex()));
    }
    TensorId intermediates =
        (getScope() / logging::format("{}_intermediates", id)).str();
    if (getGraph().getTensors().contains(intermediates)) {
      connectOutTensor(getIntermediatesOutIndex(), intermediates);
    } else {
      createAndConnectOutTensor(getIntermediatesOutIndex(), intermediates);
    }
    outInfo(getIntermediatesOutIndex()) = {dtype,
                                           {getMaxSeqLength(),
                                            getNumIntermediates(),
                                            getBatchSize(),
                                            getHiddenSize()}};
  }
}

bool PopartLSTMOp::hasBiasesInput() const {
  return input->hasIndex(getBiasesInIndex());
}

std::set<InIndex> PopartLSTMOp::optionalInputs() const {
  return {
      getBiasesInIndex(), getInitialStateInIndex(), getSequenceLensInIndex()};
}

int64_t PopartLSTMOp::getMaxSeqLength() const {
  return inShape(getInputInIndex()).at(0);
}

int64_t PopartLSTMOp::getBatchSize() const {
  return inShape(getInputInIndex()).at(1);
}

int64_t PopartLSTMOp::getInputSize() const {
  return inShape(getInputInIndex()).at(2);
}

int64_t PopartLSTMOp::getHiddenSize() const {
  return inShape(getWeightsInIndex()).at(2);
}

int PopartLSTMOp::getInBatchAxis(InIndex index) const {
  if (index == getInputInIndex() || index == getInitialStateInIndex()) {
    return 1;
  }
  return 0;
}

bool PopartLSTMOp::hasSeqLenInput() const {
  return input->hasIndex(getSequenceLensInIndex());
}

int PopartLSTMOp::getOutBatchAxis(OutIndex index) const {
  if (index == getOutputOutIndex() && outputFullSequence) {
    return 1;
  } else if (index == getIntermediatesOutIndex()) {
    return 2;
  }
  return 0;
}

PopartLSTMGradOp::PopartLSTMGradOp(const PopartLSTMOp &fwd_op)
    : Op(Onnx::GradOperators::PopartLSTMGrad, fwd_op.getSettings()),
      outputFullSequence(fwd_op.outputFullSequence),
      forwardCellStateGradId(
          getGradId(fwd_op.outId(PopartLSTMOp::getCellStateOutIndex()))),
      activation(fwd_op.getActivation()),
      recurrent_activation(fwd_op.getRecurrentActivation()) {}

std::unique_ptr<Op> PopartLSTMGradOp::clone() const {
  return std::make_unique<PopartLSTMGradOp>(*this);
}

void PopartLSTMGradOp::setup() {
  if (input->hasIndex(getBiasesInIndex())) {
    outInfo(getBiasesOutIndex()) = inInfo(getBiasesInIndex());
  }
  if (input->hasIndex(getInitialStateInIndex())) {
    outInfo(getInitialStateOutIndex()) = inInfo(getInitialStateInIndex());
  }

  outInfo(getInputOutIndex())   = inInfo(getInputInIndex());
  outInfo(getWeightsOutIndex()) = inInfo(getWeightsInIndex());
}

std::set<InIndex> PopartLSTMGradOp::optionalInputs() const {
  return {getBiasesInIndex(),
          getInitialStateInIndex(),
          getFwdCellStateGradInIndex(),
          getSequenceLensInIndex()};
}

int64_t PopartLSTMGradOp::getInputSize() const {
  return inShape(getInputInIndex()).at(2);
}

int64_t PopartLSTMGradOp::getMaxSeqLength() const {
  return inShape(getInputInIndex()).at(0);
}

int64_t PopartLSTMGradOp::getBatchSize() const {
  return inShape(getInputInIndex()).at(1);
}

int64_t PopartLSTMGradOp::getHiddenSize() const {
  return inShape(getWeightsInIndex()).at(2);
}

const std::vector<GradInOutMapper> &PopartLSTMGradOp::gradInputInfo() const {
  static std::vector<GradInOutMapper> inInfo = {
      {getInitialStateInIndex(),
       PopartLSTMOp::getInitialStateInIndex(),
       GradOpInType::In},
      {getIntermediatesInIndex(),
       PopartLSTMOp::getIntermediatesOutIndex(),
       GradOpInType::Out},
      {getWeightsInIndex(),
       PopartLSTMOp::getWeightsInIndex(),
       GradOpInType::In},
      {getBiasesInIndex(), PopartLSTMOp::getBiasesInIndex(), GradOpInType::In},
      {getInputInIndex(), PopartLSTMOp::getInputInIndex(), GradOpInType::In},
      {getFwdOutputInIndex(),
       PopartLSTMOp::getOutputOutIndex(),
       GradOpInType::Out},
      {getFwdOutputGradInIndex(),
       PopartLSTMOp::getOutputOutIndex(),
       GradOpInType::GradOut},
      {getFwdCellStateGradInIndex(),
       PopartLSTMOp::getCellStateOutIndex(),
       GradOpInType::GradOut}};

  if (input->hasIndex(getSequenceLensInIndex())) {
    inInfo.push_back({getSequenceLensInIndex(),
                      PopartLSTMOp::getSequenceLensInIndex(),
                      GradOpInType::In});
  }
  return inInfo;
}

const std::map<int, int> &PopartLSTMGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getInputOutIndex(), PopartLSTMOp::getInputInIndex()},
      {getWeightsOutIndex(), PopartLSTMOp::getWeightsInIndex()},
      {getBiasesOutIndex(), PopartLSTMOp::getBiasesInIndex()},
      {getInitialStateOutIndex(), PopartLSTMOp::getInitialStateInIndex()}};

  return outInfo;
}

namespace {

static OpDefinition::DataTypes T  = {DataType::FLOAT16, DataType::FLOAT};
static OpDefinition::DataTypes T1 = {DataType::INT32};

static OpDefinition
    lstmOpDef({OpDefinition::Inputs({
                   {"X", T},
                   {"W", T},
                   {"R", T},
                   {"B", T},
                   {"sequence_lens", T1},
                   {"initial_h", T},
                   {"initial_c", T},
                   //{"P", T },// peep hole not supported
               }),
               OpDefinition::Outputs({{"Y", T}, {"Y_h", T}, {"Y_c", T}}),
               OpDefinition::Attributes({
                   //{"activation_alpha", {"*"}},
                   //{"activation_beta", {"*"}},
                   //{"activations", {"*"}},
                   //{"clip", {"*"}},
                   {"direction", {"forward"}},
                   {"hidden_size", {"*"}},
                   {"input_forget", {"0"}},
               })});

namespace {

// returns activation and recurrent activation.
std::pair<ActivationFunction, ActivationFunction>
readActivations(const Attributes &attributes) {
  ActivationFunction activation           = ActivationFunction::Tanh;
  ActivationFunction recurrent_activation = ActivationFunction::Sigmoid;

  if (attributes.hasAttribute("activations")) {
    std::vector<std::string> afs;
    afs = attributes.getAttribute<std::vector<std::string>>("activations");
    logging::debug("LSTM activations are {}", afs);
    if (afs.size() >= 1) {
      recurrent_activation = fromString(afs.at(0));
      if (recurrent_activation == ActivationFunction::Invalid) {
        throw error("Activation type '{}' is not supported", afs.at(0));
      }
    }
    if (afs.size() >= 2) {
      activation = fromString(afs.at(1));
      if (activation == ActivationFunction::Invalid) {
        throw error("Activation type '{}' is not supported", afs.at(1));
      }
    }
    if (afs.size() >= 3 && afs[1] != afs[2]) {
      throw error(
          "LSTM activations must be the same for output and forget gates.");
    }
    if (afs.size() > 3) {
      throw error("Too many activations ({}) specified. Number of activations "
                  "should be 1 or 3.",
                  afs.size());
    }
  }

  return {activation, recurrent_activation};
}

} // namespace

static OpCreator<LSTMOp> lstmOpCreator(
    OpDefinitions({{Onnx::Operators::LSTM_1, lstmOpDef},
                   {Onnx::Operators::LSTM_7, lstmOpDef}}),
    [](const OpCreatorInfo &info) {
      if (info.attributes.getAttribute<Attributes::String>(
              "direction", "forward") != "forward") {
        throw error("LSTMOp attribute `direction' must be unset or `forward'");
      }

      const auto activations          = readActivations(info.attributes);
      const auto activation           = std::get<0>(activations);
      const auto recurrent_activation = std::get<1>(activations);

      // Activation alpha and beta are not supported.
      // In theory these should never be set, as they should only be present
      // when using certain activation functions which we do not support.
      if (info.attributes.hasAttribute("activation_alpha")) {
        throw error("LSTMOp attribute `activation_alpha' is not supported");
      }
      if (info.attributes.hasAttribute("activation_beta")) {
        throw error("LSTMOp attribute `activation_beta' is not supported");
      }

      if (info.attributes.hasAttribute("clip")) {
        throw error("LSTMOp attribute `clip' is not supported");
      }

      if (info.attributes.getAttribute<Attributes::Int>("input_forget", 0) !=
          0) {
        throw error("LSTMOp attribute `input_forget' must be set to 0");
      }

      // cannot check hidden_size till inputs are connected
      nonstd::optional<int64_t> hidden_size;
      if (info.attributes.hasAttribute("hidden_size")) {
        hidden_size =
            info.attributes.getAttribute<Attributes::Int>("hidden_size");
      }

      return std::unique_ptr<Op>(new LSTMOp(info.opid,
                                            hidden_size,
                                            activation,
                                            recurrent_activation,
                                            info.settings));
    },
    true);

static OpDefinition popartLstmOpDef(
    {OpDefinition::Inputs({{"X", T},
                           {"Weights", T},
                           {"Bias", T},
                           {"InitiState", T},
                           {"SeqLengths", T1}}),
     OpDefinition::Outputs({{"Output", T}, {"CellState", T}}),
     OpDefinition::Attributes({{"output_full_sequence", {"*"}}})});

static OpCreator<PopartLSTMOp> popartLSTMOpCreator(
    OpDefinitions({{Onnx::CustomOperators::LSTM_1, popartLstmOpDef}}),
    [](const OpCreatorInfo &info) {
      bool outputFullSequence = info.attributes.getAttribute<Attributes::Int>(
                                    "output_full_sequence", 1) != 0;

      return std::unique_ptr<Op>(
          new PopartLSTMOp(info.opid, outputFullSequence, info.settings));
    },
    true);

} // namespace

} // namespace popart
