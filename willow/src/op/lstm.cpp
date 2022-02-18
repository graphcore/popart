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

LSTMOp::LSTMOp(const OperatorIdentifier &_opid,
               nonstd::optional<int64_t> hidden_size,
               ActivationFunction activation,
               ActivationFunction recurrent_activation,
               const Op::Settings &settings_,
               const nonstd::optional<float> available_memory_proportion_)
    : BaseOnnxRNNOp(_opid, hidden_size, settings_), activation(activation),
      recurrent_activation(recurrent_activation),
      available_memory_proportion(available_memory_proportion_) {
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

  checkHiddenSize();

  int64_t hidden_size = getHiddenSize();
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
        throw error("Incorrect sequence len shape : {} != [{}]",
                    inInfo(getSequenceLensInIndex()).shape(),
                    batch_size);
      }
    }
  }

  Shape y_shape{max_seq_length, num_directions, batch_size, hidden_size};

  trySetOutInfo(getFullHiddenStateOutIndex(), {data_type, y_shape});

  Shape yhc_shape{num_directions, batch_size, hidden_size};
  trySetOutInfo(getLastHiddenStateOutIndex(), {data_type, yhc_shape});
  trySetOutInfo(getLastCellStateOutIndex(), {data_type, yhc_shape});

  maybeCreatePassThroughOutput("initstateoutput",
                               getInitialHPassThroughIndex(),
                               {data_type, Shape{batch_size, hidden_size}});
  maybeCreatePassThroughOutput("initstatecellstate",
                               getInitialCPassThroughIndex(),
                               {data_type, Shape{batch_size, hidden_size}});
  maybeCreatePassThroughOutput(
      "intermediates",
      getIntermediatesPassThroughIndex(),
      {data_type,
       Shape{max_seq_length, getNumIntermediates(), batch_size, hidden_size}});
  maybeCreatePassThroughOutput("inputweights",
                               getInputWeightsPassThroughIndex(),
                               {data_type, Shape{4, input_size, hidden_size}});
  maybeCreatePassThroughOutput("outputweights",
                               getRecurrenceWeightsPassThroughIndex(),
                               {data_type, Shape{4, hidden_size, hidden_size}});
  maybeCreatePassThroughOutput("biases",
                               getBiasesPassThroughIndex(),
                               {data_type, Shape{4, hidden_size}});
}

void LSTMOp::maybeCreatePassThroughOutput(const TensorId &new_id,
                                          OutIndex pass_through_index,
                                          const TensorInfo &out_info) {
  // If the op is being cloned, or setup is being called a second time, the
  // output may already be connected; we do not need to recreate it.
  if (!hasOutput(pass_through_index)) {
    auto tensor_id =
        (getScope() / logging::format("lstm({})_{}", id, new_id)).str();
    if (getGraph().getTensors().contains(tensor_id)) {
      connectOutTensor(pass_through_index, tensor_id);
    } else {
      createAndConnectOutTensor(pass_through_index, tensor_id);
    }
  }
  outInfo(pass_through_index) = out_info;
}

unsigned LSTMOp::getNumChannels() const { return 1; }

int64_t LSTMOp::getNumDirections() const { return 1; }

nonstd::optional<float> LSTMOp::getAvailableMemoryProportion() const {
  return available_memory_proportion;
}

bool LSTMOp::hasInitialCInput() const {
  return input->hasIndex(getInitialCInIndex());
}

bool LSTMOp::hasOutput(OutIndex index) const { return output->hasIndex(index); }

std::set<InIndex> LSTMOp::optionalInputs() const {
  auto optionals = BaseOnnxRNNOp::optionalInputs();
  optionals.insert(getInitialCInIndex());
  optionals.insert(getPeepholeInIndex());
  return optionals;
}

void LSTMOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  BaseOnnxRNNOp::appendOutlineAttributes(os);
  os.appendAttribute("activation", activation);
  os.appendAttribute("recurrent_activation", recurrent_activation);
  os.appendAttribute("available_memory_proportion",
                     available_memory_proportion);
}

int LSTMOp::getInBatchAxis(InIndex index) const {
  if (index == getInputInIndex() || index == getInitialHInIndex() ||
      index == getInitialCInIndex()) {
    return 1;
  }
  return 0;
}

int LSTMOp::getOutBatchAxis(OutIndex index) const {
  if (index == getFullHiddenStateOutIndex()) {
    return 2;
  } else if (index == getLastHiddenStateOutIndex() ||
             index == getLastCellStateOutIndex()) {
    return 1;
  }
  return 0;
}

LSTMGradOp::LSTMGradOp(const LSTMOp &fwd_op)
    : BaseOnnxRNNGradOp(Onnx::GradOperators::LSTMGrad, fwd_op),
      hasInitialCInput(fwd_op.hasInitialCInput()),
      fwd_debug_name(fwd_op.debugName()), activation(fwd_op.getActivation()),
      recurrent_activation(fwd_op.getRecurrentActivation()),
      fwdInitialCInInfo(getInitialCInInfo(fwd_op)) {
  populateInInfo();
}

nonstd::optional<TensorInfo>
LSTMGradOp::getInitialCInInfo(const LSTMOp &fwd_op) {
  if (hasInitialCInput) {
    return fwd_op.inInfo(LSTMOp::getInitialCInIndex());
  }
  return nonstd::nullopt;
}

std::unique_ptr<Op> LSTMGradOp::clone() const {
  return std::make_unique<LSTMGradOp>(*this);
}

void LSTMGradOp::setup() {
  BaseOnnxRNNGradOp::setup();
  if (hasInitialCInput) {
    outInfo(getInitialCOutIndex()) = fwdInitialCInInfo.value();
  }
}

bool LSTMGradOp::hasLastCellStateGradInput() const {
  return input->hasIndex(getLastCellStateGradInIndex());
}

std::set<InIndex> LSTMGradOp::optionalInputs() const {
  return {getLastCellStateGradInIndex(),
          getLastHiddenStateGradInIndex(),
          getSequenceLensInIndex()};
}

void LSTMGradOp::populateInInfo() {
  BaseOnnxRNNGradOp::populateInInfo();
  // initial_h, or 0 if not provided by user
  inInfoMapping.push_back({getInitialHInIndex(),
                           LSTMOp::getInitialHPassThroughIndex(),
                           GradOpInType::Out});
  // W, restructured to be used with poplar implementation
  inInfoMapping.push_back({getInputWeightsInIndex(),
                           LSTMOp::getInputWeightsPassThroughIndex(),
                           GradOpInType::Out});
  // R, restructured to be used with poplar implementation
  inInfoMapping.push_back({getRecurrenceWeightsInIndex(),
                           LSTMOp::getRecurrenceWeightsPassThroughIndex(),
                           GradOpInType::Out});
  // b, restructured to be used with poplar implementation
  inInfoMapping.push_back({getBiasesInIndex(),
                           LSTMOp::getBiasesPassThroughIndex(),
                           GradOpInType::Out});
  // intermediate tensors
  inInfoMapping.push_back({getIntermediatesInIndex(),
                           LSTMOp::getIntermediatesPassThroughIndex(),
                           GradOpInType::Out});
  // initial_c
  inInfoMapping.push_back({getInitialCInIndex(),
                           LSTMOp::getInitialCPassThroughIndex(),
                           GradOpInType::Out});
  // initial_c grad
  inInfoMapping.push_back({getLastCellStateGradInIndex(),
                           LSTMOp::getLastCellStateOutIndex(),
                           GradOpInType::GradOut});
}

const std::map<int, int> &LSTMGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo =
      [baseOutInfo = BaseOnnxRNNGradOp::gradOutToNonGradIn()]() mutable {
        baseOutInfo[LSTMGradOp::getInitialCOutIndex()] =
            LSTMOp::getInitialCInIndex();
        return baseOutInfo;
      }();

  return outInfo;
}

PopartLSTMOp::PopartLSTMOp(
    const OperatorIdentifier &opid_,
    bool outputFullSequence_,
    const Op::Settings &settings_,
    const nonstd::optional<float> available_memory_proportion_)
    : PopartLSTMOp(opid_,
                   outputFullSequence_,
                   ActivationFunction::Tanh,
                   ActivationFunction::Sigmoid,
                   settings_,
                   available_memory_proportion_) {}

PopartLSTMOp::PopartLSTMOp(
    const OperatorIdentifier &opid_,
    bool outputFullSequence_,
    ActivationFunction activation,
    ActivationFunction recurrent_activation,
    const Op::Settings &settings_,
    const nonstd::optional<float> available_memory_proportion_)
    : Op(opid_, settings_), outputFullSequence(outputFullSequence_),
      activation(activation), recurrent_activation(recurrent_activation),
      available_memory_proportion(available_memory_proportion_) {}

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

nonstd::optional<float> PopartLSTMOp::getAvailableMemoryProportion() const {
  return available_memory_proportion;
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

  // If training, create an output for the intermediates.
  if (getIr().isTraining()) {

    // If the op is being cloned, or setup is being called a second time, the
    // output may already be connected; we do not need to recreate it.
    if (!output->hasIndex(getIntermediatesOutIndex())) {
      TensorId intermediates =
          (getScope() / logging::format("{}_intermediates", id)).str();
      if (getGraph().getTensors().contains(intermediates)) {
        connectOutTensor(getIntermediatesOutIndex(), intermediates);
      } else {
        createAndConnectOutTensor(getIntermediatesOutIndex(), intermediates);
      }
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
      recurrent_activation(fwd_op.getRecurrentActivation()) {
  populateInInfo();
}

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

void PopartLSTMGradOp::populateInInfo() {
  inInfoMapping = {
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
    inInfoMapping.push_back({getSequenceLensInIndex(),
                             PopartLSTMOp::getSequenceLensInIndex(),
                             GradOpInType::In});
  }
}

const std::vector<GradInOutMapper> &PopartLSTMGradOp::gradInputInfo() const {
  return inInfoMapping;
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

      nonstd::optional<float> availableMemoryProportion;
      if (info.attributes.hasAttribute(sAvailMemAttribute)) {
        availableMemoryProportion =
            info.attributes.getAttribute<Attributes::Float>(sAvailMemAttribute);
      }

      return std::unique_ptr<Op>(new LSTMOp(info.opid,
                                            hidden_size,
                                            activation,
                                            recurrent_activation,
                                            info.settings,
                                            availableMemoryProportion));
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

      nonstd::optional<float> availableMemoryProportion;
      if (info.attributes.hasAttribute(sAvailMemAttribute)) {
        availableMemoryProportion =
            info.attributes.getAttribute<Attributes::Float>(sAvailMemAttribute);
      }

      return std::unique_ptr<Op>(new PopartLSTMOp(info.opid,
                                                  outputFullSequence,
                                                  info.settings,
                                                  availableMemoryProportion));
    },
    true);

} // namespace

} // namespace popart
