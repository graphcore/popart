// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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
               boost::optional<int64_t> hidden_size,
               const Op::Settings &settings_)
    : Op(_opid, settings_), hidden_size_attribute(hidden_size) {
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

void LSTMOp::setup() {
  if (input->hasIndex(getPeepholeInIndex())) {
    throw error("Popart does not support peephole connections");
  }
  if (input->hasIndex(getSequenceLensInIndex())) {
    logging::op::warn("Lstm optional input `sequence_lens' will be ignored");
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

  auto seq_length     = getSeqLength();
  auto num_directions = getNumDirections();
  auto batch_size     = getBatchSize();
  auto data_type      = inInfo(getInputInIndex()).data_type();
  auto input_size     = getInputSize();

  Shape y_shape{seq_length, num_directions, batch_size, hidden_size};

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
       Shape{seq_length, getNumIntermediates(), batch_size, hidden_size}});
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
      {data_type, Shape{seq_length, batch_size, input_size}});
  createPassThroughOutput(
      "output",
      getOutputPassThroughIndex(),
      {data_type, Shape{seq_length, batch_size, hidden_size}});
}

void LSTMOp::createPassThroughOutput(const TensorId &new_id,
                                     OutIndex pass_through_index,
                                     const TensorInfo &out_info) {
  auto tensor_id = logging::format("lstm({})_{}", id, new_id);
  createAndConnectOutTensor(pass_through_index, tensor_id);
  outInfo(pass_through_index) = out_info;
}

unsigned LSTMOp::getNumChannels() const { return 1; }

int64_t LSTMOp::getSeqLength() const { return inShape(getInputInIndex())[0]; }

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

bool LSTMOp::hasOutput(OutIndex index) const { return output->hasIndex(index); }

std::set<InIndex> LSTMOp::optionalInputs() const {
  return {getSequenceLensInIndex()};
}

void LSTMOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("hidden_size", hidden_size_attribute);
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
  return {getCellStateOutputGradInIndex(), getHiddenStateOutputGradInIndex()};
}

const std::vector<GradInOutMapper> &LSTMGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
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
    : Op(opid_, settings_), outputFullSequence(outputFullSequence_) {}

std::unique_ptr<Op> PopartLSTMOp::clone() const {
  return std::make_unique<PopartLSTMOp>(*this);
}

std::vector<std::unique_ptr<Op>> PopartLSTMOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<PopartLSTMGradOp>(*this));
  return upops;
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
              {getSeqLength(), getBatchSize(), getInputSize()},
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
    outputShape = {getSeqLength(), getBatchSize(), getHiddenSize()};
  } else {
    outputShape = {getBatchSize(), getHiddenSize()};
  }

  outInfo(getOutputOutIndex())    = {dtype, outputShape};
  outInfo(getCellStateOutIndex()) = {dtype, {getBatchSize(), getHiddenSize()}};

  if (getIr().isTraining()) {
    createAndConnectOutTensor(getIntermediatesOutIndex(),
                              logging::format("{}_intermediates", id));
    outInfo(getIntermediatesOutIndex()) = {dtype,
                                           {getSeqLength(),
                                            getNumIntermediates(),
                                            getBatchSize(),
                                            getHiddenSize()}};
  }
}

bool PopartLSTMOp::hasBiasesInput() const {
  return input->hasIndex(getBiasesInIndex());
}

std::set<InIndex> PopartLSTMOp::optionalInputs() const {
  return {getBiasesInIndex(), getInitialStateInIndex()};
}

int64_t PopartLSTMOp::getSeqLength() const {
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

PopartLSTMGradOp::PopartLSTMGradOp(const PopartLSTMOp &fwdOp)
    : Op(Onnx::GradOperators::PopartLSTMGrad, fwdOp.getSettings()),
      outputFullSequence(fwdOp.outputFullSequence),
      forwardCellStateGradId(
          getGradId(fwdOp.outId(PopartLSTMOp::getCellStateOutIndex()))) {}

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
          getFwdCellStateGradInIndex()};
}

int64_t PopartLSTMGradOp::getInputSize() const {
  return inShape(getInputInIndex()).at(2);
}

int64_t PopartLSTMGradOp::getSeqLength() const {
  return inShape(getInputInIndex()).at(0);
}

int64_t PopartLSTMGradOp::getBatchSize() const {
  return inShape(getInputInIndex()).at(1);
}

int64_t PopartLSTMGradOp::getHiddenSize() const {
  return inShape(getWeightsInIndex()).at(2);
}

const std::vector<GradInOutMapper> &PopartLSTMGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
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
                   //{"sequence_lens", T1 }, // not supported
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

static OpCreator<LSTMOp> lstmOpCreator(
    OpDefinitions({{Onnx::Operators::LSTM_1, lstmOpDef},
                   {Onnx::Operators::LSTM_7, lstmOpDef}}),
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      if (attr.hasAttribute("activations")) {
        throw error("LSTMOp attribute `activations' is not supported");
      }
      if (attr.hasAttribute("activation_alpha")) {
        throw error("LSTMOp attribute `activation_alpha' is not supported");
      }
      if (attr.hasAttribute("activation_beta")) {
        throw error("LSTMOp attribute `activation_alpha' is not supported");
      }

      if (attr.hasAttribute("clip")) {
        throw error("LSTMOp attribute `clip' is not supported");
      }

      if (attr.getAttribute<Attributes::String>("direction", "forward") !=
          "forward") {
        throw error("LSTMOp attribute `direction' must be unset or `forward'");
      }

      if (attr.getAttribute<Attributes::Int>("input_forget", 0) != 0) {
        throw error("LSTMOp attribute `input_forget' must be set to 0");
      }

      // cannot check hidden_size till inputs are connected
      boost::optional<int64_t> hidden_size;
      if (attr.hasAttribute("hidden_size")) {
        hidden_size = attr.getAttribute<Attributes::Int>("hidden_size");
      }

      return std::unique_ptr<Op>(new LSTMOp(_opid, hidden_size, settings));
    },
    true);

static OpDefinition popartLstmOpDef(
    {OpDefinition::Inputs({
         {"X", T},
         {"Weights", T},
         {"Bias", T},
         {"InitiState", T},
     }),
     OpDefinition::Outputs({{"Output", T}, {"CellState", T}}),
     OpDefinition::Attributes({{"output_full_sequence", {"*"}}})});

static OpCreator<PopartLSTMOp> popartLSTMOpCreator(
    OpDefinitions({{Onnx::CustomOperators::LSTM_1, popartLstmOpDef}}),
    [](const OperatorIdentifier &opid,
       const Op::Settings &settings,
       const Attributes &attr) {
      bool outputFullSequence =
          attr.getAttribute<Attributes::Int>("output_full_sequence", 1) != 0;

      return std::unique_ptr<Op>(
          new PopartLSTMOp(opid, outputFullSequence, settings));
    },
    true);

} // namespace

} // namespace popart
