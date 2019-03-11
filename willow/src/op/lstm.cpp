#include <vector>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/lstm.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

LSTMOp::LSTMOp(const OperatorIdentifier &_opid,
               boost::optional<int64_t> hidden_size,
               const Op::Settings &settings_)
    : Op(_opid, settings_), hidden_size_attribute(hidden_size) {
  // TODO : Use the output_sequence attribute in version 1
}

std::unique_ptr<Op> LSTMOp::clone() const { return make_unique<LSTMOp>(*this); }

std::vector<std::unique_ptr<Op>> LSTMOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<LSTMGradOp>(*this));
  return upops;
}

void LSTMOp::setup() {
  if (input->hasIndex(getPeepholeInIndex())) {
    throw error("Poponnx does not support peephole connections");
  }
  if (input->hasIndex(getSequenceLensInIndex())) {
    logging::op::warn("Lstm optional input `sequence_lens' will be ignored");
  }
  if (hidden_size_attribute && *hidden_size_attribute != getHiddenSize()) {
    throw error("LSTMOp hidden_size attribute, {}, does not match calculated "
                "hidden size, {}.",
                *hidden_size_attribute,
                getHiddenSize());
  }

  auto seq_length     = getSeqLength();
  auto num_directions = getNumDirections();
  auto batch_size     = getBatchSize();
  auto data_type      = inInfo(getInputInIndex()).data_type();
  auto hidden_size    = getHiddenSize();
  auto input_size     = getInputSize();

  Shape y_shape{seq_length, num_directions, batch_size, hidden_size};

  outInfo(getOutputOutIndex()) = {data_type, y_shape};

  Shape yhc_shape{num_directions, batch_size, hidden_size};
  outInfo(getHiddenStateOutIndex()) = {data_type, yhc_shape};
  outInfo(getCellStateOutIndex())   = {data_type, yhc_shape};

  createPassThroughOutput("_initstateoutput",
                          getInitStateOutputPassThroughIndex(),
                          {data_type, Shape{batch_size, hidden_size}});
  createPassThroughOutput("_initstatecellstate",
                          getInitStateCellStatePassThroughIndex(),
                          {data_type, Shape{batch_size, hidden_size}});
  createPassThroughOutput(
      "_intermediates",
      getIntermediatesPassThroughIndex(),
      {data_type,
       Shape{seq_length, getNumIntermediates(), batch_size, hidden_size}});
  createPassThroughOutput("_inputweights",
                          getInputWeightsPassThroughIndex(),
                          {data_type, Shape{4, input_size, hidden_size}});
  createPassThroughOutput("_outputweights",
                          getOutputWeightsPassThroughIndex(),
                          {data_type, Shape{4, hidden_size, hidden_size}});
  createPassThroughOutput("_biases",
                          getBiasesPassThroughIndex(),
                          {data_type, Shape{4, hidden_size}});
  createPassThroughOutput(
      "_input",
      getInputPassThroughIndex(),
      {data_type, Shape{seq_length, batch_size, input_size}});
  createPassThroughOutput(
      "_output",
      getOutputPassThroughIndex(),
      {data_type, Shape{seq_length, batch_size, hidden_size}});
}

void LSTMOp::createPassThroughOutput(const TensorId &new_id,
                                     OutIndex pass_through_index,
                                     const TensorInfo &out_info) {
  auto tensor_id = outTensor(getOutputOutIndex())->id + new_id;
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

void LSTMOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("hidden_size", hidden_size_attribute);
}

LSTMGradOp::LSTMGradOp(const LSTMOp &fwd_op)
    : Op(Onnx::GradOperators::LSTMGrad, fwd_op.getSettings()),
      forward_op(fwd_op) {}

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

const std::vector<GradInOutMapper> &LSTMGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInitStateOutputInIndex(),
       LSTMOp::getInitStateOutputPassThroughIndex(),
       GradOpInType::OUT},
      {getInitStateCellStateInIndex(),
       LSTMOp::getInitStateCellStatePassThroughIndex(),
       GradOpInType::OUT},
      {getIntermediatesInIndex(),
       LSTMOp::getIntermediatesPassThroughIndex(),
       GradOpInType::OUT},
      {getInputWeightsInIndex(),
       LSTMOp::getInputWeightsPassThroughIndex(),
       GradOpInType::OUT},
      {getOutputWeightsInIndex(),
       LSTMOp::getOutputWeightsPassThroughIndex(),
       GradOpInType::OUT},
      {getBiasesInIndex(),
       LSTMOp::getBiasesPassThroughIndex(),
       GradOpInType::OUT},
      {getInputInIndex(),
       LSTMOp::getInputPassThroughIndex(),
       GradOpInType::OUT},
      {getOutputInIndex(),
       LSTMOp::getOutputPassThroughIndex(),
       GradOpInType::OUT},

      {getCellStateOutputGradInIndex(),
       LSTMOp::getCellStateOutIndex(),
       GradOpInType::GRADOUT},
      {getHiddenStateOutputGradInIndex(),
       LSTMOp::getHiddenStateOutIndex(),
       GradOpInType::GRADOUT},
      {getOutputGradInIndex(),
       LSTMOp::getOutputOutIndex(),
       GradOpInType::GRADOUT}};
  return inInfo;
}

const std::map<int, int> &LSTMGradOp::gradOutToNonGradIn() const {
  if (out_info.empty()) {
    out_info.insert(
        std::pair<int, int>(getInputOutIndex(), LSTMOp::getInputInIndex()));
    out_info.insert(
        std::pair<int, int>(getWeightsOutIndex(), LSTMOp::getWeightsInIndex()));
    out_info.insert(std::pair<int, int>(getRecurrenceOutIndex(),
                                        LSTMOp::getRecurrenceInIndex()));
    if (forward_op.hasBiasInput()) {
      out_info.insert(
          std::pair<int, int>(getBiasOutIndex(), LSTMOp::getBiasInIndex()));
    }
    if (forward_op.hasInitialHInput()) {
      out_info.insert(std::pair<int, int>(getInitialHOutIndex(),
                                          LSTMOp::getInitialHInIndex()));
    }
    if (forward_op.hasInitialCInput()) {
      out_info.insert(std::pair<int, int>(getInitialCOutIndex(),
                                          LSTMOp::getInitialCInIndex()));
    }
  }

  return out_info;
}

const LSTMOp &LSTMGradOp::getForwardOp() const { return forward_op; }

namespace {

static OpCreator<LSTMOp> lstmOpCreator(
    {Onnx::Operators::LSTM_1, Onnx::Operators::LSTM_7},
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

      // can not check hidden_size till inputs are connected
      boost::optional<int64_t> hidden_size;
      if (attr.hasAttribute("hidden_size")) {
        hidden_size = attr.getAttribute<Attributes::Int>("hidden_size");
      }

      return std::unique_ptr<Op>(new LSTMOp(_opid, hidden_size, settings));
    },
    true);
} // namespace

} // namespace poponnx
