// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/op/lstmutil.hpp>
#include <popart/op/rnn.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>

namespace popart {

RNNOp::RNNOp(const OperatorIdentifier &_opid,
             ActivationFunction activation,
             nonstd::optional<int64_t> hidden_size,
             const Op::Settings &settings_)
    : Op(_opid, settings_), activation_attribute(activation),
      hidden_size_attribute(hidden_size) {}

std::unique_ptr<Op> RNNOp::clone() const {
  return std::make_unique<RNNOp>(*this);
}

std::vector<std::unique_ptr<Op>> RNNOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<RNNGradOp>(*this));
  return upops;
}

void RNNOp::setup() {
  if (input->hasIndex(getSequenceLensInIndex())) {
    logging::op::warn("optional input `sequence_lens' of {} will be ignored",
                      debugName());
  }
  checkHiddenSize();

  int64_t hidden_size = getHiddenSize();
  auto seq_length     = getSeqLength();
  auto num_directions = getNumDirections();
  auto batch_size     = getBatchSize();
  auto data_type      = inInfo(getInputInIndex()).data_type();

  // full hidden state
  if (output->hasIndex(getFullOutputOutIndex())) {
    Shape y_full_shape{seq_length, num_directions, batch_size, hidden_size};
    outInfo(getFullOutputOutIndex()) = {data_type, y_full_shape};
  }
  // the last element from the hidden state
  if (output->hasIndex(getLastOutputOutIndex())) {
    Shape y_last_shape{num_directions, batch_size, hidden_size};
    outInfo(getLastOutputOutIndex()) = {data_type, y_last_shape};
  }
}

int64_t RNNOp::getSeqLength() const {
  int seq_length_index = 0;
  return inShape(getInputInIndex())[seq_length_index];
}

int64_t RNNOp::getBatchSize() const {
  int batch_size_index = 1;
  return inShape(getInputInIndex())[batch_size_index];
}

int64_t RNNOp::getInputSize() const {
  int input_size_index = 2;
  return inShape(getInputInIndex())[input_size_index];
}

int64_t RNNOp::getNumDirections() const { return 1; }

void RNNOp::checkHiddenSize() const {
  if (hidden_size_attribute && *hidden_size_attribute != getHiddenSize()) {
    throw error("hidden_size attribute passed to {} ({}) does not match "
                "hidden size calculated from recurrenceWeights tensor ({}).",
                debugName(),
                *hidden_size_attribute,
                getHiddenSize());
  }
}

int64_t RNNOp::getHiddenSize() const {
  int hidden_size_index = 2;
  return inShape(getRecurrenceWeightsInIndex())[hidden_size_index];
}

bool RNNOp::hasBiasInput() const { return input->hasIndex(getBiasInIndex()); }

bool RNNOp::hasInitialHInput() const {
  return input->hasIndex(getInitialHInIndex());
}

std::set<InIndex> RNNOp::optionalInputs() const {
  return {getSequenceLensInIndex(), getInitialHInIndex(), getBiasInIndex()};
}

void RNNOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("hidden_size", hidden_size_attribute);
  os.appendAttribute("activations", activation_attribute);
}

int RNNOp::getInBatchAxis(InIndex index) const {
  if (index == getInputInIndex() || index == getInitialHInIndex()) {
    return 1;
  } else if (index == getSequenceLensInIndex()) {
    return 0;
  }
  return -1;
}

int RNNOp::getOutBatchAxis(OutIndex index) const {
  if (index == getFullOutputOutIndex()) {
    return 2;
  } else if (index == getLastOutputOutIndex()) {
    return 1;
  } else {
    throw error("getOutBatchAxis of {} received an invalid index {}",
                debugName(),
                index);
  }
}

RNNGradOp::RNNGradOp(const RNNOp &fwd_op)
    : Op(Onnx::GradOperators::RNNGrad, fwd_op.getSettings()),
      hasBiasInput(fwd_op.hasBiasInput()),
      hasInitialHInput(fwd_op.hasInitialHInput()),
      batch_size(fwd_op.getBatchSize()), input_size(fwd_op.getInputSize()),
      seq_length(fwd_op.getSeqLength()), hidden_size(fwd_op.getHiddenSize()),
      activation_attribute(fwd_op.activation_attribute),
      fwdInputInInfo(fwd_op.inInfo(RNNOp::getInputInIndex())),
      fwdInputWeightsInInfo(fwd_op.inInfo(RNNOp::getInputWeightsInIndex())),
      fwdRecurrenceWeightsInInfo(
          fwd_op.inInfo(RNNOp::getRecurrenceWeightsInIndex())),
      fwdBiasInInfo(getBiasInInfo(fwd_op)),
      fwdInitialHInInfo(getInitialHInInfo(fwd_op)) {}

nonstd::optional<TensorInfo> RNNGradOp::getBiasInInfo(const RNNOp &fwd_op) {
  if (hasBiasInput) {
    return fwd_op.inInfo(RNNOp::getBiasInIndex());
  }
  return nonstd::nullopt;
}

nonstd::optional<TensorInfo> RNNGradOp::getInitialHInInfo(const RNNOp &fwd_op) {
  if (hasInitialHInput) {
    return fwd_op.inInfo(RNNOp::getInitialHInIndex());
  }
  return nonstd::nullopt;
}

std::unique_ptr<Op> RNNGradOp::clone() const {
  return std::make_unique<RNNGradOp>(*this);
}

void RNNGradOp::setup() {
  outInfo(getInputOutIndex())             = fwdInputInInfo;
  outInfo(getInputWeightsOutIndex())      = fwdInputWeightsInInfo;
  outInfo(getRecurrenceWeightsOutIndex()) = fwdRecurrenceWeightsInInfo;

  if (hasBiasInput) {
    outInfo(getBiasOutIndex()) = fwdBiasInInfo.value();
  }
  if (hasInitialHInput) {
    outInfo(getInitialHOutIndex()) = fwdInitialHInInfo.value();
  }
}

bool RNNGradOp::hasLastOutputGradInput() const {
  return input->hasIndex(getLastOutputGradInIndex());
}
bool RNNGradOp::hasFullOutputGradInput() const {
  return input->hasIndex(getFullOutputGradInIndex());
}

std::set<InIndex> RNNGradOp::optionalInputs() const {
  return {getLastOutputGradInIndex(),
          getFullOutputGradInIndex(),
          getBiasInIndex(),
          getInitialHInIndex()};
}

const std::vector<GradInOutMapper> &RNNGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInitialHInIndex(), // initial_H
       RNNOp::getInitialHInIndex(),
       GradOpInType::In},
      {getInputWeightsInIndex(), // W
       RNNOp::getInputWeightsInIndex(),
       GradOpInType::In},
      {getRecurrenceWeightsInIndex(), // R
       RNNOp::getRecurrenceWeightsInIndex(),
       GradOpInType::In},
      {getBiasInIndex(), // b
       RNNOp::getBiasInIndex(),
       GradOpInType::In},
      {getInputInIndex(), RNNOp::getInputInIndex(), GradOpInType::In},
      {getFullOutputInIndex(), // output_full
       RNNOp::getFullOutputOutIndex(),
       GradOpInType::Out},
      {getLastOutputGradInIndex(), // d_output_last
       RNNOp::getLastOutputOutIndex(),
       GradOpInType::GradOut},
      {getFullOutputGradInIndex(), // d_output_full
       RNNOp::getFullOutputOutIndex(),
       GradOpInType::GradOut}};

  return inInfo;
}

const std::map<int, int> &RNNGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getInputOutIndex(), RNNOp::getInputInIndex()},
      {getInputWeightsOutIndex(), RNNOp::getInputWeightsInIndex()},
      {getRecurrenceWeightsOutIndex(), RNNOp::getRecurrenceWeightsInIndex()},
      {getBiasOutIndex(), RNNOp::getBiasInIndex()},
      {getInitialHOutIndex(), RNNOp::getInitialHInIndex()}};

  return outInfo;
}

namespace {

static OpDefinition::DataTypes T  = {DataType::FLOAT16, DataType::FLOAT};
static OpDefinition::DataTypes T1 = {DataType::INT32};

static OpDefinition
    rnnOpDef({OpDefinition::Inputs({{"X", T},
                                    {"W", T},
                                    {"R", T},
                                    {"B", T},
                                    //{"sequence_lens", T1 }, // not supported
                                    {"initial_h", T}}),
              OpDefinition::Outputs({{"Y", T}, {"Y_h", T}}),
              OpDefinition::Attributes({
                  //{"activation_alpha", {"*"}}, // not supported
                  //{"activation_beta", {"*"}}, // not supported
                  {"activations", {"*"}},
                  //{"clip", {"*"}}, // not supported
                  //{"direction", {"*"}}, // not supported
                  {"hidden_size", {"*"}},
              })});
namespace {

std::vector<ActivationFunction> readActivations(const Attributes &attributes,
                                                const OpCreatorInfo &info) {
  ActivationFunction activation = ActivationFunction::Tanh;

  if (attributes.hasAttribute("activations")) {
    std::vector<std::string> afs;
    afs = attributes.getAttribute<std::vector<std::string>>("activations");
    logging::trace("{} has activations {}", info.debugName(), afs);
    if (afs.size() == 1) {
      activation = fromString(afs[0]);
      if (activation == ActivationFunction::Invalid) {
        throw error("Activation type '{}' is not supported for {}",
                    afs[0],
                    info.debugName());
      }
    } else {
      throw error("{} only supports 1 activation, {} were specified",
                  info.debugName(),
                  afs.size());
    }
  }

  return {activation};
}

} // namespace

static OpCreator<RNNOp> rnnOpCreator(
    OpDefinitions({{Onnx::Operators::RNN_7, rnnOpDef}}),
    [](const OpCreatorInfo &info) {
      // ONNX specifies that activations have to be passed as an array,
      // as different directions can have different activations.
      const auto activations = readActivations(info.attributes, info);
      const auto activation  = activations[0];
      if (info.attributes.hasAttribute("activation_alpha")) {
        throw error("Attribute `activation_alpha' of {} is not supported",
                    info.debugName());
      }
      if (info.attributes.hasAttribute("activation_beta")) {
        throw error("Attribute `activation_beta' of {} is not supported",
                    info.debugName());
      }
      if (info.attributes.hasAttribute("clip")) {
        throw error("Attribute `clip' of {} is not supported",
                    info.debugName());
      }
      if (info.attributes.hasAttribute("direction")) {
        auto direction =
            info.attributes.getAttribute<Attributes::String>("direction");
        if (direction != "forward") {
          throw error("{} only supports `forward' direction", info.debugName());
        }
      }
      nonstd::optional<int64_t> hidden_size;
      if (info.attributes.hasAttribute("hidden_size")) {
        hidden_size =
            info.attributes.getAttribute<Attributes::Int>("hidden_size");
      }

      return std::make_unique<RNNOp>(
          info.opid, activation, hidden_size, info.settings);
    },
    true);

} // namespace
} // namespace popart