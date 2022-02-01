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
    : BaseOnnxRNNOp(_opid, hidden_size, settings_),
      activation_attribute(activation) {}

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
  auto max_seq_length = getMaxSeqLength();
  auto num_directions = getNumDirections();
  auto batch_size     = getBatchSize();
  auto data_type      = inInfo(getInputInIndex()).data_type();

  // full hidden state
  if (output->hasIndex(getFullHiddenStateOutIndex())) {
    Shape y_full_shape{max_seq_length, num_directions, batch_size, hidden_size};
    outInfo(getFullHiddenStateOutIndex()) = {data_type, y_full_shape};
  }
  // the last element from the hidden state
  if (output->hasIndex(getLastHiddenStateOutIndex())) {
    Shape y_last_shape{num_directions, batch_size, hidden_size};
    outInfo(getLastHiddenStateOutIndex()) = {data_type, y_last_shape};
  }
}

int64_t RNNOp::getNumDirections() const { return 1; }

void RNNOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  BaseOnnxRNNOp::appendOutlineAttributes(os);
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
  if (index == getFullHiddenStateOutIndex()) {
    return 2;
  } else if (index == getLastHiddenStateOutIndex()) {
    return 1;
  } else {
    throw error("getOutBatchAxis of {} received an invalid index {}",
                debugName(),
                index);
  }
}

RNNGradOp::RNNGradOp(const RNNOp &fwd_op)
    : BaseOnnxRNNGradOp(Onnx::GradOperators::RNNGrad, fwd_op),
      activation_attribute(fwd_op.activation_attribute) {}

std::unique_ptr<Op> RNNGradOp::clone() const {
  return std::make_unique<RNNGradOp>(*this);
}

std::set<InIndex> RNNGradOp::optionalInputs() const {
  return {getLastHiddenStateGradInIndex(),
          getFullHiddenStateGradInIndex(),
          getBiasesInIndex(),
          getInitialHInIndex(),
          getSequenceLensInIndex()};
}

const std::vector<GradInOutMapper> &RNNGradOp::gradInputInfo() const {
  // Initialize RNN-specific inputs in addition to BaseOnnxRNNGradOp ones
  static const std::vector<GradInOutMapper> inInfo =
      [baseInInfo = BaseOnnxRNNGradOp::gradInputInfo()]() mutable {
        baseInInfo.push_back({RNNGradOp::getInitialHInIndex(), // initial_h
                              RNNOp::getInitialHInIndex(),
                              GradOpInType::In});
        baseInInfo.push_back({RNNGradOp::getInputWeightsInIndex(), // W
                              RNNOp::getInputWeightsInIndex(),
                              GradOpInType::In});
        baseInInfo.push_back({RNNGradOp::getRecurrenceWeightsInIndex(), // R
                              RNNOp::getRecurrenceWeightsInIndex(),
                              GradOpInType::In});
        baseInInfo.push_back({RNNGradOp::getBiasesInIndex(), // b
                              RNNOp::getBiasesInIndex(),
                              GradOpInType::In});
        return baseInInfo;
      }();

  return inInfo;
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
