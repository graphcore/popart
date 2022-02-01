// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/gru.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>

namespace popart {

GRUOp::GRUOp(const OperatorIdentifier &_opid,
             nonstd::optional<int64_t> hidden_size,
             std::string direction,
             bool linear_before_reset,
             const Op::Settings &settings_)
    : BaseOnnxRNNOp(_opid, hidden_size, settings_),
      direction_attribute(direction),
      linear_before_reset_attribute(linear_before_reset) {
  // TODO : Use the output_sequence attribute in version 1, if needed. Currently
  // GRU_1 not supported.
}

std::unique_ptr<Op> GRUOp::clone() const {
  return std::make_unique<GRUOp>(*this);
}

std::vector<std::unique_ptr<Op>> GRUOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<GRUGradOp>(*this));
  return upops;
}

bool GRUOp::isTraining() const {
  return getGraph().getIr().getExecutionMode() == Ir::ExecutionMode::Training;
}

void GRUOp::trySetOutInfo(OutIndex index, const TensorInfo &info) {
  if (output->hasIndex(index)) {
    outInfo(index) = info;
  }
}

void GRUOp::setup() {
  if (input->hasIndex(getSequenceLensInIndex())) {
    logging::op::warn("Gru optional input `sequence_lens' will be ignored");
  }

  checkHiddenSize();

  int64_t hidden_size = getHiddenSize();
  auto max_seq_length = getMaxSeqLength();
  auto num_directions = getNumDirections();
  auto batch_size     = getBatchSize();
  auto data_type      = inInfo(getInputInIndex()).data_type();
  auto input_size     = getInputSize();

  Shape y_shape{max_seq_length, num_directions, batch_size, hidden_size};

  trySetOutInfo(getFullHiddenStateOutIndex(), {data_type, y_shape});

  Shape yhc_shape{num_directions, batch_size, hidden_size};
  trySetOutInfo(getLastHiddenStateOutIndex(), {data_type, yhc_shape});

  maybeCreatePassThroughOutput("initstateoutput",
                               getInitialHPassThroughIndex(),
                               {data_type, Shape{batch_size, hidden_size}});
  maybeCreatePassThroughOutput(
      "intermediates",
      getIntermediatesPassThroughIndex(),
      {data_type,
       Shape{max_seq_length, getNumIntermediates(), batch_size, hidden_size}});
  maybeCreatePassThroughOutput("inputweights",
                               getInputWeightsPassThroughIndex(),
                               {data_type, Shape{3, input_size, hidden_size}});
  maybeCreatePassThroughOutput("outputweights",
                               getRecurrenceWeightsPassThroughIndex(),
                               {data_type, Shape{3, hidden_size, hidden_size}});
  maybeCreatePassThroughOutput("biases",
                               getBiasesPassThroughIndex(),
                               {data_type, Shape{getNumBiases(), hidden_size}});
}

// Issue here somewhere with out index
void GRUOp::maybeCreatePassThroughOutput(const TensorId &new_id,
                                         OutIndex pass_through_index,
                                         const TensorInfo &out_info) {
  // If the op is being cloned, or setup is being called a second time, the
  // output may already be connected; we do not need to recreate it.
  if (!hasOutput(pass_through_index)) {
    auto tensor_id =
        (getScope() / logging::format("gru({})_{}", id, new_id)).str();
    if (getGraph().getTensors().contains(tensor_id)) {
      connectOutTensor(pass_through_index, tensor_id);
    } else {
      createAndConnectOutTensor(pass_through_index, tensor_id);
    }
  }
  outInfo(pass_through_index) = out_info;
}

unsigned GRUOp::getNumChannels() const { return 1; }

int64_t GRUOp::getNumDirections() const {
  if (getDirectionAttribute() == "bidirectional") {
    return 2;
  } else {
    return 1;
  }
}

bool GRUOp::hasOutput(OutIndex index) const { return output->hasIndex(index); }

void GRUOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  BaseOnnxRNNOp::appendOutlineAttributes(os);
  os.appendAttribute("direction", getDirectionAttribute());
  os.appendAttribute("linear_before_reset", getLinearBeforeResetAttribute());
}

GRUGradOp::GRUGradOp(const GRUOp &fwd_op)
    : BaseOnnxRNNGradOp(Onnx::GradOperators::GRUGrad, fwd_op),
      linear_before_reset_attribute(fwd_op.getLinearBeforeResetAttribute()) {}

std::unique_ptr<Op> GRUGradOp::clone() const {
  return std::make_unique<GRUGradOp>(*this);
}

std::set<InIndex> GRUGradOp::optionalInputs() const {
  return {getLastHiddenStateGradInIndex(),
          getSequenceLensInIndex(),
          getFullHiddenStateGradInIndex()};
}

const std::vector<GradInOutMapper> &GRUGradOp::gradInputInfo() const {
  // Initialize GRU-specific inputs in addition to BaseOnnxRNNGradOp ones
  static const std::vector<GradInOutMapper> inInfo =
      [baseInInfo = BaseOnnxRNNGradOp::gradInputInfo()]() mutable {
        baseInInfo.push_back({GRUGradOp::getInitialHInIndex(),
                              GRUOp::getInitialHPassThroughIndex(),
                              GradOpInType::Out});
        baseInInfo.push_back({GRUGradOp::getInputWeightsInIndex(),
                              GRUOp::getInputWeightsPassThroughIndex(),
                              GradOpInType::Out});
        baseInInfo.push_back({GRUGradOp::getRecurrenceWeightsInIndex(),
                              GRUOp::getRecurrenceWeightsPassThroughIndex(),
                              GradOpInType::Out});
        baseInInfo.push_back({GRUGradOp::getBiasesInIndex(),
                              GRUOp::getBiasesPassThroughIndex(),
                              GradOpInType::Out});
        baseInInfo.push_back({GRUGradOp::getIntermediatesInIndex(),
                              GRUOp::getIntermediatesPassThroughIndex(),
                              GradOpInType::Out});
        return baseInInfo;
      }();
  return inInfo;
}

namespace {

static OpDefinition::DataTypes T  = {DataType::FLOAT16, DataType::FLOAT};
static OpDefinition::DataTypes T1 = {DataType::INT32};

static OpDefinition gruOpDef({OpDefinition::Inputs({
                                  {"X", T},
                                  {"W", T},
                                  {"R", T},
                                  {"B", T},
                                  //{"sequence_lens", T1 }, // not supported
                                  {"initial_h", T}
                                  //{"P", T },// peep hole not supported
                              }),
                              OpDefinition::Outputs({{"Y", T}, {"Y_h", T}}),
                              OpDefinition::Attributes({
                                  //{"activation_alpha", {"*"}},
                                  //{"activation_beta", {"*"}},
                                  //{"activations", {"*"}},
                                  //{"clip", {"*"}},
                                  {"direction", {"*"}},
                                  {"hidden_size", {"*"}},
                                  {"linear_before_reset", {"0"}},
                              })});

static OpCreator<GRUOp> gruOpCreator(
    OpDefinitions({{Onnx::Operators::GRU_3, gruOpDef},
                   {Onnx::Operators::GRU_7, gruOpDef}}),
    [](const OpCreatorInfo &info) {
      if (info.attributes.hasAttribute("activations")) {
        throw error("GRUOp attribute `activations' is not supported");
      }
      if (info.attributes.hasAttribute("activation_alpha")) {
        throw error("GRUOp attribute `activation_alpha' is not supported");
      }
      if (info.attributes.hasAttribute("activation_beta")) {
        throw error("GRUOp attribute `activation_alpha' is not supported");
      }
      if (info.attributes.hasAttribute("clip")) {
        throw error("GRUOp attribute `clip' is not supported");
      }
      if (info.attributes.getAttribute<Attributes::Int>("input_forget", 0) !=
          0) {
        throw error("GRUOp attribute `input_forget' must be set to 0");
      }
      std::string direction = "forward";
      if (info.attributes.hasAttribute("direction")) {
        direction =
            info.attributes.getAttribute<Attributes::String>("direction");
      }
      bool linear_before_reset = 0;
      if (info.attributes.hasAttribute("linear_before_reset")) {
        linear_before_reset = info.attributes.getAttribute<Attributes::Int>(
            "linear_before_reset");
      }
      // cannot check hidden_size till inputs are connected
      nonstd::optional<int64_t> hidden_size;
      if (info.attributes.hasAttribute("hidden_size")) {
        hidden_size =
            info.attributes.getAttribute<Attributes::Int>("hidden_size");
      }

      return std::unique_ptr<Op>(new GRUOp(info.opid,
                                           hidden_size,
                                           direction,
                                           linear_before_reset,
                                           info.settings));
    },
    true);

} // namespace
} // namespace popart
