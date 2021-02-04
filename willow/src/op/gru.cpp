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
             const Op::Settings &settings_)
    : Op(_opid, settings_), hidden_size_attribute(hidden_size),
      direction_attribute(direction) {
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
  int64_t hidden_size = 0;
  if (!hidden_size_attribute) {
    hidden_size = inShape(getRecurrenceInIndex())[2];
  } else if (*hidden_size_attribute != inShape(getRecurrenceInIndex())[2]) {
    throw error("GRUOp hidden_size attribute, {}, does not match calculated "
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

  createPassThroughOutput("initstateoutput",
                          getInitStateOutputPassThroughIndex(),
                          {data_type, Shape{batch_size, hidden_size}});
  createPassThroughOutput(
      "intermediates",
      getIntermediatesPassThroughIndex(),
      {data_type,
       Shape{seq_length, getNumIntermediates(), batch_size, hidden_size}});
  createPassThroughOutput("inputweights",
                          getInputWeightsPassThroughIndex(),
                          {data_type, Shape{3, input_size, hidden_size}});
  createPassThroughOutput("outputweights",
                          getOutputWeightsPassThroughIndex(),
                          {data_type, Shape{3, hidden_size, hidden_size}});
  createPassThroughOutput("biases",
                          getBiasesPassThroughIndex(),
                          {data_type, Shape{3, hidden_size}});
  createPassThroughOutput(
      "input",
      getInputPassThroughIndex(),
      {data_type, Shape{seq_length, batch_size, input_size}});
  createPassThroughOutput(
      "output",
      getOutputPassThroughIndex(),
      {data_type, Shape{seq_length, batch_size, hidden_size}});

} // namespace popart

// Issue here somewhere with out index
void GRUOp::createPassThroughOutput(const TensorId &new_id,
                                    OutIndex pass_through_index,
                                    const TensorInfo &out_info) {
  auto tensor_id =
      (getScope() / logging::format("gru({})_{}", id, new_id)).str();
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

unsigned GRUOp::getNumChannels() const { return 1; }

int64_t GRUOp::getSeqLength() const { return inShape(getInputInIndex())[0]; }

int64_t GRUOp::getBatchSize() const { return inShape(getInputInIndex())[1]; }

int64_t GRUOp::getInputSize() const { return inShape(getInputInIndex())[2]; }

int64_t GRUOp::getNumDirections() const {
  if (getDirectionAttribute() == "bidirectional") {
    return 2;
  } else {
    return 1;
  }
}

int64_t GRUOp::getHiddenSize() const {
  return inShape(getRecurrenceInIndex())[2];
}

bool GRUOp::hasBiasInput() const { return input->hasIndex(getBiasInIndex()); }

bool GRUOp::hasInitialHInput() const {
  return input->hasIndex(getInitialHInIndex());
}

bool GRUOp::hasOutput(OutIndex index) const { return output->hasIndex(index); }

std::set<InIndex> GRUOp::optionalInputs() const {
  return {getSequenceLensInIndex()};
}

void GRUOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("hidden_size", getHiddenSizeAttribute());
  os.appendAttribute("direction", getDirectionAttribute());
}

view::Regions GRUOp::aliases(InIndex in, OutIndex out) const {
  if (in == getInputInIndex() && out == getInputPassThroughIndex()) {
    return {view::Region::getFull(inShape(in))};
  } else {
    return {view::Region::getEmpty(inRank(in))};
  }
}

view::RegMap GRUOp::fwdRegMap(InIndex in, OutIndex out) const {
  auto emptyRegion = view::Region::getEmpty(outRank(out));
  if (in == getInputInIndex() && out == getInputPassThroughIndex()) {
    return [emptyRegion](const view::Region &r) { return view::Regions(1, r); };
  } else {
    return [emptyRegion](const view::Region &r) {
      return view::Regions(1, emptyRegion);
    };
  }
}

view::RegMap GRUOp::bwdRegMap(InIndex in, OutIndex out) const {
  auto emptyRegion = view::Region::getEmpty(inRank(in));
  if (in == getInputInIndex() && out == getInputPassThroughIndex()) {
    return [emptyRegion](const view::Region &r) { return view::Regions(1, r); };
  } else {
    return [emptyRegion](const view::Region &r) {
      return view::Regions(1, emptyRegion);
    };
  }
}

GRUGradOp::GRUGradOp(const GRUOp &fwd_op)
    : Op(Onnx::GradOperators::GRUGrad, fwd_op.getSettings()),
      forward_op(fwd_op) {}

std::unique_ptr<Op> GRUGradOp::clone() const {
  return std::make_unique<GRUGradOp>(*this);
}

void GRUGradOp::setup() {
  outInfo(getInputOutIndex())   = forward_op.inInfo(GRUOp::getInputInIndex());
  outInfo(getWeightsOutIndex()) = forward_op.inInfo(GRUOp::getWeightsInIndex());
  outInfo(getRecurrenceOutIndex()) =
      forward_op.inInfo(GRUOp::getRecurrenceInIndex());

  if (forward_op.hasBiasInput()) {
    outInfo(getBiasOutIndex()) = forward_op.inInfo(GRUOp::getBiasInIndex());
  }
  if (forward_op.hasInitialHInput()) {
    outInfo(getInitialHOutIndex()) =
        forward_op.inInfo(GRUOp::getInitialHInIndex());
  }
}

bool GRUGradOp::hasHiddenStateGradInput() const {
  return input->hasIndex(getHiddenStateOutputGradInIndex());
}

std::set<InIndex> GRUGradOp::optionalInputs() const {
  return {getHiddenStateOutputGradInIndex()};
}

const std::vector<GradInOutMapper> &GRUGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInitStateOutputInIndex(),
       GRUOp::getInitStateOutputPassThroughIndex(),
       GradOpInType::Out},
      {getIntermediatesInIndex(),
       GRUOp::getIntermediatesPassThroughIndex(),
       GradOpInType::Out},
      {getInputWeightsInIndex(),
       GRUOp::getInputWeightsPassThroughIndex(),
       GradOpInType::Out},
      {getOutputWeightsInIndex(),
       GRUOp::getOutputWeightsPassThroughIndex(),
       GradOpInType::Out},
      {getBiasesInIndex(),
       GRUOp::getBiasesPassThroughIndex(),
       GradOpInType::Out},
      {getInputInIndex(), GRUOp::getInputPassThroughIndex(), GradOpInType::Out},
      {getOutputInIndex(),
       GRUOp::getOutputPassThroughIndex(),
       GradOpInType::Out},
      {getHiddenStateOutputGradInIndex(),
       GRUOp::getHiddenStateOutIndex(),
       GradOpInType::GradOut},
      {getOutputGradInIndex(),
       GRUOp::getOutputOutIndex(),
       GradOpInType::GradOut}};
  return inInfo;
}

const std::map<int, int> &GRUGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getInputOutIndex(), GRUOp::getInputInIndex()},
      {getWeightsOutIndex(), GRUOp::getWeightsInIndex()},
      {getRecurrenceOutIndex(), GRUOp::getRecurrenceInIndex()},
      {getBiasOutIndex(), GRUOp::getBiasInIndex()},
      {getInitialHOutIndex(), GRUOp::getInitialHInIndex()}};

  return outInfo;
}

const GRUOp &GRUGradOp::getForwardOp() const { return forward_op; }

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
      // cannot check hidden_size till inputs are connected
      nonstd::optional<int64_t> hidden_size;
      if (info.attributes.hasAttribute("hidden_size")) {
        hidden_size =
            info.attributes.getAttribute<Attributes::Int>("hidden_size");
      }

      return std::unique_ptr<Op>(
          new GRUOp(info.opid, hidden_size, direction, info.settings));
    },
    true);

} // namespace
} // namespace popart
