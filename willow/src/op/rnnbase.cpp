// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <popart/op/rnnbase.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

BaseOnnxRNNOp::BaseOnnxRNNOp(const OperatorIdentifier &_opid,
                             nonstd::optional<int64_t> hidden_size,
                             const Op::Settings &settings_)
    : Op(_opid, settings_), hidden_size_attribute(hidden_size) {}

void BaseOnnxRNNOp::checkHiddenSize() const {
  if (hidden_size_attribute && *hidden_size_attribute != getHiddenSize()) {
    throw error("hidden_size attribute passed to {} ({}) does not match "
                "hidden size calculated from recurrenceWeights tensor ({}).",
                debugName(),
                *hidden_size_attribute,
                getHiddenSize());
  }
}

int64_t BaseOnnxRNNOp::getMaxSeqLength() const {
  int seq_length_index = 0;
  return inShape(getInputInIndex())[seq_length_index];
}

int64_t BaseOnnxRNNOp::getBatchSize() const {
  int batch_size_index = 1;
  return inShape(getInputInIndex())[batch_size_index];
}

int64_t BaseOnnxRNNOp::getInputSize() const {
  int input_size_index = 2;
  return inShape(getInputInIndex())[input_size_index];
}

int64_t BaseOnnxRNNOp::getHiddenSize() const {
  int hidden_size_index = 2;
  return inShape(getRecurrenceWeightsInIndex())[hidden_size_index];
}

int64_t BaseOnnxRNNOp::getNumDirections() const { return 1; }

bool BaseOnnxRNNOp::hasBiasesInput() const {
  return input->hasIndex(getBiasesInIndex());
}

bool BaseOnnxRNNOp::hasInitialHInput() const {
  return input->hasIndex(getInitialHInIndex());
}

bool BaseOnnxRNNOp::hasSeqLenInput() const {
  return input->hasIndex(getSequenceLensInIndex());
}

std::set<InIndex> BaseOnnxRNNOp::optionalInputs() const {
  return {getSequenceLensInIndex(), getInitialHInIndex(), getBiasesInIndex()};
}

void BaseOnnxRNNOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("hidden_size", hidden_size_attribute);
}

BaseOnnxRNNGradOp::BaseOnnxRNNGradOp(const OperatorIdentifier &_opid,
                                     const BaseOnnxRNNOp &fwd_op)
    : Op(_opid, fwd_op.getSettings()), hasBiasesInput(fwd_op.hasBiasesInput()),
      hasInitialHInput(fwd_op.hasInitialHInput()),
      batch_size(fwd_op.getBatchSize()), input_size(fwd_op.getInputSize()),
      max_seq_length(fwd_op.getMaxSeqLength()),
      hidden_size(fwd_op.getHiddenSize()),
      fwdInputInInfo(fwd_op.inInfo(BaseOnnxRNNOp::getInputInIndex())),
      fwdInputWeightsInInfo(
          fwd_op.inInfo(BaseOnnxRNNOp::getInputWeightsInIndex())),
      fwdRecurrenceWeightsInInfo(
          fwd_op.inInfo(BaseOnnxRNNOp::getRecurrenceWeightsInIndex())),
      fwdBiasInInfo(getBiasInInfo(fwd_op)),
      fwdInitialHInInfo(getInitialHInInfo(fwd_op)) {}

nonstd::optional<TensorInfo>
BaseOnnxRNNGradOp::getBiasInInfo(const BaseOnnxRNNOp &fwd_op) {
  if (hasBiasesInput) {
    return fwd_op.inInfo(BaseOnnxRNNOp::getBiasesInIndex());
  }
  return nonstd::nullopt;
}

nonstd::optional<TensorInfo>
BaseOnnxRNNGradOp::getInitialHInInfo(const BaseOnnxRNNOp &fwd_op) {
  if (hasInitialHInput) {
    return fwd_op.inInfo(BaseOnnxRNNOp::getInitialHInIndex());
  }
  return nonstd::nullopt;
}

bool BaseOnnxRNNGradOp::hasLastHiddenStateGradInput() const {
  return input->hasIndex(getLastHiddenStateGradInIndex());
}
bool BaseOnnxRNNGradOp::hasFullHiddenStateGradInput() const {
  return input->hasIndex(getFullHiddenStateGradInIndex());
}

void BaseOnnxRNNGradOp::setup() {
  outInfo(getInputOutIndex())             = fwdInputInInfo;
  outInfo(getInputWeightsOutIndex())      = fwdInputWeightsInInfo;
  outInfo(getRecurrenceWeightsOutIndex()) = fwdRecurrenceWeightsInInfo;
  if (hasBiasesInput) {
    outInfo(getBiasesOutIndex()) = fwdBiasInInfo.value();
  }
  if (hasInitialHInput) {
    outInfo(getInitialHOutIndex()) = fwdInitialHInInfo.value();
  }
}

void BaseOnnxRNNGradOp::populateInInfo() {
  inInfoMapping = {
      // X
      {getInputInIndex(), BaseOnnxRNNOp::getInputInIndex(), GradOpInType::In},
      // full_hidden_state;
      {getFullHiddenStateInIndex(),
       BaseOnnxRNNOp::getFullHiddenStateOutIndex(),
       GradOpInType::Out},
      // last_hidden_state grad
      {getLastHiddenStateGradInIndex(),
       BaseOnnxRNNOp::getLastHiddenStateOutIndex(),
       GradOpInType::GradOut},
      // full_hidden_state grad
      {getFullHiddenStateGradInIndex(),
       BaseOnnxRNNOp::getFullHiddenStateOutIndex(),
       GradOpInType::GradOut}};

  if (input->hasIndex(getSequenceLensInIndex())) {
    // sequence_lens
    inInfoMapping.push_back({getSequenceLensInIndex(),
                             BaseOnnxRNNOp::getSequenceLensInIndex(),
                             GradOpInType::In});
  }
}

const std::vector<GradInOutMapper> &BaseOnnxRNNGradOp::gradInputInfo() const {
  return inInfoMapping;
}

const std::map<int, int> &BaseOnnxRNNGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getInputOutIndex(), BaseOnnxRNNOp::getInputInIndex()},
      {getInputWeightsOutIndex(), BaseOnnxRNNOp::getInputWeightsInIndex()},
      {getRecurrenceWeightsOutIndex(),
       BaseOnnxRNNOp::getRecurrenceWeightsInIndex()},
      {getBiasesOutIndex(), BaseOnnxRNNOp::getBiasesInIndex()},
      {getInitialHOutIndex(), BaseOnnxRNNOp::getInitialHInIndex()}};

  return outInfo;
}

} // namespace popart
