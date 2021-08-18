// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GRU_HPP
#define GUARD_NEURALNET_GRU_HPP

#include <popart/op.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {

class GRUOp : public Op {
public:
  GRUOp(const OperatorIdentifier &_opid,
        nonstd::optional<int64_t> hidden_size,
        const std::string direction,
        bool linear_before_reset,
        const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  unsigned getNumChannels() const;

  int64_t getSeqLength() const;
  int64_t getBatchSize() const;
  int64_t getInputSize() const;
  int64_t getNumDirections() const;
  int64_t getHiddenSize() const;

  bool hasBiasInput() const;
  bool hasInitialHInput() const;
  bool hasOutput(OutIndex) const;

  std::set<InIndex> optionalInputs() const final;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  bool isTraining() const;

  static InIndex getInputInIndex() { return 0; }
  static InIndex getWeightsInIndex() { return 1; }
  static InIndex getRecurrenceInIndex() { return 2; }
  static InIndex getBiasInIndex() { return 3; }
  static InIndex getSequenceLensInIndex() { return 4; }
  static InIndex getInitialHInIndex() { return 5; }

  static OutIndex getOutputOutIndex() { return 0; }
  static OutIndex getHiddenStateOutIndex() { return 1; }

  static OutIndex getInitStateOutputPassThroughIndex() { return 2; }
  static OutIndex getIntermediatesPassThroughIndex() { return 3; }
  static OutIndex getInputWeightsPassThroughIndex() { return 4; }
  static OutIndex getOutputWeightsPassThroughIndex() { return 5; }
  static OutIndex getBiasesPassThroughIndex() { return 6; }
  static OutIndex getInputPassThroughIndex() { return 7; }
  static OutIndex getOutputPassThroughIndex() { return 8; }

  // TODO: T20922 : make this outlineable similar to LSTM
  bool isOutlineable() const override { return false; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }
  std::string getDirectionAttribute() const { return direction_attribute; }
  void setDirection(const std::string &direction) {
    direction_attribute = direction;
  }

  nonstd::optional<int64_t> getHiddenSizeAttribute() const {
    return hidden_size_attribute;
  }

  int getLinearBeforeResetAttribute() const {
    return linear_before_reset_attribute;
  }

  view::Regions aliases(InIndex, OutIndex) const final;
  void growAliasModel(AliasModel &m) const override { growAliasModelMulti(m); }

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

private:
  void maybeCreatePassThroughOutput(const TensorId &new_id,
                                    OutIndex pass_through_index,
                                    const TensorInfo &out_info);
  int getNumIntermediates() { return 3 + (linear_before_reset_attribute != 0); }
  int getNumBiases() { return 3 * (1 + (linear_before_reset_attribute != 0)); }
  void trySetOutInfo(OutIndex, const TensorInfo &);

  nonstd::optional<int64_t> hidden_size_attribute;
  std::string direction_attribute   = "forward";
  int linear_before_reset_attribute = 0;
};

class GRUGradOp : public Op {
public:
  GRUGradOp(const GRUOp &);
  std::unique_ptr<Op> clone() const final;
  void setup() final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  const GRUOp &getForwardOp() const;

  bool hasHiddenStateGradInput() const;

  std::set<InIndex> optionalInputs() const final;

  static InIndex getInitStateOutputInIndex() { return 0; }
  static InIndex getIntermediatesInIndex() { return 1; }
  static InIndex getInputWeightsInIndex() { return 2; }
  static InIndex getOutputWeightsInIndex() { return 3; }
  static InIndex getBiasesInIndex() { return 4; }
  static InIndex getInputInIndex() { return 5; }
  static InIndex getOutputInIndex() { return 6; }

  static InIndex getHiddenStateOutputGradInIndex() { return 7; }
  static InIndex getOutputGradInIndex() { return 8; }

  static OutIndex getInputOutIndex() { return 0; }
  static OutIndex getWeightsOutIndex() { return 1; }
  static OutIndex getRecurrenceOutIndex() { return 2; }
  static OutIndex getBiasOutIndex() { return 3; }
  static OutIndex getInitialHOutIndex() { return 4; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

private:
  const GRUOp &forward_op;
};

} // namespace popart

#endif
