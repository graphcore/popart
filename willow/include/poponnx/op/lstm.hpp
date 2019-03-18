#ifndef GUARD_NEURALNET_LSTM_HPP
#define GUARD_NEURALNET_LSTM_HPP

#include <poponnx/op.hpp>

#include <boost/optional.hpp>

namespace poponnx {

class LSTMOp : public Op {
public:
  LSTMOp(const OperatorIdentifier &_opid,
         boost::optional<int64_t> hidden_size,
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
  bool hasInitialCInput() const;

  void appendAttributes(OpSerialiserBase &) const override;

  bool isTraining() const;

  static InIndex getInputInIndex() { return 0; }
  static InIndex getWeightsInIndex() { return 1; }
  static InIndex getRecurrenceInIndex() { return 2; }
  static InIndex getBiasInIndex() { return 3; }
  static InIndex getSequenceLensInIndex() { return 4; }
  static InIndex getInitialHInIndex() { return 5; }
  static InIndex getInitialCInIndex() { return 6; }
  static InIndex getPeepholeInIndex() { return 7; }

  static OutIndex getOutputOutIndex() { return 0; }
  static OutIndex getHiddenStateOutIndex() { return 1; }
  static OutIndex getCellStateOutIndex() { return 2; }

  static OutIndex getInitStateOutputPassThroughIndex() { return 3; }
  static OutIndex getInitStateCellStatePassThroughIndex() { return 4; }
  static OutIndex getIntermediatesPassThroughIndex() { return 5; }
  static OutIndex getInputWeightsPassThroughIndex() { return 6; }
  static OutIndex getOutputWeightsPassThroughIndex() { return 7; }
  static OutIndex getBiasesPassThroughIndex() { return 8; }
  static OutIndex getInputPassThroughIndex() { return 9; }
  static OutIndex getOutputPassThroughIndex() { return 10; }

  // T7504
  bool supportsCaching() override { return false; }

private:
  void createPassThroughOutput(const TensorId &new_id,
                               OutIndex pass_through_index,
                               const TensorInfo &out_info);
  static int getNumIntermediates() { return 6; }

  boost::optional<int64_t> hidden_size_attribute;
};

class LSTMGradOp : public Op {
public:
  LSTMGradOp(const LSTMOp &);
  void setup() final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  const LSTMOp &getForwardOp() const;

  static InIndex getInitStateOutputInIndex() { return 0; }
  static InIndex getInitStateCellStateInIndex() { return 1; }
  static InIndex getIntermediatesInIndex() { return 2; }
  static InIndex getInputWeightsInIndex() { return 3; }
  static InIndex getOutputWeightsInIndex() { return 4; }
  static InIndex getBiasesInIndex() { return 5; }
  static InIndex getInputInIndex() { return 6; }
  static InIndex getOutputInIndex() { return 7; }

  static InIndex getCellStateOutputGradInIndex() { return 8; }
  static InIndex getHiddenStateOutputGradInIndex() { return 9; }
  static InIndex getOutputGradInIndex() { return 10; }

  static OutIndex getInputOutIndex() { return 0; }
  static OutIndex getWeightsOutIndex() { return 1; }
  static OutIndex getRecurrenceOutIndex() { return 2; }
  static OutIndex getBiasOutIndex() { return 3; }
  static OutIndex getInitialHOutIndex() { return 4; }
  static OutIndex getInitialCOutIndex() { return 5; }

private:
  const LSTMOp &forward_op;
  mutable std::map<int, int> out_info;
};

} // namespace poponnx

#endif
