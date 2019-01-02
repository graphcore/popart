#ifndef GUARD_NEURALNET_LSTM_HPP
#define GUARD_NEURALNET_LSTM_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class LSTMOp : public Op {
public:
  LSTMOp(const OperatorIdentifier &_opid,
         Ir *_ir,
         const std::string &name = "",
         const Attributes &_attr = {});
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
};

} // namespace poponnx

#endif
