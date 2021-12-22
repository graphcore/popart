// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RNN_HPP
#define GUARD_NEURALNET_RNN_HPP

#include <popart/op.hpp>
#include <popart/op/lstmutil.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {

/**
 * This op applies a single-layer Elman RNN with a non-linearity to a batch of
 * input sequences.
 *
 * For each batch element, the following output is computed:
 * \f[
 *   h_t = f(W x_t + b_x + R h_{t-1} + b_h)
 * \f]
 * where:
 * - \f$f\f$ is a supported nonlinearity function
 * - \f$W\f$ is the input weight
 * - \f$x_t\f$ is the t'th element of the input sequence
 * - \f$R\f$ is the recurrence weight matrix
 * - \f$h_{t-1}\f$ is the previous output sequence element. \f$h_0\f$ can be
 * provided by the user
 * - \f$b_x\f$ and \f$b_h\f$ are the input and recurrence biases respectively
 *
 * The op outputs the full sequence \f$h_1, h_2, ...\f$, as well as the last
 * element of the sequence.
 *
 * If the biases or \f$h_0\f$ are not set, they are considered to be 0 and not
 * trained (are treated as constant 0s in the model).
 */
class RNNOp : public Op {
public:
  RNNOp(const OperatorIdentifier &_opid,
        ActivationFunction activation,
        nonstd::optional<int64_t> hidden_size,
        const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  int64_t getSeqLength() const;
  int64_t getBatchSize() const;
  int64_t getInputSize() const;
  // this is always 1 while direction attribute isn't supported
  int64_t getNumDirections() const;
  // size of the hidden weights that hold the state
  int64_t getHiddenSize() const;
  // checks that the ONNX hidden_size attribute matches up with the input tensor
  // shapes
  void checkHiddenSize() const;

  // bias is an optional input, this checks if it is present
  bool hasBiasInput() const;
  // initialH is an optional input, this checks if it is present
  bool hasInitialHInput() const;

  std::set<InIndex> optionalInputs() const final;

  void appendOutlineAttributes(OpSerialiserBase &) const override;
  int getInBatchAxis(InIndex) const override;
  int getOutBatchAxis(OutIndex) const override;

  static InIndex getInputInIndex() {
    return 0;
  } // X, [seq_length, batch_size, input_size]
  static InIndex getInputWeightsInIndex() {
    return 1;
  } // W, [num_directions, hidden_size, input_size], multiply with X_i
  static InIndex getRecurrenceWeightsInIndex() {
    return 2;
  } // R, [num_directions, hidden_size, hidden_size], multiply with H_i
  static InIndex getBiasInIndex() {
    return 3;
  } // B, optional, [num_directions, 2*hidden_size], 0 if not specified
  static InIndex getSequenceLensInIndex() {
    return 4;
  } // sequence_lens, optional, [batch_size]
  static InIndex getInitialHInIndex() {
    return 5;
  } // initial_H, optional, [num_directions, batch_size, hidden_size]

  static OutIndex getFullOutputOutIndex() {
    return 0;
  } // Y, optional, [seq_length, num_directions, batch_size, hidden_size]
  static OutIndex getLastOutputOutIndex() {
    return 1;
  } // Y_h, optional, [num_directions, batch_size, hidden_size]. Y_h = Y[-1]

  bool isOutlineable() const override { return true; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  virtual std::string getName() const final { return "RNN"; }

  const ActivationFunction activation_attribute;

private:
  const nonstd::optional<int64_t> hidden_size_attribute;
};

/**
 * Gradient operator for RNNOp
 */
class RNNGradOp : public Op {
public:
  RNNGradOp(const RNNOp &);
  std::unique_ptr<Op> clone() const final;
  void setup() final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  bool hasLastOutputGradInput() const;
  bool hasFullOutputGradInput() const;

  std::set<InIndex> optionalInputs() const final;

  static InIndex getInitialHInIndex() { return 0; }
  static InIndex getInputWeightsInIndex() { return 1; }
  static InIndex getRecurrenceWeightsInIndex() { return 2; }
  static InIndex getBiasInIndex() { return 3; }
  static InIndex getInputInIndex() { return 4; }
  static InIndex getFullOutputInIndex() { return 5; }

  static InIndex getLastOutputGradInIndex() { return 6; }
  static InIndex getFullOutputGradInIndex() { return 7; }

  static OutIndex getInputOutIndex() { return 0; }
  static OutIndex getInputWeightsOutIndex() { return 1; }
  static OutIndex getRecurrenceWeightsOutIndex() { return 2; }
  static OutIndex getBiasOutIndex() { return 3; }
  static OutIndex getInitialHOutIndex() { return 4; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  const bool hasBiasInput;
  const bool hasInitialHInput;
  const int64_t batch_size;
  const int64_t input_size;
  const int64_t seq_length;
  const int64_t hidden_size;
  const ActivationFunction activation_attribute;

private:
  // conditionally get fwdBiasInInfo and fwdInitialHInInfo
  nonstd::optional<TensorInfo> getBiasInInfo(const RNNOp &fwd_op);
  nonstd::optional<TensorInfo> getInitialHInInfo(const RNNOp &fwd_op);

  // these are used in setup
  const TensorInfo fwdInputInInfo;
  const TensorInfo fwdInputWeightsInInfo;
  const TensorInfo fwdRecurrenceWeightsInInfo;
  // might not exist if bias or initialH not provided
  const nonstd::optional<TensorInfo> fwdBiasInInfo;
  const nonstd::optional<TensorInfo> fwdInitialHInInfo;
};

} // namespace popart

#endif
