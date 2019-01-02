#include <vector>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/lstm.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

LSTMOp::LSTMOp(const OperatorIdentifier &_opid,
               Ir *_ir,
               const std::string &name,
               const Attributes &_attr)
    : Op(_opid, _ir, name, _attr) {}

std::unique_ptr<Op> LSTMOp::clone() const { return make_unique<LSTMOp>(*this); }

std::vector<std::unique_ptr<Op>> LSTMOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  return upops;
}

void LSTMOp::setup() {
  if (input->hasIndex(getPeepholeInIndex())) {
    throw error("Poponnx does not support peephole connections");
  }
  if (input->hasIndex(getSequenceLensInIndex())) {
    logging::ir::warn("Lstm optional input `sequence_lens' will be ignored");
  }

  auto seq_length     = getSeqLength();
  auto num_directions = getNumDirections();
  auto batch_size     = getBatchSize();
  auto data_type      = inInfo(getInputInIndex()).data_type();
  auto hidden_size    = getHiddenSize();

  Shape y_shape{seq_length, num_directions, batch_size, hidden_size};

  outInfo(getOutputOutIndex()) = {data_type, y_shape};

  Shape yhc_shape{num_directions, batch_size, hidden_size};
  outInfo(getHiddenStateOutIndex()) = {data_type, yhc_shape};
  outInfo(getCellStateOutIndex())   = {data_type, yhc_shape};
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

namespace {

static OpCreator<LSTMOp> lstmOpCreator(
    Onnx::Operators::LSTM,
    [](const OperatorIdentifier &opid,
       Ir *ir,
       const std::string &name = "",
       const Attributes &attr  = {}) -> std::unique_ptr<Op> {
      return std::unique_ptr<LSTMOp>(new LSTMOp(opid, ir, name, attr));
    },
    true);

} // namespace

} // namespace poponnx
