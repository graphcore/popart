#include <spdlog/fmt/fmt.h>
#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/lstm.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/lstmx.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>

#include <popnn/Lstm.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>

namespace poponnx {
namespace popx {

LSTMOpx::LSTMOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<LSTMOp>(op, Onnx::Operators::LSTM);
}

// Only create an intermediate tensor if it is consumed or used as a anchor
std::unique_ptr<poplar::Tensor> LSTMOpx::createIntermediate() const {
  if (outTensor(LSTMOp::getOutputOutIndex())->consumers.getTotal() > 0) {
    return make_unique<poplar::Tensor>();
  }

  auto anchors = op_p->pir->getDataFlow().anchors();
  if (std::find(anchors.begin(),
                anchors.end(),
                outId(LSTMOp::getOutputOutIndex())) != anchors.end()) {
    return make_unique<poplar::Tensor>();
  }

  return std::unique_ptr<poplar::Tensor>(nullptr);
}

void LSTMOpx::grow(poplar::program::Sequence &prog) const {
  growBias(prog);

  auto init_state = getInitialState();
  prepareInitialState(init_state, prog);

  auto intermediate = createIntermediate();
  poplar::Tensor output, cell_state;
  std::tie(output, cell_state) =
      popnn::lstm::lstmFwd(graph(),
                           createLSTMParams(),
                           init_state,
                           get(inId(LSTMOp::getInputInIndex())),
                           *weights,
                           intermediate.get(),
                           prog,
                           idStr(),
                           dv_p->lstmOptions,
                           &dv_p->matmulCache);

  auto lstm_op            = getLSTMOp();
  auto batch_size         = static_cast<unsigned>(lstm_op->getBatchSize());
  unsigned hidden_size    = static_cast<unsigned>(lstm_op->getHiddenSize());
  unsigned num_directions = static_cast<unsigned>(lstm_op->getNumDirections());

  if (intermediate) {
    unsigned num_fwd_intermediates =
        static_cast<unsigned>(intermediate->shape()[1]);

    insert(outId(LSTMOp::getOutputOutIndex()),
           intermediate->slice(
               num_fwd_intermediates - 1, num_fwd_intermediates, 1));
  }

  insert(outId(LSTMOp::getHiddenStateOutIndex()),
         output.reshape({num_directions, batch_size, hidden_size}));
  insert(outId(LSTMOp::getCellStateOutIndex()),
         cell_state.reshape({num_directions, batch_size, hidden_size}));
}

void LSTMOpx::growBias(poplar::program::Sequence &prog) const {
  // bias in onnx is shape [num_directions, 8 * hidden_size]
  // bias in poplibs is [4, hidden_size]
  auto lstm_op         = getLSTMOp();
  unsigned hidden_size = static_cast<unsigned>(lstm_op->getHiddenSize());

  auto biases = reshapePoplibWeightsForOnnx(getLSTMWeights().biases, false);

  if (lstm_op->hasBiasInput()) {
    auto bias_input = get(inId(LSTMOp::getBiasInIndex()));

    poplar::program::Copy copyProg(bias_input.slice(0, 4 * hidden_size, 1),
                                   biases);
    prog.add(copyProg);

    popops::mapInPlace(graph(),
                       popops::expr::BinaryOpType::ADD,
                       biases,
                       bias_input.slice(4 * hidden_size, 8 * hidden_size, 1),
                       prog,
                       idStr());
  } else {
    popops::zero(graph(), biases, prog, idStr());
  }
}

bool LSTMOpx::canCreateInput(InIndex index) const {
  if (index == LSTMOp::getInputInIndex() ||
      index == LSTMOp::getWeightsInIndex() ||
      index == LSTMOp::getRecurrenceInIndex() ||
      index == LSTMOp::getInitialHInIndex() ||
      index == LSTMOp::getInitialCInIndex()) {
    return true;
  } else {
    return false;
  }
}

poplar::Tensor LSTMOpx::createInput(InIndex index) const {
  if (index == LSTMOp::getInputInIndex()) {
    return createLSTMInput();
  } else if (index == LSTMOp::getWeightsInIndex()) {
    auto inputWeights = getLSTMWeights().inputWeights;
    return reshapePoplibWeightsForOnnx(inputWeights, true);
  } else if (index == LSTMOp::getRecurrenceInIndex()) {
    auto outputWeights = getLSTMWeights().outputWeights;
    return reshapePoplibWeightsForOnnx(outputWeights, true);
  } else if (index == LSTMOp::getInitialCInIndex()) {
    auto lstm_op = getLSTMOp();

    unsigned batch_size  = static_cast<unsigned>(lstm_op->getBatchSize());
    unsigned hidden_size = static_cast<unsigned>(lstm_op->getHiddenSize());
    unsigned num_directions =
        static_cast<unsigned>(lstm_op->getNumDirections());

    auto init_c = getInitialState().cellState;
    return init_c.reshape({num_directions, batch_size, hidden_size});
  } else if (index == LSTMOp::getInitialHInIndex()) {
    return getInitialState().output;
  } else {
    auto msg = fmt::format("LSTMOpx::createInput is not supported for index {}",
                           index);
    throw error(msg);
  }
}

poplar::Tensor
LSTMOpx::reshapePoplibWeightsForOnnx(poplar::Tensor poplib_weights,
                                     bool transpose) {
  // ONNX expects input weights in shape [num_directions, 4*hidden_size, K]
  // where
  //   num_directions is always 1 for poponnx
  //   and K is either input_size or hidden_size, for the inputWeights or
  //   outputWeights respectivly
  // and order is W[iofc]
  //
  // poplibs expects weights in shape [4, K, hidden_size]
  // and order is W[fico]
  std::vector<poplar::Interval> intervals{{0, 1}, {1, 2}, {2, 3}, {3, 4}};
  auto slices = poplib_weights.slices(intervals, 0);

  if (transpose) {
    for (int i = 0; i < slices.size(); i++) {
      slices[i] = slices[i].dimShuffle({0, 2, 1});
    }
  }

  auto wf = slices[0];
  auto wi = slices[1];
  auto wc = slices[2];
  auto wo = slices[3];

  return poplar::concat({wi, wo, wf, wc}, 1);
}

poplar::Tensor LSTMOpx::createLSTMInput() const {
  auto lstm_params = createLSTMParams();
  auto options     = dv_p->lstmOptions;
  auto cache       = &dv_p->matmulCache;

  return popnn::lstm::createInput(
      graph(), lstm_params, idStr(), options, cache);
}

popnn::lstm::LstmState LSTMOpx::getInitialState() const {
  if (!initial_state) {
    auto options     = dv_p->lstmOptions;
    auto cache       = &dv_p->matmulCache;
    auto lstm_params = createLSTMParams();

    initial_state =
        createInitialState(graph(), lstm_params, idStr(), options, cache);
  }

  return *initial_state;
}

popnn::lstm::LstmWeights LSTMOpx::getLSTMWeights() const {
  if (!weights) {
    auto lstm_params = createLSTMParams();
    auto options     = dv_p->lstmOptions;
    auto cache       = &dv_p->matmulCache;

    weights = createWeights(graph(), lstm_params, idStr(), options, cache);
  }

  return *weights;
}

popnn::lstm::LstmParams LSTMOpx::createLSTMParams() const {
  auto lstm_op = getLSTMOp();

  auto in_info         = inInfo(LSTMOp::getInputInIndex());
  auto seq_length      = static_cast<unsigned>(lstm_op->getSeqLength());
  auto batch_size      = static_cast<unsigned>(lstm_op->getBatchSize());
  unsigned input_size  = static_cast<unsigned>(lstm_op->getInputSize());
  unsigned hidden_size = static_cast<unsigned>(lstm_op->getHiddenSize());

  auto params = popnn::lstm::LstmParams(
      popType(in_info), batch_size, seq_length, {input_size, hidden_size});
  params.outputFullSequence = false;
  return params;
}

std::vector<TensorId> LSTMOpx::mustExistBeforeCreate(InIndex) const {
  return {};
}

void LSTMOpx::prepareInitialState(popnn::lstm::LstmState &init_state,
                                  poplar::program::Sequence &prog) const {
  auto hasInitC = op_p->input->hasIndex(LSTMOp::getInitialCInIndex());
  auto hasInitH = op_p->input->hasIndex(LSTMOp::getInitialHInIndex());

  if (!hasInitC && !hasInitH) {
    zeroInitialState(graph(), init_state, prog, idStr());
  } else if (!hasInitC) {
    popops::zero(graph(), init_state.cellState, prog, idStr());
  } else if (!hasInitH) {
    popops::zero(graph(), init_state.output, prog, idStr());
  }
}

LSTMOp *LSTMOpx::getLSTMOp() const { return dynamic_cast<LSTMOp *>(op_p); }

namespace {
OpxCreator<LSTMOpx> lstmOpxCreator(Onnx::Operators::LSTM);
} // namespace

} // namespace popx
} // namespace poponnx
