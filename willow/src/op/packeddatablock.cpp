#include <onnx/onnx_pb.h>

#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/packeddatablock.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

PackedDataBlockOp::PackedDataBlockOp(
    const OperatorIdentifier &opid_,
    const std::vector<int64_t> &maxSequenceLengths_,
    int64_t resultSize_,
    int64_t callbackBatchSize_,
    Graph &callback_,
    const Op::Settings &settings_)
    : SubgraphOp(opid_, settings_), maxSequenceLengths(maxSequenceLengths_),
      resultSize(resultSize_), callbackBatchSize(callbackBatchSize_),
      callback(callback_) {}

std::unique_ptr<Op> PackedDataBlockOp::clone() const {
  return std::make_unique<PackedDataBlockOp>(*this);
}

std::vector<std::unique_ptr<Op>> PackedDataBlockOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  return upops;
}

void PackedDataBlockOp::setup() {
  auto graphOutId    = callback.get().getOutputIds().at(0);
  auto &graphOutInfo = callback.get().getTensors().get(graphOutId)->info;

  Shape resultShape{resultSize};
  for (int i = 2; i < graphOutInfo.shape().size(); i++) {
    resultShape.push_back(graphOutInfo.shape().at(i));
  }

  outInfo(0) = TensorInfo{graphOutInfo.dataType(), resultShape};
}

void PackedDataBlockOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  SubgraphOp::appendOutlineAttributes(os);
  os.appendAttribute("maxSequenceLengths", maxSequenceLengths);
  os.appendAttribute("resultSize", resultSize);
  os.appendAttribute("callbackBatchSize", callbackBatchSize);
  os.appendAttribute("callback", callback.get().id.str());
}

InIndex PackedDataBlockOp::subgraphInToOpInIndex(InIndex index) const {
  throw error("PackedDataBlockOp should have been replaced using "
              "PackedDataBlock pattern.");
}
InIndex PackedDataBlockOp::opInToSubgraphInIndex(InIndex index) const {
  throw error("PackedDataBlockOp should have been replaced using "
              "PackedDataBlock pattern.");
}

OutIndex PackedDataBlockOp::subgraphOutToOpOutIndex(OutIndex index) const {
  throw error("PackedDataBlockOp should have been replaced using "
              "PackedDataBlock pattern.");
}
OutIndex PackedDataBlockOp::opOutToSubgraphOutIndex(OutIndex index) const {
  throw error("PackedDataBlockOp should have been replaced using "
              "PackedDataBlock pattern.");
}

Graph &PackedDataBlockOp::getCalledGraph() const { return callback; }

void PackedDataBlockOp::setCalledGraph(Graph &g) { callback = g; }

int64_t PackedDataBlockOp::numCallbackInputs() { return (input->n() - 2) / 3; }

int64_t PackedDataBlockOp::getCallbackIterations() {
  auto offsetsSize = inShape(1).at(0);
  return offsetsSize / callbackBatchSize;
}

int64_t PackedDataBlockOp::numDataInputs() { return (input->n() - 2) / 3; }

std::vector<PackedSequences> PackedDataBlockOp::getPackedInputs() {
  std::vector<PackedSequences> result;
  for (int i = 0; i < numDataInputs(); i++) {
    auto data    = inTensor(i * 3);
    auto offsets = inTensor(i * 3 + 1);
    auto lengths = inTensor(i * 3 + 2);
    PackedSequences x{data, offsets, lengths};
    result.push_back(x);
  }
  return result;
}

PackedSequences PackedDataBlockOp::getPackedOutput() {
  return {outTensor(0), inTensor(input->n() - 2), inTensor(input->n() - 1)};
}

std::vector<TensorInfo> PackedDataBlockOp::callbackSequenceInInfos() {
  std::vector<TensorInfo> result;
  for (int i = 0; i < numDataInputs(); i++) {
    auto shape  = inShape(dataIndex(i));
    shape.at(0) = maxSequenceLengths.at(i) * callbackBatchSize;
    result.push_back({inInfo(dataIndex(i)).dataType(), shape});
  }
  return result;
}

namespace {

static OpDefinition packedDataBlockOpDef({OpDefinition::Inputs({}),
                                          OpDefinition::Outputs({}),
                                          OpDefinition::Attributes({})});

static OpCreator<PackedDataBlockOp> packeddatablock_OpCreator(
    OpDefinitions({{Onnx::CustomOperators::PackedDataBlock,
                    packedDataBlockOpDef}}),
    [](const OpCreatorInfo &info) -> std::unique_ptr<Op> {
      auto &attr  = info.attributes;
      auto &graph = info.settings.graph.get();
      auto &ir    = graph.getIr();

      int64_t resultSize = attr.getAttribute<Attributes::Int>("resultSize");
      int64_t callbackBatchSize =
          attr.getAttribute<Attributes::Int>("callbackBatchSize");
      std::vector<int64_t> maxSequenceLengths =
          attr.getAttribute<Attributes::Ints>("maxSequenceLengths");

      ONNX_NAMESPACE::GraphProto callback =
          info.attributes.getAttribute<Attributes::Graph>("callback");

      Graph *callbackGraph;
      if (ir.hasGraph(callback.name())) {
        callbackGraph = &ir.getGraph(callback.name());
      } else {
        callbackGraph = &ir.createGraph(callback.name());

        auto inputs        = info.getInputIds();
        int64_t num_inputs = (inputs.size() - 2) / 3;

        for (int64_t input_index = 0; input_index < num_inputs; input_index++) {
          auto tid    = inputs.at(input_index * 3);
          auto tensor = graph.getTensors().get(tid);
          auto &tinfo = tensor->info;

          auto maxSequenceLength = maxSequenceLengths.at(input_index);
          Shape inputShape{callbackBatchSize, maxSequenceLength};
          for (int64_t dim_index = 1; dim_index < tinfo.shape().size();
               dim_index++) {
            inputShape.push_back(tinfo.shape().at(dim_index));
          }

          auto scopedId =
              callbackGraph->addScope(callback.input(input_index).name());
          callbackGraph->addInput(scopedId, {tinfo.dataType(), inputShape});
        }

        callbackGraph->constructFromOnnxGraph(callback);
        for (auto &output : callback.output()) {
          auto scopedId = callbackGraph->addScope(output.name());
          callbackGraph->markAsOutput(scopedId);
        }
      }

      return std::make_unique<PackedDataBlockOp>(info.opid,
                                                 maxSequenceLengths,
                                                 resultSize,
                                                 callbackBatchSize,
                                                 *callbackGraph,
                                                 info.settings);
    } // namespace
    ,
    true);

} // namespace
} // namespace popart
