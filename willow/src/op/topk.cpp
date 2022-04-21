// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/topk.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

TopKOp::TopKOp(const OperatorIdentifier &opid_,
               int64_t K_,
               int64_t axis_,
               bool largest_,
               bool sorted_,
               const Op::Settings &settings_)
    : BaseSortOp(opid_, axis_, settings_), K(K_), largest(largest_),
      sorted(sorted_) {}

std::unique_ptr<Op> TopKOp::clone() const {
  return std::make_unique<TopKOp>(*this);
}

void TopKOp::setup() {

  validateAxis();

  auto shape = inShape(getInIndex());
  if (shape.at(getAxis()) < getK()) {
    throw error("Cannot take top-{} on dim of size {}, invalid Op {}",
                getK(),
                getAxis(),
                str());
  }

  shape[getAxis()] = getK();

  outInfo(getIndicesOutIndex()) = TensorInfo(DataType::INT32, shape);
  outInfo(getValuesOutIndex()) =
      TensorInfo(inInfo(getInIndex()).dataType(), shape);
}

void TopKOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  BaseSortOp::appendOutlineAttributes(os);

  if (opid.version == 1) {
    os.appendAttribute("K", K);
  } else {
    // Append the determined K so that this op may be used
    // in outlining
    os.appendAttribute("_K", K);
  }
}

int64_t TopKOp::getK() const { return K; }

std::vector<std::unique_ptr<Op>> TopKOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.push_back(std::make_unique<TopKGradOp>(*this));
  return result;
}

TopKGradOp::TopKGradOp(const TopKOp &topk)
    : Op(Onnx::GradOperators::TopKGrad, topk.getSettings()),
      axis(topk.getAxis()), gradOutInfo(topk.inInfo(BaseSortOp::getInIndex())) {
}

std::unique_ptr<Op> TopKGradOp::clone() const {
  return std::make_unique<TopKGradOp>(*this);
}

const std::vector<GradInOutMapper> &TopKGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      // gradient of the TopK values output:
      {gradInIndex(), TopKOp::getValuesOutIndex(), GradOpInType::GradOut},
      // The indices output of the TopK Op:
      {indicesInIndex(), TopKOp::getIndicesOutIndex(), GradOpInType::Out}};
  return inInfo;
}

const std::map<int, int> &TopKGradOp::gradOutToNonGradIn() const {

  static const std::map<int, int> outInfo = {
      {gradOutIndex(), BaseSortOp::getInIndex()}};

  return outInfo;
}

void TopKGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axis", axis);
}

int64_t TopKGradOp::getAxis() const { return axis; }

const TensorInfo &TopKGradOp::getGradOutInfo() const { return gradOutInfo; }

void TopKGradOp::setup() { outInfo(gradOutIndex()) = gradOutInfo; }

namespace {
Op *topKFactory(const OpCreatorInfo &info, Graph &graph) {
  // largest is optional
  bool largest = checkedIntToBool(
      info.attributes.getAttribute<Attributes::Int>("largest", 1));
  // sorted is optional
  bool sorted = checkedIntToBool(
      info.attributes.getAttribute<Attributes::Int>("sorted", 1));

  // axis is optional
  int64_t axis = info.attributes.getAttribute<Attributes::Int>("axis", -1);

  int64_t K;
  if (info.opid.version == 1) {
    // k is required, so has no default value.
    K = info.attributes.getAttribute<Attributes::Int>("k");
  } else if (info.opid.version == 10 || info.opid.version == 11) {
    int kInIndex = 1;
    K            = info.getInputScalarValue<int64_t>(kInIndex);
  } else {
    throw error("Unsupported operator version {} for topK", info.opid.version);
  }

  Op *op = graph.createOp<TopKOp>(
      info.opid, K, axis, largest, sorted, info.settings);

  // Connect only the first input.
  op->connectInTensor(TopKOp::getInIndex(), info.getInputIds().at(0));
  op->createAndConnectOutTensor(TopKOp::getValuesOutIndex(),
                                info.getOutputIds().at(0));
  op->createAndConnectOutTensor(TopKOp::getIndicesOutIndex(),
                                info.getOutputIds().at(1));

  return op;
}

static OpDefinition::DataTypes T_V1 = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition::DataTypes T = {DataType::UINT8,
                                    DataType::UINT16,
                                    DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT8,
                                    DataType::INT16,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};
static OpDefinition::DataTypes K = {DataType::INT64};

static OpDefinition
    topKOpV1Def({OpDefinition::Inputs({{"X", T_V1}}),
                 OpDefinition::Outputs({{"Values", T}, {"Indicies", K}}),
                 OpDefinition::Attributes({
                     {"axis", {"*"}},
                     {"k", {"*"}},
                 })});

static OpDefinition
    topKOpDef({OpDefinition::Inputs({
                   {"X", T},
                   {"K", K, true},
               }),
               OpDefinition::Outputs({{"Values", T}, {"Indicies", K}}),
               OpDefinition::Attributes({
                   {"axis", {"*"}},
               })});

static OpCreator<TopKOp>
    TopKOpCreator(OpDefinitions({
                      {Onnx::Operators::TopK_1, topKOpV1Def},
                      {Onnx::Operators::TopK_10, topKOpDef},
                      {Onnx::Operators::TopK_11, topKOpDef},
                  }),
                  topKFactory,
                  true);
} // namespace

} // namespace popart
